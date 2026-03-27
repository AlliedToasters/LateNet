# Fact-Checking Pipeline: LLM-as-Judge Validation

Reference implementation from `latent-lab/experiments/got_lodo/` — used to validate a 3000-row Geometry of Truth test set. Adapt and scale for LateNet.

## Overview

Validate labeled true/false statements by asking independent models "Is this statement true or false?" then dropping rows where any model disagrees with the label. Conservative but high-quality.

## Architecture

```
input dataset (statements + labels)
         |
         +---> NDIF / nnsight         (Llama 405B Instruct, logit-level True/False)
         |     +-- retry script        (NDIF is flaky, retry failures)
         |
         +---> Anthropic API           (Sonnet first pass -> Opus escalation on disputes)
         |
         v
    merge + consensus  --->  validated.parquet  (all rows + validation columns)
                        --->  clean.parquet     (contested rows dropped)
```

## Step 1: NDIF / Llama 405B Instruct (logit-level)

Uses `lmprobe.extraction.ActivationExtractor` with `backend="nnsight"` and `remote=True`. The model runs on NDIF's servers — no local GPU needed. Reads **logits** for "True" and "False" tokens instead of generating text.

```python
from lmprobe.extraction import ActivationExtractor
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-405B-Instruct"
TOKENIZER_MODEL = "meta-llama/Llama-3.1-405B"

tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
true_id = tok.encode("True", add_special_tokens=False)[0]
false_id = tok.encode("False", add_special_tokens=False)[0]

ext = ActivationExtractor(
    model_name=MODEL,
    device="cpu",
    layers=[],        # no activations — logits only
    backend="nnsight",
)

# Format as instruct chat (Llama 3.1 template)
def chat_format(stmt):
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Is this statement true or false? Answer with just 'True' or 'False'.\n\n{stmt}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# Extract logits remotely via NDIF (logits-only, top-k=10)
prompt = chat_format("The city of Paris is in France.")
logits, _mask, logits_indices = ext.extract_logits_only(
    [prompt], remote=True, logit_top_k=10,
)

last_logits = logits[0, -1, :]
last_indices = logits_indices[0, -1, :]
# Find True/False logits in the top-k
t_logit = float("-inf")
f_logit = float("-inf")
for i, vid in enumerate(last_indices.tolist()):
    if vid == true_id:
        t_logit = last_logits[i].item()
    elif vid == false_id:
        f_logit = last_logits[i].item()
model_says_true = t_logit > f_logit
logit_gap = abs(t_logit - f_logit)  # confidence signal
```

**Details:**
- Requires `NNSIGHT_API_KEY` env var
- Processes one prompt per NDIF trace (no true batching)
- NDIF is flaky — build a retry script with exponential backoff
- Outputs per row: `true_logit`, `false_logit`, `model_says_true`, `logit_gap`, `agrees_with_label`

### Retry pattern for NDIF errors

```python
for attempt in range(max_retries):
    try:
        logits, _mask, logits_indices = ext.extract_logits_only(
            [prompt], remote=True, logit_top_k=10,
        )
        # ... parse top-k logits ...
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # exponential backoff
        else:
            result["error"] = str(e)
```

## Step 2: Anthropic API — Sonnet with Opus Escalation

Two-phase approach:
1. **Phase 1**: Run every row through Sonnet (cheap, fast)
2. **Phase 2**: Rows where Sonnet *disagrees with the label* get escalated to Opus (expensive, ~1% of rows)

```python
from anthropic import Anthropic

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"

PROMPT = (
    "Is this statement true or false? "
    "Answer with exactly one word: 'True' or 'False'.\n\n"
    "{statement}"
)

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

def query_model(client, model, statement, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{
                    "role": "user",
                    "content": PROMPT.format(statement=statement),
                }],
            )
            text = response.content[0].text.strip().lower()
            if "true" in text:
                return {"says_true": True, "raw": text, "error": None}
            elif "false" in text:
                return {"says_true": False, "raw": text, "error": None}
            else:
                return {"says_true": None, "raw": text, "error": "ambiguous_response"}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"says_true": None, "raw": None, "error": str(e)}

# Phase 1: Sonnet on all rows
for idx, row in df.iterrows():
    result = query_model(client, SONNET, row["statement"])
    agrees = result["says_true"] == (row["label"] == 1)
    results[idx]["sonnet"] = {**result, "agrees": agrees}

# Phase 2: Opus on disagreements only
sonnet_disagrees = [idx for idx, r in results.items()
                    if r["sonnet"]["agrees"] is False]

for idx in sonnet_disagrees:
    result = query_model(client, OPUS, results[idx]["statement"])
    agrees = result["says_true"] == (results[idx]["label"] == 1)
    results[idx]["opus"] = {**result, "agrees": agrees}
```

**Details:**
- `max_tokens=5` — just want "True" or "False", nothing else
- Parse is forgiving: `"true" in text.lower()` handles minor variations
- Rate limit with `time.sleep(0.1)` between calls

## Step 3: Consensus and Clean Dataset

Merge both validation sources and drop any row where *any* validator disagreed:

```python
# A row is "contested" if ANY validator disagrees
contested = (
    (df["llama_agrees"] == False) |
    (df["sonnet_agrees"] == False)
)

# Clean = everything that passed all validators
clean_df = df[~contested].copy()
```

## Checkpoint/Resume Pattern

Both scripts save progress to JSON checkpoints — mandatory for long runs and flaky APIs.

```python
# After each batch
checkpoint = {
    "next_idx": batch_end,
    "results": all_results,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(CHECKPOINT_PATH, "w") as f:
    json.dump(checkpoint, f)

# On startup with --resume
if args.resume and os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)
    all_results = checkpoint["results"]
    start_idx = checkpoint["next_idx"]
```

For the Anthropic leg, a per-response JSON cache with atomic writes is even safer:

```python
def save_cache(cache):
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.rename(CACHE_PATH)  # atomic on POSIX
```

## Results from GoT Validation

| Metric | Value |
|--------|-------|
| Original rows | 3000 |
| Llama 405B agreement | 99.5% (2986/3000) |
| Claude Sonnet agreement | 99.2% (2976/3000) |
| Opus escalated | 24 rows (19 agreed, 5 disagreed) |
| Rows dropped | 35 (1.2%) |
| Clean dataset | 2965 rows (1480 true / 1485 false) |

The 35 dropped rows were: wrong labels (Southern Alps/Australia), genuinely ambiguous facts (Bern as Swiss capital), model knowledge gaps, and obscure entities.

## How to Reproduce (original experiment)

```bash
# Requires: NNSIGHT_API_KEY, ANTHROPIC_API_KEY

# 1. Validate via NDIF (Llama 405B Instruct)
python experiments/got_lodo/validate_test_set.py
python experiments/got_lodo/retry_validation_errors.py

# 2. Validate via Anthropic (Sonnet -> Opus escalation)
python experiments/got_lodo/validate_test_set_anthropic.py

# 3. Build final clean dataset
python experiments/got_lodo/build_validated_dataset.py
```

Source: `latent-lab/experiments/got_lodo/`

## Adapting for LateNet

The core pattern to reuse:

1. **Prompt**: Simple forced-choice question with single-word response
2. **NDIF leg**: `ActivationExtractor(backend="nnsight")` + `extract_logits_only(remote=True, logit_top_k=10)` for logit-level judgments from 405B without a local GPU
3. **Anthropic leg**: Sonnet first pass (cheap), Opus escalation only on disagreements (~1% of rows)
4. **Consensus**: Drop rows where any validator disagrees
5. **Checkpoint everything**: NDIF is flaky, API calls are slow — resumability is mandatory
