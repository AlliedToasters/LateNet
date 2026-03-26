"""CLI: Dataset statistics (domain coverage, tier distribution, etc.).

Can read from a parquet file (--input) or from the persistent ledger (--ledger).

Usage:
    latenet-stats --input candidates.parquet
    latenet-stats --ledger               # read from data/ledger.parquet
    latenet-stats --ledger --ledger-dir data/
"""

import argparse
from pathlib import Path

import pandas as pd


def _print_basic_stats(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Print basic row/pair/generator stats."""
    print(f"=== {label} ===")
    print(f"Total rows: {len(df)}")
    print(f"  True: {(df['label'] == True).sum()}")
    print(f"  False: {(df['label'] == False).sum()}")
    print(f"  Unique pairs: {df['pair_id'].nunique()}")
    print()

    print(f"Generators ({df['generator'].nunique()}):")
    for gen, count in df['generator'].value_counts().items():
        print(f"  {gen}: {count}")
    print()

    print(f"Relation types ({df['relation_type'].nunique()}):")
    for rt, count in df['relation_type'].value_counts().items():
        print(f"  {rt}: {count}")
    print()

    print(f"Domains ({df['domain'].nunique()}):")
    for dom, count in df['domain'].value_counts().head(20).items():
        print(f"  {dom}: {count}")
    print()

    print("Difficulty distribution:")
    for diff, count in df['difficulty'].value_counts().items():
        print(f"  {diff or '(none)'}: {count}")
    print()

    if "tier" in df.columns:
        print("Tier distribution:")
        for tier, count in df['tier'].value_counts().sort_index().items():
            print(f"  Tier {tier}: {count}")
        print()


def _print_validation_stats(df: pd.DataFrame) -> None:
    """Print validation-specific stats (only if validation columns present)."""
    if "contested" not in df.columns:
        return

    contested = (df["contested"] == True).sum()  # noqa: E712
    clean = len(df) - contested
    print(f"Validation:")
    print(f"  Clean rows: {clean} ({100*clean/max(len(df),1):.1f}%)")
    print(f"  Contested rows: {contested} ({100*contested/max(len(df),1):.1f}%)")

    if "llama_agrees" in df.columns:
        llama_ok = (df["llama_agrees"] == True).sum()  # noqa: E712
        llama_err = df["llama_error"].notna().sum() if "llama_error" in df.columns else 0
        print(f"  Llama agreement: {llama_ok}/{len(df)} ({100*llama_ok/max(len(df),1):.1f}%), errors: {llama_err}")

    if "sonnet_agrees" in df.columns:
        sonnet_ok = (df["sonnet_agrees"] == True).sum()  # noqa: E712
        print(f"  Sonnet agreement: {sonnet_ok}/{len(df)} ({100*sonnet_ok/max(len(df),1):.1f}%)")

    if "opus_agrees" in df.columns:
        escalated = df["opus_says_true"].notna().sum() if "opus_says_true" in df.columns else 0
        print(f"  Opus escalations: {escalated}")
    print()


def _print_strata_table(df: pd.DataFrame) -> None:
    """Print per-stratum pair counts (clean vs disputed)."""
    if "contested" not in df.columns:
        counts = df.groupby(["generator", "difficulty"])["pair_id"].nunique()
        print("Strata (pairs):")
        for (gen, diff), n in counts.items():
            print(f"  {gen}/{diff or '(none)'}: {n}")
        print()
        return

    clean = df[df["contested"] != True]  # noqa: E712
    disputed = df[df["contested"] == True]  # noqa: E712

    clean_counts = clean.groupby(["generator", "difficulty"])["pair_id"].nunique()
    disputed_counts = disputed.groupby(["generator", "difficulty"])["pair_id"].nunique()

    all_strata = sorted(set(clean_counts.index) | set(disputed_counts.index))

    print("Strata (clean / disputed pairs):")
    for gen, diff in all_strata:
        c = clean_counts.get((gen, diff), 0)
        d = disputed_counts.get((gen, diff), 0)
        print(f"  {gen}/{diff or '(none)'}: {c} clean, {d} disputed")
    print()


def _print_provenance_stats(df: pd.DataFrame) -> None:
    """Print provenance breakdown (git hashes, batches)."""
    if "git_hash" not in df.columns:
        return

    print("Provenance:")
    print(f"  Batches: {df['batch_id'].nunique() if 'batch_id' in df.columns else 'n/a'}")
    print(f"  Git hashes: {df['git_hash'].nunique()}")
    for h, count in df['git_hash'].value_counts().items():
        dirty = ""
        if "git_dirty" in df.columns:
            dirty_rows = df[df["git_hash"] == h]["git_dirty"]
            if dirty_rows.any():
                dirty = " (dirty)"
        print(f"    {h}: {count} rows{dirty}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print dataset coverage statistics")
    parser.add_argument("--input", type=str, default=None, help="Input parquet file")
    parser.add_argument("--ledger", action="store_true", help="Read from the persistent ledger")
    parser.add_argument("--ledger-dir", type=str, default="data", help="Ledger directory")
    args = parser.parse_args()

    if args.ledger:
        from latenet.io.ledger import load_ledger
        df = load_ledger(Path(args.ledger_dir))
        if len(df) == 0:
            print("Ledger is empty. Run latenet-batch first.")
            return
        label = f"Ledger ({args.ledger_dir}/ledger.parquet)"
    elif args.input:
        df = pd.read_parquet(args.input)
        label = args.input
    else:
        parser.error("Provide --input FILE or --ledger")
        return

    _print_basic_stats(df, label)
    _print_validation_stats(df)
    _print_strata_table(df)
    _print_provenance_stats(df)


if __name__ == "__main__":
    main()
