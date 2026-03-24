# Data Sources

External data dependencies for domain-specific generators.

## WordNet
- **Source:** NLTK WordNet corpus (`nltk.corpus.wordnet`)
- **Install:** `python -c "import nltk; nltk.download('wordnet')"`
- **Used by:** `WordNetGenerator`

## Geography (planned)
- **Source:** Natural Earth, GeoNames
- **Used by:** `GeographyGenerator`

## Temporal (planned)
- **Source:** Curated timeline data, historical databases
- **Used by:** `TemporalGenerator`

## Chemistry (planned)
- **Source:** Periodic table data, PubChem
- **Used by:** `ChemistryGenerator`

## Language (planned)
- **Source:** Translation dictionaries
- **Used by:** `LanguageGenerator`

## Magnitude (planned)
- **Source:** World Bank, census data, reference tables
- **Used by:** `MagnitudeGenerator`

## Authorship (planned)
- **Source:** Structured literary/scientific databases
- **Used by:** `AuthorshipGenerator`

## Biology (planned)
- **Source:** NCBI taxonomy
- **Used by:** `BiologyGenerator`
