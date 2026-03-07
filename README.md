# b1-method

Domain-independent convergent derivation of canonical basis vectors from K independent sources.

## Install

```bash
pip install b1-method
```

## Quick Start

```python
from b1_method import B1Analysis

alignment = {
    "Extraversion":      ["Y", "Y", "Y", "Y", "Y", "Y"],
    "Agreeableness":     ["Y", "Y", "Y*", "Y", "Y", "Y"],
    "Conscientiousness": ["Y", "Y", "Y", "Y", "Y", "Y"],
    "Neuroticism":       ["Y", "Y", "Y*", "N*", "Y", "Y"],
    "Openness":          ["Y", "Y", "Y", "N*", "Y", "Y"],
    "Honesty-Humility":  ["N", "N", "Y", "Y*", "N", "N"],
}

result = B1Analysis(alignment, domain="Personality").run()
B1Analysis.print_report(result)
```

## CLI

```bash
b1-method run alignment.csv --sources sources.csv --domain Personality
b1-method temporal alignment.csv --sources sources.csv --domain Personality
b1-method version
```

## How It Works

Given K independent source assessments proposing competing dimensional structures for the same domain, B1 produces a tier-classified, independence-verified basis:

- **Tier 1** (count >= ceil(2K/3)): Strong convergence — confirmed basis vectors
- **Tier 2** (count >= ceil(K/3)): Partial convergence — contested candidates
- **Tier 3** (count < ceil(K/3)): Weak/non-convergent — insufficient support

The number of Tier 1 candidates is a **lower bound** on the domain's dimensionality.

## License

MIT
