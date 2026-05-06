# PEGA

**Peptide Evolution via Genetic Algorithm**

[![CI](https://github.com/YOUR_USER/PEGA/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/PEGA/actions)
[![PyPI](https://img.shields.io/pypi/v/pega-amp)](https://pypi.org/project/pega-amp/)
[![Python](https://img.shields.io/pypi/pyversions/pega-amp)](https://pypi.org/project/pega-amp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status: alpha — active development.** Core scoring pipeline is functional. Genetic algorithm operators are under construction.

PEGA integrates up to ten published antimicrobial peptide (AMP) predictors into a unified ensemble scoring framework. Individual predictor scores are combined through ensemble methods to produce robust AMP probability estimates from a FASTA input file.

The tool is designed for computational biologists and does not require programming experience beyond the command line.

---

## Installation

```bash
pip install pega-amp
```

With deep-learning predictors (AMPnet, AMP_CG):

```bash
pip install "pega-amp[deep]"
```

With modlAMP predictors:

```bash
pip install "pega-amp[modlamp]"
```

From source:

```bash
git clone https://github.com/YOUR_USER/PEGA.git
cd PEGA
pip install -e .
```

---

## Quick start

```bash
# Check which predictors are available in your environment
pega list

# Score sequences
pega score --fasta sequences.fasta

# Score and save results to a file
pega score --fasta sequences.fasta --out results.tsv

# Download pre-trained model weights
pega download-models
```

---

## Predictors

| ID | Name | Method | Requires |
|----|------|--------|----------|
| 1 | ampnet | CNN (TensorFlow) | `pip install "pega-amp[deep]"` |
| 2 | ampep | Random Forest | `pip install ampep` |
| 3 | ampir_mature | Logistic regression | R + `ampir` package |
| 4 | ampir_precursor | Logistic regression | R + `ampir` package |
| 5 | amp_cg | ESM-2 Transformer (PyTorch) | `pip install "pega-amp[deep]"` |
| 6 | macrel | SVM | conda env `macrel_env` |
| 7 | amplify_balanced | Deep learning | conda env `amplify_env` |
| 8 | amplify_imbalanced | Deep learning | conda env `amplify_env` |
| 9 | modlamp_rf | Random Forest | `pip install "pega-amp[modlamp]"` |
| 10 | modlamp_svm | SVM | `pip install "pega-amp[modlamp]"` |

PEGA detects which predictors are available in your environment automatically. Use `pega list` to see the current status.

---

## External dependencies

Some predictors require tools that cannot be installed via `pip`:

**ampir** (predictors 3 & 4) — R package:
```r
install.packages("ampir")
```

**macrel** (predictor 6):
```bash
conda create -n macrel_env -c bioconda macrel
```

**AMPlify** (predictors 7 & 8):
```bash
conda create -n amplify_env -c bioconda amplify
```

---

## Project structure

```
pega/
├── __init__.py         # public API
├── base.py             # BasePredictor abstract class
├── registry.py         # predictor auto-discovery
├── utils.py            # calculate_scores() orchestrator
├── ensemble.py         # ensemble methods
├── cli.py              # command-line interface
├── predictors/         # one module per predictor
├── operators/          # genetic algorithm operators (in development)
├── population/         # population management (in development)
├── preprocess/         # sequence preprocessing (in development)
├── engine/             # GA engine (in development)
└── models/             # pre-trained weights (downloaded separately)
```

---

## Citation

If you use PEGA in your research, please cite:

```bibtex
@software{pega2025,
  author = {Your Name},
  title  = {PEGA: Peptide Evolution via Genetic Algorithm},
  year   = {2025},
  url    = {https://github.com/YOUR_USER/PEGA}
}
```

Please also cite the original publications for each predictor you use. See the documentation for the full list.

---

## License

MIT — see [LICENSE](LICENSE).
