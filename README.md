# PEGA

**Peptide Evolution via Genetic Algorithm**

[![CI](https://github.com/fcabezasmera/PEGA/actions/workflows/ci.yml/badge.svg)](https://github.com/fcabezasmera/PEGA/actions)
[![PyPI](https://img.shields.io/pypi/v/pega-amp)](https://pypi.org/project/pega-amp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status: alpha — active development.**  Core scoring pipeline is functional.  Genetic algorithm operators are under construction.

PEGA integrates up to ten published antimicrobial peptide (AMP) predictors into a unified ensemble scoring framework.  Individual predictor scores are combined through ensemble methods to produce robust AMP probability estimates from a FASTA input file.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fcabezasmera/PEGA.git
cd PEGA
```

### 2. Create the main environment

`pega_env` is the mother environment from which all PEGA commands are run.
It includes Python 3.12, TensorFlow 2.17, PyTorch 2.5.1, modlAMP, amPEPpy, and R 4.4.

```bash
conda env create -f envs/pega_env.yml
conda activate pega_env
pip install -e .
```

### 3. Create predictor-specific environments (optional)

Some predictors require isolated conda environments due to dependency conflicts.
PEGA calls them internally via `conda run` — you do not need to activate them manually.

```bash
# Macrel (predictor 6)
conda env create -f envs/macrel_env.yml

# AMPlify (predictors 7 & 8) — requires Python 3.6 + TensorFlow 1.10, do not upgrade
conda env create -f envs/amplify_env.yml
```

Or use the built-in setup command (from within `pega_env`):

```bash
conda activate pega_env
pega setup                    # create all environments + install ampir
pega setup --status           # check status without installing
pega setup --envs macrel_env  # create only one environment
```

### 4. Install the ampir R package (optional)

ampir is installed inside the R bundled in `pega_env`:

```bash
conda activate pega_env
pega setup --envs pega_env    # skip if already created
# installs ampir via Rscript inside pega_env
```

Or manually:

```bash
conda activate pega_env
Rscript -e 'install.packages("ampir", repos="https://cloud.r-project.org")'
```

---

## Quick start

```bash
conda activate pega_env

pega list                                          # show available predictors
pega score --fasta sequences.fasta                 # score with all available predictors
pega score --fasta sequences.fasta --out results.tsv
pega score --fasta sequences.fasta --predictors ampnet modlamp_rf
```

---

## Predictors

| ID | Name | Method | Requires |
|----|------|--------|----------|
| 1 | ampnet | CNN (TensorFlow) | pega_env |
| 2 | ampep | Random Forest | pega_env |
| 3 | ampir_mature | Logistic regression | pega_env + ampir R package |
| 4 | ampir_precursor | Logistic regression | pega_env + ampir R package |
| 5 | amp_cg | ESM-2 Transformer (PyTorch) | pega_env |
| 6 | macrel | SVM | macrel_env |
| 7 | amplify_balanced | Deep learning | amplify_env |
| 8 | amplify_imbalanced | Deep learning | amplify_env |
| 9 | modlamp_rf | Random Forest | pega_env |
| 10 | modlamp_svm | SVM | pega_env |

---

## Project structure

```
PEGA/
├── envs/
│   ├── pega_env.yml        # main environment (mother)
│   ├── macrel_env.yml      # macrel predictor
│   └── amplify_env.yml     # AMPlify predictor
├── pega/
│   ├── __init__.py
│   ├── base.py             # BasePredictor abstract class
│   ├── registry.py         # predictor auto-discovery
│   ├── utils.py            # calculate_scores() orchestrator
│   ├── ensemble.py         # ensemble methods
│   ├── cli.py              # command-line interface
│   ├── setup_envs.py       # environment setup utilities
│   ├── download_models.py  # model weight downloader
│   ├── predictors/         # one module per predictor
│   ├── operators/          # GA operators (in development)
│   ├── population/         # population management (in development)
│   ├── preprocess/         # sequence preprocessing (in development)
│   ├── engine/             # GA engine (in development)
│   └── models/             # pre-trained weights (not tracked by git)
└── pyproject.toml
```

---

## Citation

If you use PEGA in your research, please cite:

```bibtex
@software{pega2025,
  author = {fcabezasmera},
  title  = {PEGA: Peptide Evolution via Genetic Algorithm},
  year   = {2025},
  url    = {https://github.com/fcabezasmera/PEGA}
}
```

Please also cite the original publication for each predictor used.

---

## License

MIT — see [LICENSE](LICENSE).
