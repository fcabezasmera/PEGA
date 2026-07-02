<div align="center">

```
 ██████╗ ███████╗ ██████╗  █████╗ 
 ██╔══██╗██╔════╝██╔════╝ ██╔══██╗
 ██████╔╝█████╗  ██║  ███╗███████║
 ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║
 ██║     ███████╗╚██████╔╝██║  ██║   ██╗ █████╗ ██╗  ██╗
 ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝ ██╔══██╗╚██╗██╔╝
                                         █████╔╝  ╚███╔╝ 
                                         ██╔══╝   ██╔╝   
                                         ╚═╝      ╚═╝    
```

**Peptide Evolution via Genetic Algorithm**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/fcabezasmera/PEGA)
[![Predictors](https://img.shields.io/badge/predictors-10-teal.svg)](https://github.com/fcabezasmera/PEGA)

*Multi-predictor ensemble scoring for antimicrobial peptide research*

</div>

---

## Overview

PEGA.py integrates **10 published AMP predictors** into a single, parallelized scoring
pipeline. Given a FASTA file, it returns predictor scores and validated ensemble scores,
saved automatically as TSV files. It supports datasets from a few sequences up to
**1M+ sequences** via chunked screening mode.

---

## Predictors

| ID | Name | Method | Dependency |
|----|------|--------|------------|
| 1 | AMPnet | CNN (TensorFlow, batch) | `pega_env` |
| 2 | amPEPpy | Random Forest | `pega_env` |
| 3 | ampir (mature) | SVM | `pega_env` + R |
| 4 | ampir (precursor) | SVM | `pega_env` + R |
| 5 | AMP_CG | ESM-2 Transformer (PyTorch) | `pega_env` |
| 6 | Macrel | SVM + physico-chemical | `macrel_env` |
| 7 | AMPlify (balanced) | Attentive deep learning | `amplify_env` |
| 8 | AMPlify (imbalanced) | Attentive deep learning | `amplify_env` |
| 9 | modlAMP RF | Random Forest (PepCATS) | `pega_env` |
| 10 | modlAMP SVM | SVM (PepCATS) | `pega_env` |

## Validated Ensembles

| Ensemble | Predictors | Task | MCC | AUC |
|----------|-----------|------|-----|-----|
| `ensemble_AMP_score` | AMP_CG, ampir, AMPlify, AMPnet, modlAMP RF | Antimicrobial | 0.767 | 0.932 |
| `ensemble_AVP_score` | AMP_CG, ampir, modlAMP SVM | Antiviral | 0.456 | 0.785 |
| `ensemble_AFP_score` | AMP_CG, AMPnet, Macrel | Antifungal | 0.757 | 0.929 |
| `ensemble_ABP_score` | AMP_CG, ampir, AMPlify, AMPnet, Macrel, modlAMP RF | Antibiofilm | 0.875 | 0.975 |

---

## Installation — Local

### Requirements

- Linux 64-bit
- Miniconda or Anaconda
- mamba (recommended)
- git with Git LFS

### Step 1 — Clone

```bash
git clone https://github.com/fcabezasmera/PEGA.git
cd PEGA
```

> Model weights (~106 MB) are stored via Git LFS and downloaded automatically.
> If LFS is not available: `PEGA download-models` after installation.

> **Remote machines / SSH nodes**: the installation procedure is identical —
> run all steps on the target node after connecting via SSH.
> No scheduler-specific configuration is needed.

### Step 2 — Main environment

```bash
conda env create -f envs/pega_env.yml
conda activate pega_env
pip install -e .
PEGA --version
PEGA list
```

> The yml pins **Python 3.10** — required for TensorFlow 2.17 and modlAMP
> compatibility. Do not change this to 3.11/3.12.

### Step 3 — ampir (R package)

```bash
conda activate pega_env

# Update r-rlang first — ampir requires >=1.1.7, conda may install an older version
conda install -n pega_env -c conda-forge r-rlang -y

# Install R dependencies
conda install -n pega_env -c conda-forge \
    r-caret r-data.table r-recipes r-modelmetrics r-ipred -y

# Install ampir from CRAN
$CONDA_PREFIX/bin/Rscript -e \
    'install.packages("ampir", repos="https://cloud.r-project.org")'
```

> **If you see** `namespace 'rlang' X.X.X is already loaded, but >= 1.1.7 is required`:
> the `r-rlang` update above fixes this. If the error persists, restart the R session
> and try again.

### Step 4 — amPEPpy

```bash
conda activate pega_env
pip install git+https://github.com/tlawrence3/amPEPpy.git
```

### Step 4b — modlAMP

modlAMP **cannot** be installed via the `pega_env.yml` (its pinned
`mysql-connector-python==8.0.17` dependency does not exist on PyPI).
Install it separately after the environment is created:

```bash
conda activate pega_env
pip install modlamp
```

### Step 5 — External predictor environments

**Macrel** (predictor 6):
```bash
conda create --name macrel_env -c bioconda -c conda-forge macrel -y
```

**AMPlify** (predictors 7 & 8) — requires Python 3.6, use mamba:
```bash
conda create -n amplify_env python=3.6 -y
conda activate amplify_env
mamba install bioconda::amplify
conda deactivate
```

---

## Quick Start

```bash
conda activate pega_env

# Check predictors
PEGA list

# Validate FASTA
PEGA validate --fasta sequences.fasta

# Score (all predictors + ensembles)
PEGA score --fasta sequences.fasta --jobs -1 --quiet

# Score — ensemble columns only (faster)
PEGA score --fasta sequences.fasta \
    --ensembles AMP AVP AFP ABP \
    --no-individual-scores \
    --jobs -1 --quiet

# Large dataset screening
PEGA screen --fasta large.fasta \
    --chunk 50000 \
    --ensembles AMP AVP AFP ABP \
    --no-individual-scores \
    --quiet
```

---

## Output

### `PEGA score` → single TSV

```
PEGA_results_YYYYMMDD_HHMMSS.tsv
```

Columns: `seq_name` | `original_header` | scores | ensemble scores

### `PEGA screen` → folder with 3 files

```
PEGA_screen_YYYYMMDD_HHMMSS/
├── names.tsv       # pep_id | original_header | length
├── rejected.tsv    # original_header | length | reason
├── rejected.fasta  # FASTA of rejected sequences
└── scores.tsv      # pep_id | scores | ensemble scores
```

Join scores ↔ names on `pep_id`:

```python
import pandas as pd
names  = pd.read_csv("PEGA_screen_*/names.tsv", sep="\t")
scores = pd.read_csv("PEGA_screen_*/scores.tsv", sep="\t")
df     = scores.merge(names, on="pep_id")
```

All scores are probabilities in `[0, 1]`. Higher = more likely antimicrobial.

---

## Python API

```python
import pega

# Score with all predictors
df = pega.score("sequences.fasta")

# Score specific predictors
df = pega.score("sequences.fasta",
                predictors=["ampnet", "amp_cg", "modlamp_rf"])

# Score and export
df = pega.score("sequences.fasta", export_tsv="results.tsv")
```

---

## Project structure

```
PEGA/
├── envs/
│   ├── pega_env.yml          # main environment
│   ├── macrel_env.yml        # Macrel predictor
│   └── amplify_env.yml       # AMPlify predictor
├── pega/
│   ├── predictors/           # one module per predictor
│   ├── operators/            # GA crossover, mutation, selection
│   ├── population/           # GA population management
│   ├── preprocess/           # FASTA validation
│   ├── ensemble.py           # validated ensemble scores
│   ├── utils.py              # scoring + screen orchestration
│   └── cli.py                # PEGA CLI
├── tests/
│   └── data/test.fasta
└── pyproject.toml
```

---

## Citation

```bibtex
@software{pega2025,
  author = {fcabezasmera},
  title  = {PEGA.py: Peptide Evolution via Genetic Algorithm},
  year   = {2025},
  url    = {https://github.com/fcabezasmera/PEGA}
}
```

Please also cite the original publication for each predictor used.

---

## License

MIT — see [LICENSE](LICENSE).
