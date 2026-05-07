<div align="center">

```
 ██████╗ ███████╗ ██████╗  █████╗     ██████╗ ██╗   ██╗
 ██╔══██╗██╔════╝██╔════╝ ██╔══██╗    ██╔══██╗╚██╗ ██╔╝
 ██████╔╝█████╗  ██║  ███╗███████║    ██████╔╝ ╚████╔╝
 ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║    ██╔═══╝   ╚██╔╝
 ██║     ███████╗╚██████╔╝██║  ██║    ██║        ██║
 ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝    ╚═╝        ╚═╝
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

PEGA.py integrates **10 published AMP predictors** into a single, parallelized scoring pipeline.
Given a FASTA file, it returns a table with one probability score per predictor per sequence,
saved automatically as a timestamped TSV file.

```bash
PEGA score --fasta sequences.fasta --jobs 8
```

```
  Input      : sequences.fasta
  Sequences  : 1,024  (len: 10–98, avg 35.2)
  Predictors : all available
  Jobs       : 8
  Output     : PEGA_results_20260506_183000.tsv
  Status     : OK — all sequences are valid.

  Running 10 predictors with 8 parallel workers...

  [ 1/10]  modlAMP SVM ✓  1.4s
  [ 2/10]  modlAMP RF  ✓  1.8s
  [ 3/10]  amPEPpy     ✓  2.1s
  ...
  [10/10]  AMPlify (balanced)  ✓  13.0s

  Saved → PEGA_results_20260506_183000.tsv  (1024 sequences × 10 predictors)
```

---

## Predictors

| ID | Name | Method | Dependency | Reference |
|----|------|--------|------------|-----------|
| 1 | AMPnet | CNN (TensorFlow) | `pega_env` | — |
| 2 | amPEPpy | Random Forest | `pega_env` | Lawrence et al. 2021 |
| 3 | ampir (mature) | SVM | `pega_env` + R | Fingerhut et al. 2021 |
| 4 | ampir (precursor) | SVM | `pega_env` + R | Fingerhut et al. 2021 |
| 5 | AMP_CG | ESM-2 Transformer (PyTorch) | `pega_env` | — |
| 6 | Macrel | SVM + physico-chemical features | `macrel_env` | Santos-Júnior et al. 2020 |
| 7 | AMPlify (balanced) | Attentive deep learning | `amplify_env` | Li et al. 2022 |
| 8 | AMPlify (imbalanced) | Attentive deep learning | `amplify_env` | Li et al. 2022 |
| 9 | modlAMP RF | Random Forest on PepCATS | `pega_env` | Müller et al. 2017 |
| 10 | modlAMP SVM | SVM on PepCATS | `pega_env` | Müller et al. 2017 |

---

## Installation

### Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| Linux (64-bit) | — | Supported platform |
| Miniconda / Anaconda | ≥ 23 | Environment management |
| mamba | ≥ 1.0 | Required for AMPlify environment |

> **Note:** PEGA.py has been tested on Ubuntu 22.04 with an x86_64 CPU.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/fcabezasmera/PEGA.git
cd PEGA
```

---

### Step 2 — Create the main environment

`pega_env` is the mother environment from which all `PEGA` commands are run.
It bundles Python 3.10, TensorFlow 2.17, PyTorch 2.5, modlAMP, amPEPpy, and R 4.4.

```bash
conda env create -f envs/pega_env.yml
conda activate pega_env
pip install -e .
```

Verify:

```bash
PEGA --version
PEGA list
```

---

### Step 3 — Install the ampir R package

R is already bundled inside `pega_env`. Install the compiled dependencies first to
avoid source-compilation failures, then install ampir from CRAN:

```bash
conda activate pega_env

conda install -n pega_env -c conda-forge \
    r-caret r-data.table r-recipes r-modelmetrics r-ipred -y

$CONDA_PREFIX/bin/Rscript -e \
    'install.packages("ampir", repos="https://cloud.r-project.org")'
```

---

### Step 4 — Install amPEPpy

amPEPpy is not on PyPI and must be installed from its GitHub repository:

```bash
conda activate pega_env
pip install git+https://github.com/tlawrence3/amPEPpy.git
```

---

### Step 5 — Create external predictor environments

Macrel and AMPlify require isolated conda environments due to dependency conflicts with
newer Python versions. PEGA calls them internally via `conda run` — you do not need to
activate them manually.

**Macrel** (predictor 6):

```bash
conda create --name macrel_env -c bioconda -c conda-forge macrel -y
```

**AMPlify** (predictors 7 & 8):

> AMPlify 2.0 requires Python 3.6 and TensorFlow 1.10.
> `mamba` is required to resolve this old dependency tree.

```bash
# Install mamba if not already available
conda install -n base -c conda-forge mamba -y

# Create the environment
conda create -n amplify_env python=3.6 -y
conda activate amplify_env
mamba install bioconda::amplify
conda deactivate
```

Alternatively, use the built-in setup script:

```bash
conda activate pega_env
bash setup_environments.sh           # create all environments
bash setup_environments.sh --status  # check status without installing
```

---

### Step 6 — Model weights

Pre-trained model weights are included in the repository via **Git LFS**.
They are downloaded automatically when you clone:

```bash
git clone https://github.com/fcabezasmera/PEGA.git
```

If you cloned without LFS (e.g. with `GIT_LFS_SKIP_SMUDGE=1`), pull the models manually:

```bash
git lfs pull
```

Files included in `pega/models/`:

```
pega/models/
├── convolutional_nn_1.h5   # AMPnet       (42 MB)
├── best_model.pth          # AMP_CG       (31 MB)
├── amPEP.model             # amPEPpy      (17 MB)
├── modlamp_RF.joblib       # modlAMP RF   (12 MB)
└── modlamp_SVM.joblib      # modlAMP SVM  ( 2 MB)
```

---

### Verifying the full installation

```bash
conda activate pega_env
PEGA list
```

Expected output:

```
   ID  Predictor                 Status         Type
  ────────────────────────────────────────────────────
    1  AMPnet                    [available]    pip
    2  amPEPpy                   [available]    pip
    3  ampir (mature)            [available]    r
    4  ampir (precursor)         [available]    r
    5  AMP_CG                    [available]    pip
    6  Macrel                    [available]    conda
    7  AMPlify (balanced)        [available]    conda
    8  AMPlify (imbalanced)      [available]    conda
    9  modlAMP RF                [available]    pip
   10  modlAMP SVM               [available]    pip
  ────────────────────────────────────────────────────
  10 of 10 predictors available.
```

---

## Usage

### Validate a FASTA file

```bash
PEGA validate --fasta sequences.fasta
```

Checks for non-ASCII characters, non-canonical amino acids, stop codons,
and sequences shorter than 10 residues.

### Score sequences

```bash
# All predictors, sequential
PEGA score --fasta sequences.fasta

# All predictors, parallel (recommended for large datasets)
PEGA score --fasta sequences.fasta --jobs 8

# Specific predictors only
PEGA score --fasta sequences.fasta --predictors ampnet amp_cg modlamp_rf

# Custom output file
PEGA score --fasta sequences.fasta --out my_results.tsv

# Suppress framework warnings
PEGA score --fasta sequences.fasta --quiet
```

> Results are always saved to a TSV file.
> Default name: `PEGA_results_YYYYMMDD_HHMMSS.tsv`

### Python API

```python
import pega

# Score with all available predictors
df = pega.score("sequences.fasta")

# Score with specific predictors
df = pega.score(
    "sequences.fasta",
    predictors=["ampnet", "amp_cg", "modlamp_rf"],
    export_tsv="results.tsv",
)

print(df.head())
```

---

## Output format

Tab-separated file with one row per sequence and one score column per predictor.
All scores are probabilities in `[0, 1]` — higher means more likely antimicrobial.

| seq_name | AMPnet_score | amPEPpy_score | ampir_mature_score | ... | modlAMP_SVM_score |
|----------|-------------|---------------|-------------------|-----|------------------|
| SEQ001 | 0.812 | 0.562 | 0.745 | ... | 0.589 |
| SEQ002 | 0.134 | 0.218 | 0.301 | ... | 0.102 |

---

## Project structure

```
PEGA/
├── envs/
│   ├── pega_env.yml           # main environment
│   ├── macrel_env.yml         # Macrel predictor
│   └── amplify_env.yml        # AMPlify predictor
├── pega/
│   ├── __init__.py            # public API
│   ├── base.py                # BasePredictor abstract class
│   ├── registry.py            # auto-discovery registry
│   ├── utils.py               # scoring orchestrator
│   ├── ensemble.py            # ensemble methods (coming soon)
│   ├── cli.py                 # PEGA command-line interface
│   ├── setup_envs.py          # environment setup utilities
│   ├── download_models.py     # model weight downloader
│   ├── predictors/            # one module per predictor
│   ├── preprocess/            # FASTA validation
│   ├── operators/             # GA operators (in development)
│   ├── population/            # population management (in development)
│   ├── engine/                # GA engine (in development)
│   └── models/                # pre-trained weights (not tracked by git)
├── tests/
│   └── data/
│       └── test.fasta
└── setup_environments.sh
```

---

## Citation

If you use PEGA.py in your research, please cite this repository and the original
publications for each predictor used:

```bibtex
@software{pega2025,
  author = {fcabezasmera},
  title  = {PEGA.py: Peptide Evolution via Genetic Algorithm},
  year   = {2025},
  url    = {https://github.com/fcabezasmera/PEGA}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
