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

### Step 2 — Main environment

```bash
conda env create -f envs/pega_env.yml
conda activate pega_env
pip install -e .
PEGA --version
PEGA list
```

### Step 3 — ampir (R package)

```bash
conda activate pega_env
conda install -n pega_env -c conda-forge \
    r-caret r-data.table r-recipes r-modelmetrics r-ipred -y
$CONDA_PREFIX/bin/Rscript -e \
    'install.packages("ampir", repos="https://cloud.r-project.org")'
```

### Step 4 — amPEPpy

```bash
conda activate pega_env
pip install git+https://github.com/tlawrence3/amPEPpy.git
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

## Installation — HPC Cluster

This section covers installation on a cluster with **conda + mamba** available
and **internet access** on login and compute nodes.

> All installation steps must be run on the **login node**.
> Never run heavy computations directly on the login node — use job scripts.

### Step 1 — Clone the repository

```bash
# Choose a location with enough space (~500 MB for envs + models)
cd $HOME   # or /scratch/$USER if HOME has quota limits
git clone https://github.com/fcabezasmera/PEGA.git
cd PEGA
```

### Step 2 — Choose a conda prefix (important on clusters)

If your `$HOME` has disk quotas, install environments in scratch or a project directory:

```bash
# Check available space
df -h $HOME
df -h /scratch/$USER   # or your cluster's scratch path

# Option A — install in HOME (default)
conda env create -f envs/pega_env.yml

# Option B — install in custom prefix (if HOME is limited)
conda env create -f envs/pega_env.yml --prefix /scratch/$USER/envs/pega_env
conda activate /scratch/$USER/envs/pega_env
```

> If using `--prefix`, replace `conda activate pega_env` with
> `conda activate /scratch/$USER/envs/pega_env` everywhere below.

### Step 3 — Install PEGA

```bash
conda activate pega_env
pip install -e $HOME/PEGA    # adjust path if you cloned elsewhere
PEGA --version
```

### Step 4 — Install ampir, amPEPpy, Macrel, AMPlify

Same as local installation (Steps 3–5 above). Run these on the **login node**:

```bash
# ampir
conda activate pega_env
conda install -n pega_env -c conda-forge \
    r-caret r-data.table r-recipes r-modelmetrics r-ipred -y
$CONDA_PREFIX/bin/Rscript -e \
    'install.packages("ampir", repos="https://cloud.r-project.org")'

# amPEPpy
pip install git+https://github.com/tlawrence3/amPEPpy.git

# Macrel
conda create --name macrel_env -c bioconda -c conda-forge macrel -y

# AMPlify
conda create -n amplify_env python=3.6 -y
conda activate amplify_env
mamba install bioconda::amplify
conda deactivate
```

### Step 5 — Model weights

```bash
conda activate pega_env
PEGA download-models
# or if using git LFS:
git lfs pull
```

### Step 6 — Verify installation

```bash
conda activate pega_env
PEGA list
```

All 10 predictors should show `[available]`.

---

### Running PEGA on the cluster

#### Interactive session (testing)

```bash
# Request an interactive node — syntax varies by cluster
# SLURM:
srun --ntasks=1 --cpus-per-task=24 --mem=32G --time=02:00:00 --pty bash

# PBS:
qsub -I -l nodes=1:ppn=24,mem=32gb,walltime=02:00:00

# Once on the compute node:
conda activate pega_env
PEGA score --fasta sequences.fasta --jobs -1 --quiet
```

#### Job script — standard scoring

Save as `pega_score.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pega_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pega_%j.log

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pega_env

# Run scoring
PEGA score \
    --fasta sequences.fasta \
    --ensembles AMP AVP AFP ABP \
    --no-individual-scores \
    --jobs -1 \
    --quiet
```

Submit:
```bash
sbatch pega_score.sh            # SLURM
qsub pega_score.sh              # PBS
bsub < pega_score.sh            # LSF
```

#### Job script — large dataset screening (1M+ sequences)

Save as `pega_screen.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pega_screen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=pega_screen_%j.log

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pega_env

PEGA screen \
    --fasta /path/to/large_dataset.fasta \
    --chunk 50000 \
    --ensembles AMP AVP AFP ABP \
    --no-individual-scores \
    --dir /scratch/$USER/pega_results/ \
    --quiet
```

> **Tip:** Use `--dir /scratch/$USER/...` to write output to scratch — home
> directories often have I/O rate limits that slow TSV writing.

#### Recommended resources by dataset size

| Sequences | `--chunk` | RAM | CPUs | Est. time (ensembles only) |
|-----------|-----------|-----|------|---------------------------|
| < 10,000 | — (use `PEGA score`) | 16 GB | 12 | < 30 min |
| 10k–100k | 10,000 | 32 GB | 24 | 2–4 h |
| 100k–1M | 50,000 | 64 GB | 24 | 8–24 h |
| > 1M | 100,000 | 128 GB | 32 | > 24 h |

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
