"""
pega.predictors.amplify
=======================
AMPlify predictors — balanced and imbalanced models (conda).

AMPlify writes output TSV files to the current working directory.
PEGA runs it in a temporary directory and parses column 4 (0-indexed)
as the score, matching the original PEGA implementation.

Output file columns (0-indexed)
--------------------------------
0  seq_name
1  sequence
2  log-scaled score
3  AMP / non-AMP label
4  probability score   ← used by PEGA

Sharding
--------
AMPlify (TensorFlow 1.x, CPU-only) is by far PEGA's slowest predictor —
on a 10,000-sequence chunk it can take ~40x longer than the next slowest
predictor, so the other 9 finish and leave most node cores idle for the
bulk of the chunk's wall time. Each AMPlify subprocess is pinned to a
single internal thread (see ``_ENV_SINGLE_THREADED``) to avoid the
opposite failure mode — severe oversubscription when it previously tried
to grab every core while running alongside everything else. To use the
cores that would otherwise sit idle, PEGA splits the input into several
sub-batches and runs them as concurrent single-threaded AMPlify
subprocesses (see ``_run_amplify`` / ``_default_amplify_shards``).

Installation
------------
    conda create -n amplify_env python=3.6 -y
    conda activate amplify_env
    mamba install bioconda::amplify

Reference
---------
Li C. et al. (2022). AMPlify. *BMC Genomics*, 23, 77.
https://github.com/BirolLab/AMPlify
"""

from __future__ import annotations

import concurrent.futures
import functools
import glob
import math
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from pega.base import BasePredictor

_CONDA_ENV = "amplify_env"

# AMPlify's TensorFlow 1.x backend defaults its internal thread pool to the
# machine's full core count; PEGA already parallelises at a higher level
# (--jobs and, below, sharding), so each subprocess is pinned to one thread.
_ENV_SINGLE_THREADED = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}

# Below this many sequences per shard, the fixed per-invocation overhead
# (conda run + loading the TF model) dominates over the actual scoring
# time, so splitting further would only add multiple copies of that
# overhead without any real gain. ~0.235s/sequence observed in practice.
_MIN_SEQS_PER_SHARD = 500


def _default_amplify_shards() -> int:
    """Number of concurrent single-threaded AMPlify subprocesses to run.

    Override with the ``PEGA_AMPLIFY_SHARDS`` environment variable if the
    default doesn't suit your node — e.g. its core count, or how many
    other predictors you expect running concurrently alongside it.
    """
    override = os.environ.get("PEGA_AMPLIFY_SHARDS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, cpus // 4)


def _run_amplify_batch(fasta_path: str, model_type: str, work_dir: str) -> pd.DataFrame:
    """Run one AMPlify subprocess over a single FASTA file/shard."""
    score_col = f"AMPlify_{model_type}_score"

    cmd = ["conda", "run", "-n", _CONDA_ENV, "AMPlify", "-s", str(fasta_path)]
    if model_type == "imbalanced":
        cmd += ["-m", "imbalanced"]

    env = {**os.environ, **_ENV_SINGLE_THREADED}

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,           # AMPlify writes output files here
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"AMPlify ({model_type}) failed (exit {exc.returncode}).\n"
            f"{exc.stderr.decode(errors='replace')}"
        ) from exc

    # AMPlify writes: AMPlify_{model_type}_results_<timestamp>.tsv
    pattern = os.path.join(work_dir, f"AMPlify_{model_type}_results_*.tsv")
    matching = glob.glob(pattern)
    if not matching:
        # Fallback: balanced uses 'balanced' in name regardless
        pattern = os.path.join(work_dir, "AMPlify_balanced_results_*.tsv")
        matching = glob.glob(pattern)

    if not matching:
        raise FileNotFoundError(
            f"AMPlify ({model_type}) produced no output file. "
            "Check your AMPlify installation."
        )

    output_file = max(matching, key=os.path.getctime)

    # Columns (0-indexed): 0=seq_name, 1=seq, 2=log_score, 3=label, 4=prob
    raw = pd.read_csv(output_file, sep="\t", skiprows=1, header=None)
    raw[4] = pd.to_numeric(raw[4], errors="coerce")
    raw = raw[raw[4].notnull()]

    return pd.DataFrame({
        "seq_name": raw[0].astype(str).values,
        score_col: raw[4].values,
    })


def _run_amplify(fasta_path: Path, model_type: str) -> pd.DataFrame:
    """Run AMPlify, sharding the input across parallel subprocesses.

    See the module docstring's "Sharding" section for why.
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}.")

    n_shards = min(
        _default_amplify_shards(),
        max(1, len(records) // _MIN_SEQS_PER_SHARD),
    )

    with tempfile.TemporaryDirectory() as parent_dir:
        if n_shards <= 1:
            return _run_amplify_batch(
                str(fasta_path), model_type, parent_dir
            ).reset_index(drop=True)

        shard_size = math.ceil(len(records) / n_shards)
        shard_jobs = []
        for i in range(0, len(records), shard_size):
            shard_records = records[i:i + shard_size]
            shard_dir = os.path.join(parent_dir, f"shard_{i}")
            os.makedirs(shard_dir, exist_ok=True)
            shard_fasta = os.path.join(shard_dir, "shard.fasta")
            SeqIO.write(shard_records, shard_fasta, "fasta")
            shard_jobs.append((shard_fasta, shard_dir))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shard_jobs)) as executor:
            futures = [
                executor.submit(_run_amplify_batch, shard_fasta, model_type, shard_dir)
                for shard_fasta, shard_dir in shard_jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    return pd.concat(results, ignore_index=True)


class AmplifyBalancedPredictor(BasePredictor):
    """AMPlify balanced model (conda)."""

    name = "amplify_balanced"
    display_name = "AMPlify (balanced)"
    predictor_id = 7
    description = "Deep learning AMP predictor — balanced model (AMPlify / conda)."
    category = "conda"

    @classmethod
    def score_column(cls) -> str:
        return "AMPlify_balanced_score"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached — spawns a "conda run" subprocess, and availability can't
        # change mid-process, so repeated calls (e.g. once per chunk in
        # screen_sequences) would otherwise re-spawn it needlessly.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "AMPlify", "--help"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score with AMPlify balanced model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPlify_balanced_score"]``.
        """
        return _run_amplify(self._validate_fasta(fasta_path), "balanced")


class AmplifyImbalancedPredictor(BasePredictor):
    """AMPlify imbalanced model (conda)."""

    name = "amplify_imbalanced"
    display_name = "AMPlify (imbalanced)"
    predictor_id = 8
    description = "Deep learning AMP predictor — imbalanced model (AMPlify / conda)."
    category = "conda"

    @classmethod
    def score_column(cls) -> str:
        return "AMPlify_imbalanced_score"

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_available(cls) -> bool:
        # Cached — spawns a "conda run" subprocess, and availability can't
        # change mid-process, so repeated calls (e.g. once per chunk in
        # screen_sequences) would otherwise re-spawn it needlessly.
        if not cls._executable_on_path("conda"):
            return False
        result = subprocess.run(
            ["conda", "run", "-n", _CONDA_ENV, "AMPlify", "--help"],
            capture_output=True,
        )
        return result.returncode == 0

    def score(self, fasta_path: str | Path) -> pd.DataFrame:
        """Score with AMPlify imbalanced model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``["seq_name", "AMPlify_imbalanced_score"]``.
        """
        return _run_amplify(self._validate_fasta(fasta_path), "imbalanced")
