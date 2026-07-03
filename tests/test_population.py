"""
tests/test_population.py
========================
Tests for pega.population.
Run with: python tests/test_population.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Mock calculate_scores so we don't need the real predictors ──────────────
from Bio import SeqIO

def _mock_calculate_scores(fasta_path, predictor_names=None, jobs=1, validate=False, **_kwargs):
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    n = len(records)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "seq_name":     [r.id for r in records],
        "AMPnet_score": rng.uniform(0.1, 0.9, n),
        "AMP_CG_score": rng.uniform(0.1, 0.9, n),
        "Macrel_score": rng.uniform(0.1, 0.9, n),
    })

import pega.utils as _utils

from pega.population import (
    generate_random, generate_from_seed, generate_from_fasta,
    score_sequences, compute_fitness, calculate_statistics,
    log_generation, to_fasta, from_fasta,
)

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
ok = fail = 0

def check(cond: bool, msg: str) -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {msg}")
    else:
        fail += 1
        print(f"  [FAIL] {msg}")



def main() -> int:
    """Run the full manual check suite and return a process exit code."""
    global ok, fail
    ok = fail = 0

    # Patch calculate_scores for the duration of this run only, so the mock
    # never leaks into other test modules sharing the same pytest process.
    _original_calculate_scores = _utils.calculate_scores
    _utils.calculate_scores = _mock_calculate_scores
    try:
        return _run_checks()
    finally:
        _utils.calculate_scores = _original_calculate_scores


def _run_checks() -> int:
    global ok, fail

    # ── generate_random ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  generate_random")
    print("═"*60)
    pop = generate_random(50, 20)
    check(len(pop) == 50,                                     "50 sequences generated")
    check(all(len(s) == 20 for s in pop),                     "all length 20")
    check(all(set(s).issubset(CANONICAL) for s in pop),       "all canonical AAs")
    check(len(set(pop)) == 50,                                 "all unique (allow_duplicates=False)")
    pop_dup = generate_random(10, 5, allow_duplicates=True)
    check(len(pop_dup) == 10,                                  "allow_duplicates=True: 10 seqs")


    # ── generate_from_seed ───────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  generate_from_seed")
    print("═"*60)
    seed = "ACDEFGHIKLMNPQRS"   # 16 aa
    for strat in ["blosum62", "pam30", "pam250", "property", "random"]:
        pop = generate_from_seed(seed, 20, 16, mutation_rate=0.2, strategy=strat)
        check(len(pop) == 20,                                      f"{strat}: 20 sequences")
        check(all(len(s) == 16 for s in pop),                      f"{strat}: all length 16")
        check(all(set(s).issubset(CANONICAL) for s in pop),        f"{strat}: canonical AAs")

    # Multi-seed
    pop = generate_from_seed(["ACDEFGHIKL", "LKIHGFEDCA"], 30, 10)
    check(len(pop) == 30,                                          "multi-seed: 30 seqs")
    check(all(len(s) == 10 for s in pop),                          "multi-seed: all length 10")

    # Seed shorter than target → expand
    pop = generate_from_seed("ACDE", 10, 15)
    check(len(pop) == 10,                                          "expansion: 10 seqs")
    check(all(len(s) == 15 for s in pop),                          "expansion: length 15")

    # Seed longer than target → trim
    pop = generate_from_seed("ACDEFGHIKLMNPQRSTVWY", 10, 8)
    check(len(pop) == 10,                                          "trimming: 10 seqs")
    check(all(len(s) == 8 for s in pop),                           "trimming: length 8")


    # ── generate_from_fasta ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  generate_from_fasta")
    print("═"*60)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for i, seq in enumerate(["ACDEFGHIKL", "LKIHGFEDCA", "MNPQRSTVWY"]):
            f.write(f">seq{i}\n{seq}\n")
        fasta_path = f.name

    pop = generate_from_fasta(fasta_path, 15, 10)
    check(len(pop) == 15,                                     "from_fasta: 15 seqs")
    check(all(len(s) == 10 for s in pop),                     "from_fasta: all length 10")
    check(all(set(s).issubset(CANONICAL) for s in pop),       "from_fasta: canonical AAs")
    os.unlink(fasta_path)


    # ── score_sequences ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  score_sequences")
    print("═"*60)
    seqs = generate_random(8, 15)
    df = score_sequences(seqs, generation_label="G_0")
    check(len(df) == 8,                                                "8 rows")
    check(list(df.columns[:3]) == ["seq_name", "seq_aa", "mean_geometric"],
          f"column order: {list(df.columns[:3])}")
    check("seq_aa" in df.columns,                                      "seq_aa present")
    check("mean_geometric" in df.columns,                              "mean_geometric present")
    score_cols = [c for c in df.columns if c.endswith("_score")]
    check(len(score_cols) == 3,                                        f"3 score cols: {score_cols}")
    check(all(df["mean_geometric"] > 0),                               "fitness > 0")
    check(all(df["mean_geometric"] <= 1.0 + 1e-9),                     "fitness ≤ 1")
    check(set(df["seq_aa"].tolist()) == set(seqs),                     "all seqs scored")


    # ── compute_fitness ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  compute_fitness")
    print("═"*60)
    df2 = df.drop(columns=["mean_geometric"])
    df2 = compute_fitness(df2)
    check("mean_geometric" in df2.columns,                             "recomputed")
    check(all(df2["mean_geometric"] > 0),                              "fitness > 0")

    # Verify mathematical correctness
    for _, row in df.iterrows():
        scores = row[score_cols].values.astype(float)
        expected = float(np.exp(np.mean(np.log(scores + 1e-9))))
        actual   = float(row["mean_geometric"])
        check(abs(actual - expected) < 1e-9,
              f"geomean correct for {row['seq_name']}: {actual:.6f} ≈ {expected:.6f}")
        break  # one row is sufficient for mathematical verification


    # ── calculate_statistics ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  calculate_statistics")
    print("═"*60)
    stats = calculate_statistics(df)
    check(len(stats) > 0,                                              f"{len(stats)} stats")
    for col in score_cols + ["mean_geometric"]:
        for stat in ("mean", "min", "max", "std"):
            check(f"{col}_{stat}" in stats,                            f"{col}_{stat} present")


    # ── log_generation ───────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  log_generation")
    print("═"*60)
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
        tsv_path = f.name
    os.unlink(tsv_path)

    log_generation(df, "G_0", tsv_path)                              # no time
    log_generation(df, "G_1", tsv_path, generation_time=1.234, new_members=5)
    log_generation(df, "G_2", tsv_path, generation_time=0.987, new_members=3)

    log_df = pd.read_csv(tsv_path, sep="\t")
    check(len(log_df) == 3,                                           "3 generations logged")
    check("generation" in log_df.columns,                             "generation column")
    check("new_members" in log_df.columns,                            "new_members column")
    check("generation_time_s" in log_df.columns,                      "generation_time_s column")
    check(pd.isna(log_df.loc[0, "generation_time_s"]),                "G_0: time = NaN (missing)")
    check(float(log_df.loc[1, "generation_time_s"]) == 1.234,         "G_1: time = 1.234")
    check(int(log_df.loc[1, "new_members"]) == 5,                     "G_1: new_members = 5")
    check(log_df.loc[0, "generation"] == "G_0",                       "generation label correct")

    # Header is written only once (check row count stays at 3 after re-reading)
    log_generation(df, "G_3", tsv_path, generation_time=0.5)
    log_df2 = pd.read_csv(tsv_path, sep="\t")
    check(len(log_df2) == 4,                                          "4th generation appended cleanly")
    os.unlink(tsv_path)


    # ── to_fasta / from_fasta ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  to_fasta / from_fasta")
    print("═"*60)
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
        out_path = f.name

    # from DataFrame
    to_fasta(df, out_path)
    loaded = from_fasta(out_path)
    check(len(loaded) == 8,                                           "8 sequences loaded")
    check(all(s in df["seq_aa"].values for s in loaded),              "sequences match DataFrame")

    # from list
    to_fasta(seqs, out_path, generation_label="G_1")
    loaded2 = from_fasta(out_path)
    check(len(loaded2) == len(seqs),                                  "list → FASTA → list: count")
    check(sorted(loaded2) == sorted(s.upper() for s in seqs),         "list → FASTA → list: content")
    os.unlink(out_path)


    # ── error handling ───────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  error handling")
    print("═"*60)
    for fn, args, label in [
        (generate_random,    (0, 10),           "n=0"),
        (generate_random,    (10, 0),           "length=0"),
        (generate_from_seed, (["A","B"], 1, 5), "seeds > n"),
        (score_sequences,    ([], "G_0"),       "empty sequence list"),
    ]:
        try:
            fn(*args)
            check(False, f"{label}: should raise ValueError")
        except ValueError as e:
            check(True, f"{label} → ValueError: {e}")



    print("\n" + "═"*60)
    print(f"  RESULTS: {ok} PASS  /  {fail} FAIL")
    print("═"*60)
    return 0 if fail == 0 else 1


def test_population() -> None:
    """Pytest entry point — reruns the manual suite and asserts success."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
