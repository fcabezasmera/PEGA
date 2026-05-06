"""
pega.preprocess.validator
=========================
FASTA sequence validation for PEGA.

Checks performed
----------------
1. File exists and is non-empty ASCII text.
2. At least one sequence is present.
3. All sequences use only the 20 canonical amino acids.
4. Warnings for sequences shorter than 10 residues (some predictors skip these).
5. Warnings for non-standard amino acids (B, J, O, U, X, Z).
6. Warnings for stop codons (*) at the end of sequences.
"""

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO

# The 20 canonical amino acids.
_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

# Ambiguous / non-canonical single-letter codes.
_AMBIGUOUS = set("BJOUXZ")

_MIN_LEN = 10  # sequences shorter than this receive NaN from some predictors


def validate_fasta(fasta_path: str | Path, verbose: bool = True) -> bool:
    """Validate a FASTA file for use with PEGA predictors.

    Parameters
    ----------
    fasta_path:
        Path to the FASTA file to validate.
    verbose:
        If ``True``, print a formatted report to stdout.

    Returns
    -------
    bool
        ``True`` if the file is valid (no critical errors).
        Warnings do not affect the return value.
    """
    path = Path(fasta_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # File-level checks
    # ------------------------------------------------------------------
    if not path.exists():
        errors.append(f"File not found: {path}")
        _print_report(path, [], errors, warnings, verbose)
        return False

    if path.stat().st_size == 0:
        errors.append("File is empty.")
        _print_report(path, [], errors, warnings, verbose)
        return False

    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        errors.append(f"Non-ASCII characters detected: {exc}")
        _print_report(path, [], errors, warnings, verbose)
        return False

    # ------------------------------------------------------------------
    # Sequence-level checks
    # ------------------------------------------------------------------
    records = list(SeqIO.parse(str(path), "fasta"))

    if not records:
        errors.append("No sequences found. Check FASTA format.")
        _print_report(path, records, errors, warnings, verbose)
        return False

    short: list[str] = []
    noncanonical: dict[str, set[str]] = {}
    ambiguous: dict[str, set[str]] = {}
    stop_codon: list[str] = []

    for rec in records:
        seq = str(rec.seq).upper()

        # Stop codon at end
        if seq.endswith("*"):
            stop_codon.append(rec.id)
            seq = seq[:-1]  # strip for further checks

        # Length
        if len(seq) < _MIN_LEN:
            short.append(rec.id)

        # Amino acid composition
        chars = set(seq)
        bad = chars - _CANONICAL - _AMBIGUOUS - {"*"}
        amb = chars & _AMBIGUOUS

        if bad:
            noncanonical[rec.id] = bad
        if amb:
            ambiguous[rec.id] = amb

    # Critical errors
    if noncanonical:
        ids = list(noncanonical)[:5]
        chars = {c for v in noncanonical.values() for c in v}
        errors.append(
            f"{len(noncanonical)} sequence(s) contain invalid characters "
            f"{sorted(chars)}: {', '.join(ids)}"
            + (" ..." if len(noncanonical) > 5 else "")
        )

    # Warnings
    if short:
        warnings.append(
            f"{len(short)} sequence(s) shorter than {_MIN_LEN} residues "
            f"(some predictors will return NaN): {', '.join(short[:5])}"
            + (" ..." if len(short) > 5 else "")
        )
    if ambiguous:
        ids = list(ambiguous)[:5]
        chars = {c for v in ambiguous.values() for c in v}
        warnings.append(
            f"{len(ambiguous)} sequence(s) contain ambiguous amino acids "
            f"{sorted(chars)}: {', '.join(ids)}"
            + (" ..." if len(ambiguous) > 5 else "")
        )
    if stop_codon:
        warnings.append(
            f"{len(stop_codon)} sequence(s) end with a stop codon (*) "
            f"that will be ignored: {', '.join(stop_codon[:5])}"
        )

    _print_report(path, records, errors, warnings, verbose)
    return len(errors) == 0


def _print_report(
    path: Path,
    records: list,
    errors: list[str],
    warnings: list[str],
    verbose: bool,
) -> None:
    if not verbose:
        return

    lengths = [len(r.seq) for r in records] if records else []
    n = len(records)

    print(f"  File       : {path.name}")
    if n:
        print(f"  Sequences  : {n}")
        print(f"  Length     : min={min(lengths)}  max={max(lengths)}  "
              f"avg={round(sum(lengths)/n, 1)}")

    if not errors and not warnings:
        print("  Status     : OK — all sequences are valid.\n")
        return

    for w in warnings:
        print(f"  [WARNING]  {w}")
    for e in errors:
        print(f"  [ERROR]    {e}")

    status = "INVALID" if errors else "OK (with warnings)"
    print(f"  Status     : {status}\n")
