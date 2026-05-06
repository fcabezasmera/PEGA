"""
pega.cli
========
Command-line interface for PEGA.py.

Commands
--------
PEGA list                      Show all predictors and their status.
PEGA score  --fasta <file>     Score sequences in a FASTA file.
PEGA validate --fasta <file>   Validate a FASTA file before scoring.
PEGA setup  [options]          Create conda environments.
PEGA download-models           Download pre-trained model weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pega

_BANNER = f"""\
  ____  _____ ____    _
 |  _ \\| ____/ ___|  / \\
 | |_) |  _|| |  _  / _ \\
 |  __/| |__| |_| |/ ___ \\
 |_|   |_____\\____/_/   \\_\\

 Peptide Evolution via Genetic Algorithm  v{pega.__version__}
 https://github.com/fcabezasmera/PEGA
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suppress_tf_warnings() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import warnings
    warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    from pega.registry import registry
    if not args.quiet:
        print(_BANNER)
    print(registry.summary())
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if args.quiet:
        _suppress_tf_warnings()

    from pega.utils import calculate_scores

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        return 1

    predictors = args.predictors or None

    if not args.quiet:
        print(_BANNER)
        print(f"  Input     : {fasta}")
        print(f"  Predictors: {'all available' if predictors is None else ', '.join(predictors)}")
        print(f"  Jobs      : {args.jobs}")
        if args.out:
            print(f"  Output    : {args.out}")
        print()

    try:
        df = calculate_scores(
            fasta_path=fasta,
            predictor_names=predictors,
            export_tsv=args.out,
            jobs=args.jobs,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(df.to_string(index=False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a FASTA file before scoring."""
    from pega.base import BasePredictor
    from Bio import SeqIO

    fasta = Path(args.fasta)
    if not args.quiet:
        print(_BANNER)

    try:
        BasePredictor._validate_fasta(fasta)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [FAIL] {exc}", file=sys.stderr)
        return 1

    records = list(SeqIO.parse(str(fasta), "fasta"))
    short = [r.id for r in records if len(r.seq) < 10]
    nonstandard_aa = set("BJOUXZ")
    invalid = [r.id for r in records if set(str(r.seq).upper()) & nonstandard_aa]

    print(f"  [OK]   {fasta.name}")
    print(f"         {len(records)} sequences")

    if short:
        print(f"  [WARN] {len(short)} sequence(s) shorter than 10 aa "
              f"(will return NaN in some predictors): {', '.join(short[:5])}")
    if invalid:
        print(f"  [WARN] {len(invalid)} sequence(s) contain non-standard amino acids: "
              f"{', '.join(invalid[:5])}")
    if not short and not invalid:
        print("         All sequences look valid.")

    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from pega.setup_envs import print_status, setup_all
    if not args.quiet:
        print(_BANNER)
    if args.status:
        print("Environment status:\n")
        print_status()
        return 0
    setup_all(
        envs=args.envs or None,
        include_r=not args.no_r,
        force=args.force,
    )
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    from pega.download_models import download_models
    if not args.quiet:
        print(_BANNER)
    download_models(model_dir=args.dir, overwrite=args.overwrite)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PEGA",
        description="PEGA.py — Peptide Evolution via Genetic Algorithm\n"
                    "Multi-predictor scoring for antimicrobial peptide research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PEGA list\n"
            "  PEGA score --fasta sequences.fasta\n"
            "  PEGA score --fasta sequences.fasta --predictors AMPnet modlamp_rf\n"
            "  PEGA score --fasta sequences.fasta --out results.tsv --jobs 4\n"
            "  PEGA validate --fasta sequences.fasta\n"
            "  PEGA setup --status\n"
            "  PEGA download-models\n"
        ),
    )
    parser.add_argument("--version", action="version",
                        version=f"PEGA.py {pega.__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="Suppress banner and TensorFlow/HuggingFace warnings.")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # -- list ----------------------------------------------------------------
    lp = sub.add_parser("list", help="Show all predictors and their availability.")
    lp.add_argument("-q", "--quiet", action="store_true", default=False)

    # -- score ---------------------------------------------------------------
    sp = sub.add_parser(
        "score",
        help="Score sequences in a FASTA file.",
        description=(
            "Run AMP predictors on the sequences in a FASTA file.\n"
            "By default, all available predictors are used."
        ),
    )
    sp.add_argument("--fasta", "-f", required=True, metavar="FILE",
                    help="Path to the input FASTA file.")
    sp.add_argument("--predictors", "-p", nargs="+", metavar="NAME",
                    help="Predictor names to use (e.g. ampnet modlamp_rf). "
                         "Default: all available. Run 'PEGA list' for names.")
    sp.add_argument("--out", "-o", metavar="FILE",
                    help="Save results to this TSV file.")
    sp.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                    help="Number of predictors to run in parallel. "
                         "Use -1 for all CPU threads. Default: 1.")
    sp.add_argument("-q", "--quiet", action="store_true", default=False,
                    help="Suppress banner and framework warnings.")

    # -- validate ------------------------------------------------------------
    vp = sub.add_parser("validate", help="Validate a FASTA file before scoring.")
    vp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    vp.add_argument("-q", "--quiet", action="store_true", default=False)

    # -- setup ---------------------------------------------------------------
    ep = sub.add_parser("setup", help="Create conda environments for external predictors.")
    ep.add_argument("--envs", nargs="+", metavar="NAME",
                    help="Specific environments to create. Default: all.")
    ep.add_argument("--no-r", action="store_true", default=False,
                    help="Skip R / ampir installation.")
    ep.add_argument("--force", action="store_true", default=False,
                    help="Recreate environments that already exist.")
    ep.add_argument("--status", action="store_true", default=False,
                    help="Show environment status without installing.")
    ep.add_argument("-q", "--quiet", action="store_true", default=False)

    # -- download-models -----------------------------------------------------
    dp = sub.add_parser("download-models",
                        help="Download pre-trained model weights from GitHub Releases.")
    dp.add_argument("--dir", metavar="PATH", default=None,
                    help="Destination directory (default: pega/models/).")
    dp.add_argument("--overwrite", action="store_true", default=False,
                    help="Re-download files that already exist.")
    dp.add_argument("-q", "--quiet", action="store_true", default=False)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HANDLERS = {
    "list":            cmd_list,
    "score":           cmd_score,
    "validate":        cmd_validate,
    "setup":           cmd_setup,
    "download-models": cmd_download_models,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(_HANDLERS[args.command](args))


if __name__ == "__main__":
    main()
