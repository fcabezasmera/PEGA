"""
pega.cli
========
Command-line interface for PEGA.

Commands
--------
pega list                     Show all registered predictors and their status.
pega score --fasta <file>     Score sequences in a FASTA file.
pega download-models          Download pre-trained model weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pega

# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

_BANNER = f"""\
  ____  _____ ____    _
 |  _ \\| ____/ ___|  / \\
 | |_) |  _|| |  _  / _ \\
 |  __/| |__| |_| |/ ___ \\
 |_|   |_____\\____/_/   \\_\\

 Peptide Evolution via Genetic Algorithm
 v{pega.__version__}  |  https://github.com/fcabezasmera/PEGA
"""

# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    """Handle: pega list"""
    from pega.registry import registry

    print(_BANNER)
    print(registry.summary())
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Handle: pega score"""
    from pega.utils import calculate_scores

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(
            f"Error: FASTA file not found: {fasta}\n"
            "Please verify the path and try again.",
            file=sys.stderr,
        )
        return 1

    predictors = args.predictors if args.predictors else None

    print(_BANNER)
    print(f"Input file  : {fasta}")
    print(f"Predictors  : {'all available' if predictors is None else ', '.join(predictors)}")
    print(f"Ensemble    : {'yes' if not args.no_ensemble else 'no'}")
    print(f"Membership  : {'yes' if args.membership else 'no'}")
    if args.out:
        print(f"Output file : {args.out}")
    print()

    try:
        df = calculate_scores(
            fasta_path=fasta,
            predictor_names=predictors,
            apply_ensemble=not args.no_ensemble,
            membership=args.membership,
            export_tsv=args.out,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(df.to_string(index=False))
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    """Handle: pega download-models"""
    from pega.download_models import download_models

    print(_BANNER)
    download_models(model_dir=args.dir, overwrite=args.overwrite)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pega",
        description=(
            "PEGA — Peptide Evolution via Genetic Algorithm\n"
            "Multi-predictor ensemble scoring for antimicrobial peptide research.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pega list\n"
            "  pega score --fasta sequences.fasta\n"
            "  pega score --fasta sequences.fasta --predictors ampnet modlamp_rf\n"
            "  pega score --fasta sequences.fasta --out results.tsv\n"
            "  pega download-models\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"pega {pega.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # -- list ----------------------------------------------------------------
    subparsers.add_parser(
        "list",
        help="Show all registered predictors and their availability status.",
        description=(
            "Display a table of all PEGA predictors, indicating which ones are\n"
            "available in the current environment and what is required to enable\n"
            "those that are not."
        ),
    )

    # -- score ---------------------------------------------------------------
    score_parser = subparsers.add_parser(
        "score",
        help="Score sequences in a FASTA file using available AMP predictors.",
        description=(
            "Run one or more AMP predictors on the sequences in a FASTA file\n"
            "and report the individual and ensemble scores.\n\n"
            "By default, all available predictors are used and ensemble summary\n"
            "columns are appended to the output."
        ),
    )
    score_parser.add_argument(
        "--fasta", "-f",
        required=True,
        metavar="FILE",
        help="Path to the input FASTA file.",
    )
    score_parser.add_argument(
        "--predictors", "-p",
        nargs="+",
        metavar="NAME",
        help=(
            "Names of the predictors to use (e.g. ampnet modlamp_rf).  "
            "Run 'pega list' to see available names.  "
            "Default: all available predictors."
        ),
    )
    score_parser.add_argument(
        "--out", "-o",
        metavar="FILE",
        help="Save results to this file in tab-separated values (TSV) format.",
    )
    score_parser.add_argument(
        "--no-ensemble",
        action="store_true",
        default=False,
        help="Do not append ensemble summary columns to the output.",
    )
    score_parser.add_argument(
        "--membership",
        action="store_true",
        default=False,
        help="Append fuzzy membership columns for each predictor score.",
    )

    # -- download-models -----------------------------------------------------
    dl_parser = subparsers.add_parser(
        "download-models",
        help="Download pre-trained model weights from the PEGA GitHub release.",
        description=(
            "Download all pre-trained model weight files required by the\n"
            "pip-installable predictors (AMPnet, AMP-CG, amPEPpy, modlAMP).\n\n"
            "Files are saved to pega/models/ by default.  Set the environment\n"
            "variable PEGA_MODEL_DIR to use a different location."
        ),
    )
    dl_parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help="Destination directory (default: pega/models/).",
    )
    dl_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-download model files even if they already exist.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HANDLERS = {
    "list": cmd_list,
    "score": cmd_score,
    "download-models": cmd_download_models,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    handler = _HANDLERS[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
