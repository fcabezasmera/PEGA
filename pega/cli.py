"""
pega.cli
========
Command-line interface for PEGA.

Commands
--------
pega list                     Show all registered predictors and their status.
pega score --fasta <file>     Score sequences in a FASTA file.
pega download-models          Download pre-trained model weights.
pega setup                    Create conda environments for external predictors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pega

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
    from pega.registry import registry
    print(_BANNER)
    print(registry.summary())
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from pega.utils import calculate_scores

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        return 1

    predictors = args.predictors if args.predictors else None

    print(_BANNER)
    print(f"Input file  : {fasta}")
    print(f"Predictors  : {'all available' if predictors is None else ', '.join(predictors)}")
    if args.out:
        print(f"Output file : {args.out}")
    print()

    try:
        df = calculate_scores(
            fasta_path=fasta,
            predictor_names=predictors,
            export_tsv=args.out,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(df.to_string(index=False))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from pega.setup_envs import print_status, setup_all
    print(_BANNER)
    if args.status:
        print("Environment status:\n")
        print_status()
        return 0
    setup_all(
        envs=args.envs if args.envs else None,
        include_r=not args.no_r,
        force=args.force,
    )
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
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
        description="PEGA — Peptide Evolution via Genetic Algorithm\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pega list\n"
            "  pega score --fasta sequences.fasta\n"
            "  pega score --fasta sequences.fasta --predictors ampnet modlamp_rf\n"
            "  pega score --fasta sequences.fasta --out results.tsv\n"
            "  pega setup --status\n"
            "  pega download-models\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"pega {pega.__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # -- list ----------------------------------------------------------------
    sub.add_parser("list", help="Show all predictors and their availability status.")

    # -- score ---------------------------------------------------------------
    sp = sub.add_parser("score", help="Score sequences in a FASTA file.")
    sp.add_argument("--fasta", "-f", required=True, metavar="FILE",
                    help="Path to the input FASTA file.")
    sp.add_argument("--predictors", "-p", nargs="+", metavar="NAME",
                    help="Predictor names to use. Default: all available.")
    sp.add_argument("--out", "-o", metavar="FILE",
                    help="Save results to this TSV file.")

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

    # -- download-models -----------------------------------------------------
    dp = sub.add_parser("download-models", help="Download pre-trained model weights.")
    dp.add_argument("--dir", metavar="PATH", default=None,
                    help="Destination directory (default: pega/models/).")
    dp.add_argument("--overwrite", action="store_true", default=False,
                    help="Re-download files that already exist.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_HANDLERS = {
    "list": cmd_list,
    "score": cmd_score,
    "setup": cmd_setup,
    "download-models": cmd_download_models,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(_HANDLERS[args.command](args))


if __name__ == "__main__":
    main()
