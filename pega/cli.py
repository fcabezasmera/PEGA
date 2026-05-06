"""
pega.cli
========
Command-line interface for PEGA.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pega

# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

_BANNER = f"""\

  ██████╗ ███████╗ ██████╗  █████╗   .██████╗ ██╗   ██╗
  ██╔══██╗██╔════╝██╔════╝ ██╔══██╗  ██╔══██╗╚██╗ ██╔╝
  ██████╔╝█████╗  ██║  ███╗███████║  ██████╔╝ ╚████╔╝
  ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║  ██╔═══╝   ╚██╔╝
  ██║     ███████╗╚██████╔╝██║  ██║  ██║        ██║
  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝  ╚═╝        ╚═╝

  Peptide Evolution via Genetic Algorithm  v{pega.__version__}
  https://github.com/fcabezasmera/PEGA
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suppress_warnings() -> None:
    """Suppress TensorFlow, absl, and HuggingFace informational output."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _fasta_stats(fasta_path: Path) -> dict:
    """Return basic statistics for a FASTA file."""
    from Bio import SeqIO
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    lengths = [len(r.seq) for r in records]
    return {
        "n": len(records),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }


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
        _suppress_warnings()

    from pega.utils import calculate_scores

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        return 1

    predictors = args.predictors or None

    if not args.quiet:
        print(_BANNER)
        stats = _fasta_stats(fasta)
        print(f"  Input      : {fasta.name}")
        print(f"  Sequences  : {stats['n']}  "
              f"(len: {stats['min_len']}–{stats['max_len']}, avg {stats['avg_len']})")
        print(f"  Predictors : {'all available' if predictors is None else ', '.join(predictors)}")
        print(f"  Jobs       : {args.jobs}")
        if args.out:
            print(f"  Output     : {args.out}")
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
    from pega.preprocess.validator import validate_fasta
    if not args.quiet:
        print(_BANNER)
    ok = validate_fasta(Path(args.fasta), verbose=True)
    return 0 if ok else 1


def cmd_setup(args: argparse.Namespace) -> int:
    from pega.setup_envs import print_status, setup_all
    if not args.quiet:
        print(_BANNER)
    if args.status:
        print("Environment status:\n")
        print_status()
        return 0
    setup_all(envs=args.envs or None, include_r=not args.no_r, force=args.force)
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
            "  PEGA validate --fasta sequences.fasta\n"
            "  PEGA score --fasta sequences.fasta\n"
            "  PEGA score --fasta sequences.fasta --predictors AMPnet modlAMP_RF\n"
            "  PEGA score --fasta sequences.fasta --out results.tsv --jobs 4\n"
            "  PEGA score --fasta sequences.fasta --quiet\n"
            "  PEGA setup --status\n"
            "  PEGA download-models\n"
        ),
    )
    parser.add_argument("--version", action="version",
                        version=f"PEGA.py {pega.__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # list
    lp = sub.add_parser("list", help="Show all predictors and their availability.")
    lp.add_argument("-q", "--quiet", action="store_true", default=False)

    # score
    sp = sub.add_parser("score", help="Score sequences in a FASTA file.")
    sp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    sp.add_argument("--predictors", "-p", nargs="+", metavar="NAME",
                    help="Predictor names (run 'PEGA list' for names). Default: all.")
    sp.add_argument("--out", "-o", metavar="FILE", help="Save results as TSV.")
    sp.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                    help="Parallel workers. Use -1 for all CPU threads. Default: 1.")
    sp.add_argument("-q", "--quiet", action="store_true", default=False,
                    help="Suppress banner and framework warnings.")

    # validate
    vp = sub.add_parser("validate", help="Validate a FASTA file before scoring.")
    vp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    vp.add_argument("-q", "--quiet", action="store_true", default=False)

    # setup
    ep = sub.add_parser("setup", help="Create conda environments.")
    ep.add_argument("--envs", nargs="+", metavar="NAME")
    ep.add_argument("--no-r", action="store_true", default=False)
    ep.add_argument("--force", action="store_true", default=False)
    ep.add_argument("--status", action="store_true", default=False)
    ep.add_argument("-q", "--quiet", action="store_true", default=False)

    # download-models
    dp = sub.add_parser("download-models", help="Download pre-trained model weights.")
    dp.add_argument("--dir", metavar="PATH", default=None)
    dp.add_argument("--overwrite", action="store_true", default=False)
    dp.add_argument("-q", "--quiet", action="store_true", default=False)

    return parser


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
