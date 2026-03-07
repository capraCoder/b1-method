"""
b1_method.cli — Command-line interface for the B1 method.

Entry points:
    b1-method run <alignment.csv> [--sources <sources.csv>] [--domain <name>]
    b1-method temporal <alignment.csv> --sources <sources.csv> [--ground-truth <gt.json>]
    b1-method version

Designed for ``python -m b1_method`` or as a console_scripts entry point.
"""

import argparse
import json
import os
import sys

from b1_method import __version__
from b1_method.core import B1Analysis
from b1_method.temporal import TemporalSimulation, print_temporal_report
from b1_method.io import (
    load_alignment_csv,
    load_sources_csv,
    load_combined_csv,
    save_report_json,
)


def _build_parser():
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="b1-method",
        description=(
            "B1 Convergent Derivation Method: domain-independent derivation "
            "of canonical basis vectors from K independent sources."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run B1 analysis on an alignment matrix.",
        description=(
            "Load an alignment CSV (and optional sources CSV), run the full "
            "B1 pipeline (C1-C5), and print the report."
        ),
    )
    run_parser.add_argument(
        "alignment_csv",
        help="Path to the alignment matrix CSV file.",
    )
    run_parser.add_argument(
        "--sources",
        dest="sources_csv",
        default=None,
        help="Path to the sources metadata CSV file (optional).",
    )
    run_parser.add_argument(
        "--domain",
        default=None,
        help="Domain name for the report header (e.g. 'Personality').",
    )
    run_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the JSON report (optional).",
    )

    # --- temporal ---
    temporal_parser = subparsers.add_parser(
        "temporal",
        help="Run temporal holdout simulation.",
        description=(
            "Feed sources to B1 in chronological order and compare "
            "intermediate verdicts against ground-truth findings."
        ),
    )
    temporal_parser.add_argument(
        "alignment_csv",
        help="Path to the alignment matrix CSV file.",
    )
    temporal_parser.add_argument(
        "--sources",
        dest="sources_csv",
        required=True,
        help="Path to the sources metadata CSV file (required for temporal).",
    )
    temporal_parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        default=None,
        help=(
            "Path to a JSON file mapping ground-truth questions to boolean "
            "answers. If omitted, simulation runs without accuracy scoring."
        ),
    )
    temporal_parser.add_argument(
        "--domain",
        default=None,
        help="Domain name for the report header.",
    )
    temporal_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the JSON report (optional).",
    )

    # --- version ---
    subparsers.add_parser(
        "version",
        help="Print the package version.",
    )

    return parser


def _cmd_run(args):
    """Execute the 'run' subcommand."""
    # Load alignment data
    alignment, source_names = load_alignment_csv(args.alignment_csv)

    # Load sources if provided
    sources = None
    if args.sources_csv:
        sources = load_sources_csv(args.sources_csv)

    # Build and run analysis
    domain = args.domain or _guess_domain(args.alignment_csv)
    analysis = B1Analysis(
        alignment=alignment,
        sources=sources,
        domain=domain,
    )
    result = analysis.run()

    # Print report
    B1Analysis.print_report(result)

    # Save JSON if requested
    if args.output:
        save_report_json(result, args.output)
        print(f"\nReport saved to: {args.output}")

    return 0


def _cmd_temporal(args):
    """Execute the 'temporal' subcommand."""
    # Load alignment and sources
    alignment, source_names = load_alignment_csv(args.alignment_csv)
    source_meta = load_sources_csv(args.sources_csv)

    # Build temporal source dicts: merge metadata with per-source factor verdicts
    # Alignment is {candidate: [s1_val, s2_val, ...]}, sources are ordered by CSV column
    # TemporalSimulation expects each source dict to have a "factors" key
    candidates = list(alignment.keys())
    temporal_sources = []
    for i, meta in enumerate(source_meta):
        factors = {}
        for candidate in candidates:
            row = alignment[candidate]
            factors[candidate] = row[i] if i < len(row) else "N"
        temporal_sources.append({**meta, "factors": factors})

    # Load ground truth if provided
    ground_truth = {}
    if args.ground_truth:
        gt_path = os.path.expanduser(args.ground_truth)
        if not os.path.isfile(gt_path):
            print(f"Error: ground-truth file not found: {gt_path}",
                  file=sys.stderr)
            return 1
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    # Build and run simulation
    domain = args.domain or _guess_domain(args.alignment_csv)
    sim = TemporalSimulation(
        domain_name=domain,
        sources=temporal_sources,
        ground_truth=ground_truth,
        candidates=candidates,
    )
    result = sim.run()

    # Print report
    print_temporal_report(result)

    # Save JSON if requested
    if args.output:
        save_report_json(result, args.output)
        print(f"\nReport saved to: {args.output}")

    return 0


def _cmd_version():
    """Execute the 'version' subcommand."""
    print(f"b1-method {__version__}")
    return 0


def _guess_domain(filepath):
    """Guess the domain name from a file path."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Strip common prefixes/suffixes
    for prefix in ("b1_", "alignment_", "sources_"):
        if basename.lower().startswith(prefix):
            basename = basename[len(prefix):]
    return basename.replace("_", " ").title()


def main():
    """CLI entry point for the B1 method package."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "temporal":
            return _cmd_temporal(args)
        elif args.command == "version":
            return _cmd_version()
        else:
            parser.print_help()
            return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main() or 0)
