#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tor_text.py
===========
CLI tool to generate a Trie of Rules (TOR) and export a vector PDF graph
from any arbitrary text (inline string, text file, or interactive prompt).
* Does NOT perform cluster extraction or clustering operations. *
"""

import sys
import os
import tempfile
import argparse
from datetime import datetime

# Ensure clean temp cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.pipeline import run_tor_for_text, ensure_directories

# Portuguese split-mode names are still accepted for backwards compatibility
SPLIT_MODE_ALIASES = {"paragrafos": "paragraphs", "frases": "sentences", "linhas": "lines"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds Trie of Rules (TOR) and exports a vector PDF graph from arbitrary text (without clusters)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t", "--text",
        type=str,
        help="Direct raw text string to process."
    )
    group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to text file (.txt, .csv, etc.)."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for output PDF file (default: outputs/tor_text_<timestamp>.pdf)."
    )
    parser.add_argument(
        "--png",
        type=str,
        default=None,
        help="Optional path to also export a PNG image."
    )
    parser.add_argument(
        "-s", "--min-support",
        type=float,
        default=None,
        help="Minimum support threshold for ARM (optional: if omitted, calculated dynamically)."
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=50,
        help="Target number of words for dynamic support calculation (default: 50)."
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Minimum number of nodes required in the graph (default: 4)."
    )
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=None,
        help="Target exact number of nodes desired in graph (optional binary search)."
    )
    parser.add_argument(
        "-a", "--alg",
        type=str,
        default="FP-max",
        choices=["FP-max", "Apriori", "FP-growth"],
        help="Association Rule Mining algorithm (default: FP-max)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="paragraphs",
        choices=["paragraphs", "sentences", "lines", *SPLIT_MODE_ALIASES],
        help="How to segment text into transactions: 'paragraphs', 'sentences', or 'lines' (default: paragraphs)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive Matplotlib plot window."
    )

    return parser.parse_args()


def read_interactive_text():
    """Reads multi-line text from stdin until two consecutive empty lines."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE: No text or file provided.")
    print("Type or paste your text below.")
    print("Press Enter twice on an empty line to finish:")
    print("=" * 60 + "\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line and lines and lines[-1] == "":
            lines.pop()
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main():
    args = parse_args()
    ensure_directories()

    if args.text:
        text_input = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        text_input = args.file
    else:
        text_input = read_interactive_text()

    if not text_input or not text_input.strip():
        print("[ERROR] No text provided. Exiting.")
        sys.exit(1)

    # Determine destination PDF path
    if args.output:
        output_pdf = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join("outputs", f"tor_text_{timestamp}.pdf")

    split_mode = SPLIT_MODE_ALIASES.get(args.split, args.split)

    print("\n" + "-" * 60)
    print("Executing TOR for raw text:")
    print(f" - Algorithm: {args.alg}")
    if args.min_support is not None:
        print(f" - Explicit Support: {args.min_support}")
    elif args.target_nodes is not None:
        print(f" - Target Nodes: ~{args.target_nodes}")
    else:
        print(f" - Dynamic Support Target: {args.target_words} words (min {args.min_nodes} nodes)")
    print(f" - Split Mode: {split_mode}")
    print(f" - PDF Destination: {output_pdf}")
    print("-" * 60)

    try:
        tor, pdf_path = run_tor_for_text(
            text_or_file=text_input,
            alg=args.alg,
            min_support=args.min_support,
            target_words=args.target_words,
            min_nodes=args.min_nodes,
            target_nodes=args.target_nodes,
            output_pdf=output_pdf,
            output_png=args.png,
            split_mode=split_mode,
            show=args.show
        )

        print("\n" + "=" * 60)
        print(" [SUCCESS] Processing completed!")
        print(f" - Total nodes in TOR graph: {tor.trie.number_of_nodes()}")
        print(f" - PDF graph generated at: {os.path.abspath(pdf_path)}")
        if args.png:
            print(f" - PNG image generated at: {os.path.abspath(args.png)}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[FAILURE] An error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
