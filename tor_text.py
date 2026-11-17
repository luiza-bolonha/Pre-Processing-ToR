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
import argparse
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.pipeline import executar_tor_para_texto, garantir_diretorios


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
        default="paragrafos",
        choices=["paragrafos", "frases", "linhas"],
        help="How to segment text into transactions: 'paragrafos', 'frases', or 'linhas' (default: paragrafos)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive Matplotlib plot window."
    )

    return parser.parse_args()


def obter_texto_interativo():
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE: No text or file provided.")
    print("Type or paste your text below.")
    print("Press Enter twice on an empty line to finish:")
    print("=" * 60 + "\n")

    linhas = []
    while True:
        try:
            linha = input()
            if not linha and (len(linhas) > 0 and linhas[-1] == ""):
                linhas.pop()
                break
            linhas.append(linha)
        except EOFError:
            break

    texto = "\n".join(linhas).strip()
    return texto


def main():
    args = parse_args()
    garantir_diretorios()

    texto_input = None

    if args.text:
        texto_input = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        texto_input = args.file
    else:
        texto_input = obter_texto_interativo()

    if not texto_input or not texto_input.strip():
        print("[ERROR] No text provided. Exiting.")
        sys.exit(1)

    # Determine destination PDF path
    if args.output:
        output_pdf = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join("outputs", f"tor_text_{timestamp}.pdf")

    print("\n" + "-" * 60)
    print("Executing TOR for raw text:")
    print(f" - Algorithm: {args.alg}")
    if args.min_support is not None:
        print(f" - Explicit Support: {args.min_support}")
    elif args.target_nodes is not None:
        print(f" - Target Nodes: ~{args.target_nodes}")
    else:
        print(f" - Dynamic Support Target: {args.target_words} words (min {args.min_nodes} nodes)")
    print(f" - Split Mode: {args.split}")
    print(f" - PDF Destination: {output_pdf}")
    print("-" * 60)

    try:
        tor, pdf_path = executar_tor_para_texto(
            texto_ou_arquivo=texto_input,
            alg=args.alg,
            min_support=args.min_support,
            target_words=args.target_words,
            min_nodes=args.min_nodes,
            target_nodes=args.target_nodes,
            output_pdf=output_pdf,
            output_png=args.png,
            modo_divisao=args.split,
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
