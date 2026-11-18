#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_news.py
============
Complete analysis and rule mining (TOR) pipeline for journalistic news articles.
Can be run via command line or imported as a module.
"""

import os
import sys
import argparse
import networkx as nx

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import utils.pre_processing_news as ppn
import utils.analize_tor as ator
import utils.processing_file as pf
from utils.pipeline import (
    ensure_directories,
    build_tor,
    compute_optimal_support,
    split_and_save_clusters,
    save_common_visualizations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Trie of Rules (TOR) Pipeline for Journalistic News Articles")
    parser.add_argument(
        "--file",
        type=str,
        default="manifestacao_bolsonaro",
        help="Dataset filename located in datasets/ (without .csv extension)."
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="FP-max",
        choices=["FP-max", "Apriori", "FP-growth"],
        help="Association Rule Mining algorithm (default: FP-max)."
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="Minimum support threshold (optional: if omitted, calculated dynamically based on target words/nodes)."
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
        "--no-paragraphs", "--no-paragrafos",
        dest="no_paragraphs",
        action="store_true",
        help="Do not split text into paragraphs (use whole document as single transaction)."
    )
    parser.add_argument(
        "--title", "--titulo",
        dest="title",
        action="store_true",
        help="Use 'Título' (Title) column instead of 'Texto' (Body text)."
    )
    parser.add_argument(
        "--concordance",
        type=str,
        default="20 minutos",
        help="Keyword for KWIC context concordance analysis."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display interactive graphical plots during execution."
    )
    return parser.parse_args()


def run_pipeline_news(
    file_name_base="manifestacao_bolsonaro",
    alg="FP-max",
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    split_paragraphs=True,
    use_title=False,
    concordance_term="20 minutos",
    plot=False
):
    """Executes the full news analysis pipeline."""
    ensure_directories()

    input_file = os.path.join("datasets", f"{file_name_base}.csv")
    column = "Título" if use_title else "Texto"

    print("=" * 60)
    print(f"Starting News Pipeline: {file_name_base}")
    print(f" - Input File: {input_file} | Column: {column}")
    print(f" - Algorithm: {alg}")
    if min_support is not None:
        print(f" - Explicit Support: {min_support}")
    elif target_nodes is not None:
        print(f" - Target Nodes: ~{target_nodes}")
    else:
        print(f" - Dynamic Support Target: {target_words} words (min {min_nodes} nodes)")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/6] Preprocessing news articles...")
    transactions, news_texts = ppn.pre_processing_database(
        file_path=input_file,
        split_paragraphs=split_paragraphs,
        column=column
    )
    print(f" - Total transactions/records processed: {len(transactions)}")

    # Support is computed first because it is part of the output file names
    if min_support is None or target_nodes is not None:
        min_support = compute_optimal_support(
            transactions=transactions,
            alg=alg,
            target_words=target_words,
            min_nodes=min_nodes,
            target_nodes=target_nodes,
            min_support=min_support,
        )

    if use_title:
        mode_tag = "titulos"
    else:
        mode_tag = "p" if split_paragraphs else "n"
    run_id = f"{file_name_base}_{mode_tag}_{alg}_{min_support}"

    # 2. Build TOR
    print(f"\n[2/6] Generating Trie of Rules (TOR) with support={min_support}...")
    output_graphml = os.path.join("outputs", f"{run_id}.graphml")
    build_tor(transactions, alg=alg, min_support=min_support, output_graphml=output_graphml)

    # 3. Render vector PDF
    print("\n[3/6] Drawing graph and saving PDF...")
    output_pdf = os.path.join("outputs", f"{run_id}_grafo.pdf")
    output_png = os.path.join("outputs", f"{run_id}_grafo.png")
    G = nx.read_graphml(output_graphml)
    pf.plot_graph(G, output_pdf=output_pdf, output_png=output_png, show=plot)

    # 4. Cluster Partitioning
    print("\n[4/6] Separating central core from peripheral clusters...")
    split_and_save_clusters(output_graphml, run_id)

    # 5. Semantic Analysis
    print("\n[5/6] Analyzing context and article recommendation...")
    if concordance_term:
        print(f" - KWIC Concordance for term: '{concordance_term}':")
        ator.concordance(news_texts, concordance_term)

    print("\n - Computing most informative and concise article:")
    recommended = ator.recommend_text(news_texts, output_graphml)
    print("Recommended Article:\n", recommended)

    # 6. Auxiliary Visualizations
    print("\n[6/6] Generating auxiliary visualizations...")
    simple_transactions, _ = ppn.pre_processing_database(
        file_path=input_file,
        split_paragraphs=split_paragraphs,
        column=column,
        simple=True
    )
    save_common_visualizations(
        simple_transactions,
        run_id,
        ranking_folder="ranking_palavras",
        plot_frequency=plot
    )

    print("\n" + "=" * 60)
    print(" [SUCCESS] News Pipeline completed!")
    print("=" * 60)


def main():
    args = parse_args()
    run_pipeline_news(
        file_name_base=args.file,
        alg=args.alg,
        min_support=args.min_support,
        target_words=args.target_words,
        min_nodes=args.min_nodes,
        target_nodes=args.target_nodes,
        split_paragraphs=not args.no_paragraphs,
        use_title=args.title,
        concordance_term=args.concordance,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
