#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_social_media.py
====================
Complete analysis and rule mining (TOR) pipeline for social media data (Twitter/X).
Can be run via command line or imported as a module.
"""

import os
import sys
import argparse
import pandas as pd
import networkx as nx

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import utils.pre_processing_social_media as psm
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
    parser = argparse.ArgumentParser(description="Trie of Rules (TOR) Pipeline for Social Media / Tweets")
    parser.add_argument(
        "--file",
        type=str,
        default="vacinal_2025_05",
        help="Dataset filename located in process_datasets/ (without .csv extension)."
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="Apriori",
        choices=["FP-max", "Apriori", "FP-growth"],
        help="Association Rule Mining algorithm (default: Apriori)."
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
        "--cluster-ia",
        type=int,
        default=10,
        help="Cluster index to generate AI synthesis prompt for (default: 10)."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display interactive graphical plots during execution."
    )
    return parser.parse_args()


def run_pipeline_social_media(
    file_name_base="vacinal_2025_05",
    alg="Apriori",
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    cluster_ia=10,
    plot=False
):
    """Executes the full social media analysis pipeline."""
    ensure_directories()

    input_file = os.path.join("process_datasets", f"{file_name_base}.csv")

    print("=" * 60)
    print(f"Starting Social Media Pipeline: {file_name_base}")
    print(f" - Input File: {input_file}")
    print(f" - Algorithm: {alg}")
    if min_support is not None:
        print(f" - Explicit Support: {min_support}")
    elif target_nodes is not None:
        print(f" - Target Nodes: ~{target_nodes}")
    else:
        print(f" - Dynamic Support Target: {target_words} words (min {min_nodes} nodes)")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/6] Preprocessing tweets...")
    transactions, original_texts = psm.pre_processing_database(input_file)
    print(f" - Total tweets processed: {len(transactions)}")

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

    run_id = f"{file_name_base}_{alg}_{min_support}"

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

    # 5. Extract Hierarchical Clusters, Map Tweets, and Generate AI Prompt
    print("\n[5/6] Extracting clusters and mapping tweets...")
    all_clusters = ator.extract_clusters(output_graphml, show_results=False)
    print(f" - Total hierarchical clusters identified: {len(all_clusters)}")

    tweets_by_cluster = ator.map_texts_to_clusters(
        clusters=all_clusters,
        tokenized_texts=transactions,
        original_texts=original_texts
    )

    # Save CSV per cluster
    csv_folder = "outputs_clusters_csv"
    os.makedirs(csv_folder, exist_ok=True)
    for cluster in tweets_by_cluster:
        if cluster["n_tweets"] == 0:
            continue
        name = cluster["root"].replace(" ", "_")[:30]
        csv_path = os.path.join(csv_folder, f"{run_id}_cluster_{cluster['cluster_idx']}_{name}.csv")
        pd.DataFrame(cluster["tweets"]).to_csv(csv_path, index=False)

    print(f" - Individual cluster CSV files saved to: {csv_folder}/")

    # AI Prompt generation
    if all_clusters and 1 <= cluster_ia <= len(all_clusters):
        print(f"\n - Generating LLM Synthesis Prompt for Cluster {cluster_ia}:")
        prompt = ator.build_llm_prompt(all_clusters[cluster_ia - 1])
        print("-" * 50)
        print(prompt)
        print("-" * 50)

    # 6. Auxiliary Visualizations
    print("\n[6/6] Generating auxiliary visualizations...")
    simple_transactions, _ = psm.pre_processing_database(file_path=input_file, simple=True)
    save_common_visualizations(
        simple_transactions,
        run_id,
        ranking_folder="ranking_palavras",
        plot_frequency=plot
    )

    print("\n" + "=" * 60)
    print(" [SUCCESS] Social Media Pipeline completed!")
    print("=" * 60)


def main():
    args = parse_args()
    run_pipeline_social_media(
        file_name_base=args.file,
        alg=args.alg,
        min_support=args.min_support,
        target_words=args.target_words,
        min_nodes=args.min_nodes,
        target_nodes=args.target_nodes,
        cluster_ia=args.cluster_ia,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
