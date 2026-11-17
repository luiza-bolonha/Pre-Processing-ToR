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
    garantir_diretorios,
    gerar_tor,
    calcular_suporte_otimo,
    separar_e_salvar_clusters,
    salvar_visualizacoes_comuns,
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
    garantir_diretorios()

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
    pre_result, txt_database = psm.pre_processing_database(input_file)
    print(f" - Total tweets processed: {len(pre_result)}")

    # Calculate optimal support if not specified or target_nodes passed
    if min_support is None or target_nodes is not None:
        min_support = calcular_suporte_otimo(
            dados_processados=pre_result,
            alg=alg,
            target_words=target_words,
            min_nodes=min_nodes,
            target_nodes=target_nodes,
            min_support=min_support,
            verbose=True
        )

    identificador = f"{file_name_base}_{alg}_{min_support}"

    # 2. Build TOR
    print(f"\n[2/6] Generating Trie of Rules (TOR) with support={min_support}...")
    output_graphml = os.path.join("outputs", f"{identificador}.graphml")
    tor = gerar_tor(pre_result, alg=alg, min_support=min_support, output_graphml=output_graphml)

    # 3. Render vector PDF
    print("\n[3/6] Drawing graph and saving PDF...")
    output_pdf = os.path.join("outputs", f"{identificador}_grafo.pdf")
    output_png = os.path.join("outputs", f"{identificador}_grafo.png")
    G = nx.read_graphml(output_graphml)
    pf.plotar_grafo(G, output_pdf=output_pdf, output_png=output_png, show=plot)

    # 4. Cluster Partitioning
    print("\n[4/6] Separating central core from peripheral clusters...")
    separar_e_salvar_clusters(output_graphml, identificador)

    # 5. Extract Hierarchical Clusters, Map Tweets, and Generate AI Prompt
    print("\n[5/6] Extracting clusters and mapping tweets...")
    all_clusters = ator.extrair_clusters(output_graphml, show_results=False)
    print(f" - Total hierarchical clusters identified: {len(all_clusters)}")

    tweets_por_cluster = ator.tweets_clusters(
        clusters=all_clusters,
        textos_tokenizados=pre_result,
        textos_originais=txt_database
    )

    # Save CSV per cluster
    pasta_csv = "outputs_clusters_csv"
    os.makedirs(pasta_csv, exist_ok=True)
    for c in tweets_por_cluster:
        if c["n_tweets"] == 0:
            continue
        nome = c["raiz"].replace(" ", "_")[:30]
        df_cluster = pd.DataFrame(c["tweets"])
        caminho_csv = os.path.join(pasta_csv, f"{identificador}_cluster_{c['cluster_idx']}_{nome}.csv")
        df_cluster.to_csv(caminho_csv, index=False)

    print(f" - Individual cluster CSV files saved to: {pasta_csv}/")

    # AI Prompt generation
    if all_clusters and 1 <= cluster_ia <= len(all_clusters):
        print(f"\n - Generating LLM Synthesis Prompt for Cluster {cluster_ia}:")
        prompt = ator.prompt_IA(all_clusters[cluster_ia - 1])
        print("-" * 50)
        print(prompt)
        print("-" * 50)

    # 6. Auxiliary Visualizations
    print("\n[6/6] Generating auxiliary visualizations...")
    pre_simples, _ = psm.pre_processing_database(file_path=input_file, simples=True)
    salvar_visualizacoes_comuns(
        pre_simples,
        identificador,
        pasta_ranking="ranking_palavras",
        plotar_frequencia=plot
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
