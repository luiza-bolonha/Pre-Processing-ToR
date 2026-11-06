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
        default=0.009,
        help="Minimum support threshold (default: 0.009)."
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
    min_support=0.009,
    cluster_ia=10,
    plot=False
):
    """Executes the full social media analysis pipeline."""
    garantir_diretorios()

    input_file = os.path.join("process_datasets", f"{file_name_base}.csv")
    identificador = f"{file_name_base}_{alg}_{min_support}"

    print("=" * 60)
    print(f"Starting Social Media Pipeline: {file_name_base}")
    print(f" - Input File: {input_file}")
    print(f" - Algorithm: {alg} | Minimum Support: {min_support}")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/6] Preprocessing tweets...")
    pre_result, txt_database = psm.pre_processing_database(input_file)
    print(f" - Total tweets processed: {len(pre_result)}")

    # 2. Build TOR
    print("\n[2/6] Generating Trie of Rules (TOR)...")
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
        cluster_ia=args.cluster_ia,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
