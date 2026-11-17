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
    garantir_diretorios,
    gerar_tor,
    calcular_suporte_otimo,
    separar_e_salvar_clusters,
    salvar_visualizacoes_comuns,
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
        "--no-paragrafos",
        action="store_true",
        help="Do not split text into paragraphs (use whole document as single transaction)."
    )
    parser.add_argument(
        "--titulo",
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
    separar_paragrafos=True,
    is_titulo=False,
    termo_concordance="20 minutos",
    plot=False
):
    """Executes the full news analysis pipeline."""
    garantir_diretorios()

    input_file = os.path.join("datasets", f"{file_name_base}.csv")
    column = "Título" if is_titulo else "Texto"

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
    pre_result, noticias = ppn.pre_processing_database(
        file_path=input_file,
        separar_paragrafos=separar_paragrafos,
        column=column
    )
    print(f" - Total transactions/records processed: {len(pre_result)}")

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

    obs = "p" if separar_paragrafos else "n"
    obs = "titulos" if is_titulo else obs
    identificador = f"{file_name_base}_{obs}_{alg}_{min_support}"

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

    # 5. Semantic Analysis
    print("\n[5/6] Analyzing context and article recommendation...")
    if termo_concordance:
        print(f" - KWIC Concordance for term: '{termo_concordance}':")
        ator.concordance(noticias, termo_concordance)

    print("\n - Computing most informative and concise article:")
    texto_recomendado = ator.recomendar_texto(noticias, output_graphml)
    print("Recommended Article:\n", texto_recomendado)

    # 6. Auxiliary Visualizations
    print("\n[6/6] Generating auxiliary visualizations...")
    pre_simples, _ = ppn.pre_processing_database(
        file_path=input_file,
        separar_paragrafos=separar_paragrafos,
        column=column,
        simples=True
    )
    salvar_visualizacoes_comuns(
        pre_simples,
        identificador,
        pasta_ranking="ranking_palavras",
        plotar_frequencia=plot
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
        separar_paragrafos=not args.no_paragrafos,
        is_titulo=args.titulo,
        termo_concordance=args.concordance,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
