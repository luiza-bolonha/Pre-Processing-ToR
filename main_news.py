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
        default=0.05,
        help="Minimum support threshold (default: 0.05)."
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
    min_support=0.05,
    separar_paragrafos=True,
    is_titulo=False,
    termo_concordance="20 minutos",
    plot=False
):
    """Executes the full news analysis pipeline."""
    garantir_diretorios()

    input_file = os.path.join("datasets", f"{file_name_base}.csv")
    column = "Título" if is_titulo else "Texto"

    obs = "p" if separar_paragrafos else "n"
    obs = "titulos" if is_titulo else obs
    identificador = f"{file_name_base}_{obs}_{alg}_{min_support}"

    print("=" * 60)
    print(f"Starting News Pipeline: {file_name_base}")
    print(f" - Input File: {input_file} | Column: {column}")
    print(f" - Algorithm: {alg} | Minimum Support: {min_support}")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/6] Preprocessing news articles...")
    pre_result, noticias = ppn.pre_processing_database(
        file_path=input_file,
        separar_paragrafos=separar_paragrafos,
        column=column
    )
    print(f" - Total transactions/records processed: {len(pre_result)}")

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
        separar_paragrafos=not args.no_paragrafos,
        is_titulo=args.titulo,
        termo_concordance=args.concordance,
        plot=args.plot
    )


if __name__ == "__main__":
    main()
