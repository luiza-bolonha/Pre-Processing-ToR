#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
=======
Unified command-line entry point for the Pre-Processing-ToR framework:
1. news         -> Journalistic news articles pipeline
2. social       -> Social media / tweets pipeline
3. text         -> Generates Trie of Rules (TOR) in PDF from arbitrary raw text (no clusters)
"""

import sys
import os
import argparse

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import main_news
import main_social_media
import tor_text


def main():
    parser = argparse.ArgumentParser(
        description="Pre-Processing-ToR: Text Preprocessing & Trie of Rules (TOR) Mining Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Subcommand: news
    p_news = subparsers.add_parser("news", help="Execute news articles pipeline")
    p_news.add_argument("--file", type=str, default="manifestacao_bolsonaro", help="Dataset filename in datasets/.")
    p_news.add_argument("--alg", type=str, default="FP-max", choices=["FP-max", "Apriori", "FP-growth"])
    p_news.add_argument("--min-support", type=float, default=0.05)
    p_news.add_argument("--no-paragrafos", action="store_true", help="Do not split by paragraphs")
    p_news.add_argument("--titulo", action="store_true", help="Use 'Título' column instead of 'Texto'")
    p_news.add_argument("--concordance", type=str, default="20 minutos", help="Concordance search keyword")
    p_news.add_argument("--plot", action="store_true", help="Display interactive plots")

    # Subcommand: social
    p_social = subparsers.add_parser("social", help="Execute social media (tweets) pipeline")
    p_social.add_argument("--file", type=str, default="vacinal_2025_05", help="Dataset filename in process_datasets/.")
    p_social.add_argument("--alg", type=str, default="Apriori", choices=["FP-max", "Apriori", "FP-growth"])
    p_social.add_argument("--min-support", type=float, default=0.009)
    p_social.add_argument("--cluster-ia", type=int, default=10, help="Cluster index for LLM prompt")
    p_social.add_argument("--plot", action="store_true", help="Display interactive plots")

    # Subcommand: text
    p_text = subparsers.add_parser("text", help="Build TOR and export vector PDF from raw text (no clusters)")
    g_text = p_text.add_mutually_exclusive_group()
    g_text.add_argument("-t", "--text", type=str, help="Direct inline text string.")
    g_text.add_argument("-f", "--file", type=str, help="Path to raw text file.")
    p_text.add_argument("-o", "--output", type=str, default=None, help="Destination PDF path.")
    p_text.add_argument("--png", type=str, default=None, help="Optional destination PNG path.")
    p_text.add_argument("-s", "--min-support", type=float, default=0.05)
    p_text.add_argument("-a", "--alg", type=str, default="FP-max", choices=["FP-max", "Apriori", "FP-growth"])
    p_text.add_argument("--split", type=str, default="paragrafos", choices=["paragrafos", "frases", "linhas"])
    p_text.add_argument("--show", action="store_true", help="Display interactive plot window")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "news":
        main_news.run_pipeline_news(
            file_name_base=args.file,
            alg=args.alg,
            min_support=args.min_support,
            separar_paragrafos=not args.no_paragrafos,
            is_titulo=args.titulo,
            termo_concordance=args.concordance,
            plot=args.plot
        )
    elif args.command == "social":
        main_social_media.run_pipeline_social_media(
            file_name_base=args.file,
            alg=args.alg,
            min_support=args.min_support,
            cluster_ia=args.cluster_ia,
            plot=args.plot
        )
    elif args.command == "text":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        tor_text.main()


if __name__ == "__main__":
    main()
