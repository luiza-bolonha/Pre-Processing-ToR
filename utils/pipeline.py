import os
import re
import csv
import pandas as pd
import networkx as nx
import nltk
import spacy
from trieofrules import trieofrules
import utils.analize_tor as ator
import utils.other_visualizations as ovi
import utils.processing_file as pf

# Ensure stopwords are available
try:
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
except Exception:
    try:
        nltk.download('stopwords', quiet=True)
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    except Exception:
        stopwords = set()

# Shared SpaCy model singleton
_nlp = None

def get_spacy_model():
    """Loads SpaCy model on demand."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('pt_core_news_lg')
        except Exception:
            _nlp = spacy.load('pt_core_news_sm')
    return _nlp

def garantir_diretorios(pastas=None):
    """Ensures default project directories exist."""
    if pastas is None:
        pastas = [
            "outputs",
            "outputs_clusters",
            "outputs_clusters_csv",
            "ranking_palavras",
            "datasets",
            "process_datasets"
        ]
    for p in pastas:
        os.makedirs(p, exist_ok=True)

def gerar_tor(dados_processados, alg="FP-max", min_support=0.05, output_graphml=None):
    """
    Generates the Trie of Rules (TOR) from tokenized transactions.
    Optionally saves the resulting graph in .graphml format.
    """
    print(f"Generating Trie of Rules (TOR) with alg='{alg}' and min_support={min_support}...")
    tor = trieofrules(data=dados_processados, alg=alg, min_support=min_support)

    if output_graphml:
        dir_name = os.path.dirname(output_graphml)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        print(f"Saving graph to: {output_graphml}")
        tor.save_graph(filename=output_graphml)

    return tor

def separar_e_salvar_clusters(graphml_path, file_name, pasta="outputs_clusters"):
    """
    Decomposes graph into central core and peripheral sub-clusters,
    saving both sub-graphs to disk.
    """
    os.makedirs(pasta, exist_ok=True)
    print("Separating central core from peripheral clusters...")
    g_central, g_perifericos = ator.separar_clusters_por_tamanho(graphml_path)

    path_central = os.path.join(pasta, f"{file_name}_cluster_central.graphml")
    path_perifericos = os.path.join(pasta, f"{file_name}_clusters_perifericos.graphml")

    nx.write_graphml(g_central, path_central)
    nx.write_graphml(g_perifericos, path_perifericos)
    print(f"Clusters saved to:\n - Central Core: {path_central}\n - Peripherals: {path_perifericos}")

    return g_central, g_perifericos, path_central, path_perifericos

def salvar_visualizacoes_comuns(dados_simples, file_name, pasta_ranking="ranking_palavras", plotar_frequencia=False):
    """
    Exports word frequency ranking CSV and co-occurrence graph (.gdf).
    """
    os.makedirs(pasta_ranking, exist_ok=True)

    # 1. Word Ranking
    ranking = ovi.ranking_words(dados_simples, remove_duplicidades=True)
    file_ranking = os.path.join(pasta_ranking, f"{file_name}_ranking_palavras.csv")
    with open(file_ranking, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(ranking)
    print(f"Word ranking saved to: {file_ranking}")

    # 2. Co-occurrence graph (.gdf)
    file_grafo = os.path.join(pasta_ranking, f"{file_name}_grafo.gdf")
    ovi.generate_graph_file(dados_simples, file_grafo)
    print(f"Co-occurrence graph (.gdf) saved to: {file_grafo}")

    df_ranking = pd.read_csv(file_ranking, names=['palavra', 'quantidade'])

    if plotar_frequencia:
        ovi.scartter_plot_from_ranking(df_ranking)

    return df_ranking, file_ranking, file_grafo

def processar_texto_livre(texto_ou_linhas, modo_divisao="paragrafos", min_tokens=1):
    """
    Takes any raw text (single string or list of strings),
    splits into transactions, and extracts meaningful tokens (removing stopwords and symbols).
    Returns list of token lists (transactions) suitable for ARM/TOR.
    """
    nlp = get_spacy_model()

    # Split into units/transactions
    if isinstance(texto_ou_linhas, list):
        blocos = [str(b).strip() for b in texto_ou_linhas if str(b).strip()]
    elif modo_divisao == "linhas":
        blocos = [l.strip() for l in texto_ou_linhas.splitlines() if l.strip()]
    elif modo_divisao == "frases":
        blocos = [f.strip() for f in re.split(r'[\.\?\!\;\n]+', texto_ou_linhas) if f.strip()]
    else:  # 'paragrafos'
        blocos = [p.strip() for p in re.split(r'\n\s*\n', texto_ou_linhas) if p.strip()]
        if len(blocos) <= 1:
            blocos = [p.strip() for p in texto_ou_linhas.splitlines() if p.strip()]

    transacoes = []
    for doc in nlp.pipe(blocos, batch_size=50):
        tokens = []
        for token in doc:
            if token.is_alpha and token.text.lower() not in stopwords and len(token.text) > 2:
                tokens.append(token.text.lower())
        tokens_unicos = list(dict.fromkeys(tokens))
        if len(tokens_unicos) >= min_tokens:
            transacoes.append(tokens_unicos)

    return transacoes

def executar_tor_para_texto(
    texto_ou_arquivo,
    alg="FP-max",
    min_support=0.05,
    output_pdf="outputs/tor_texto.pdf",
    output_png=None,
    modo_divisao="paragrafos",
    show=False
):
    """
    Direct pipeline for arbitrary text:
    1. Reads text (direct string or file)
    2. Tokenizes into transactions
    3. Builds Trie of Rules (TOR)
    4. Renders and saves vector PDF
    * Does NOT compute or modify clusters. *
    """
    # 1. Fetch text content
    if os.path.exists(texto_ou_arquivo):
        print(f"Reading content from file: {texto_ou_arquivo}")
        with open(texto_ou_arquivo, "r", encoding="utf-8", errors="replace") as f:
            texto = f.read()
    else:
        texto = str(texto_ou_arquivo)

    if not texto.strip():
        raise ValueError("Provided text is empty!")

    # 2. Tokenize transactions
    print(f"Preprocessing text with split mode: '{modo_divisao}'...")
    transacoes = processar_texto_livre(texto, modo_divisao=modo_divisao)
    print(f"Total transactions identified: {len(transacoes)}")

    if len(transacoes) == 0:
        raise ValueError("No valid transactions found after stopwords removal.")

    # Dynamically adjust min_support if transactions count is low
    if min_support * len(transacoes) < 1.0:
        min_support = max(0.01, 1.0 / len(transacoes))
        print(f"Adjusting min_support to {round(min_support, 3)} to accommodate transaction size.")

    # 3. Generate TOR
    tor = gerar_tor(transacoes, alg=alg, min_support=min_support)
    print(f"TOR successfully constructed! Nodes in graph: {tor.trie.number_of_nodes()}")

    # 4. Export visual PDF
    print(f"Drawing graph and generating PDF: {output_pdf}")
    pf.plotar_grafo(
        tor.trie,
        output_pdf=output_pdf,
        output_png=output_png,
        show=show
    )

    return tor, output_pdf
