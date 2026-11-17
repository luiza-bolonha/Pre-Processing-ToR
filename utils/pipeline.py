import os
import re
import csv
import tempfile
from collections import Counter
import pandas as pd
import networkx as nx
import nltk
import spacy

# Ensure clean temp cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

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

def calcular_suporte_otimo(
    dados_processados,
    alg="FP-max",
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    min_support=None,
    verbose=True
):
    """
    Calculates the optimal ARM minimum support value dynamically.
    - If min_support is None:
        1. Estimates base support using target_words (default: 50 most frequent words).
        2. Verifies whether the resulting Trie has at least min_nodes (default: 4 nodes).
           If not, decreases support until the tree meets min_nodes.
    - If target_nodes is provided:
        Uses binary search to find the support value that achieves approximately target_nodes.
    - If min_support is explicitly provided:
        Validates against min_nodes (adjusting if fewer nodes were produced).
    """
    N = len(dados_processados)
    if N == 0:
        return 0.05

    doc_freq = Counter()
    for trans in dados_processados:
        doc_freq.update(set(trans))

    U = len(doc_freq)
    if U == 0:
        return 0.05

    freqs = sorted(doc_freq.values(), reverse=True)
    min_possible_support = max(0.0001, 1.0 / N)

    # 1. Modo busca por número de nós exato (target_nodes)
    if target_nodes is not None and target_nodes > 0:
        low = min_possible_support
        high = max(min_possible_support, freqs[0] / N)
        best_support = low
        best_diff = float("inf")

        for _ in range(8):
            mid = (low + high) / 2
            try:
                tor = trieofrules(data=dados_processados, alg=alg, min_support=mid)
                nodes = tor.trie.number_of_nodes()
            except Exception:
                nodes = 0

            diff = abs(nodes - target_nodes)
            if diff < best_diff:
                best_diff = diff
                best_support = mid

            if nodes < target_nodes:
                high = mid  # suporte alto gera poucos nós, precisamos diminuir
            else:
                low = mid   # suporte baixo gera muitos nós, precisamos aumentar

        final_support = round(best_support, 4 if best_support >= 0.001 else 5)
        if verbose:
            print(f"[Suporte Dinâmico] Alvo: ~{target_nodes} nós -> Suporte calculado: {final_support}")
        return final_support

    # 2. Modo baseado em palavras frequentes (target_words) e mínimo de nós (min_nodes)
    if min_support is not None:
        support = float(min_support)
        motivo = f"valor explícito ({min_support})"
    else:
        target_w = target_words if target_words is not None and target_words > 0 else 50
        idx = min(target_w - 1, U - 1)
        freq_corte = freqs[idx]
        support = max(min_possible_support, freq_corte / N)
        palavras_usadas = min(target_w, U)
        motivo = f"baseado em {palavras_usadas} palavras mais frequentes (freq={freq_corte}/{N})"

    # Avalia quantos nós o suporte atual gera no grafo
    try:
        tor = trieofrules(data=dados_processados, alg=alg, min_support=support)
        nodes = tor.trie.number_of_nodes()
    except Exception:
        nodes = 0

    curr_support = support
    ajustado = False
    tentativas = 0
    alvo_min_nodes = min_nodes if (min_nodes is not None and min_nodes > 0) else 4

    while nodes < alvo_min_nodes and curr_support > min_possible_support and tentativas < 8:
        tentativas += 1
        ajustado = True
        curr_support = max(min_possible_support, curr_support * 0.7)
        try:
            tor = trieofrules(data=dados_processados, alg=alg, min_support=curr_support)
            nodes = tor.trie.number_of_nodes()
        except Exception:
            nodes = 0

    final_support = round(curr_support, 4 if curr_support >= 0.001 else 5)
    if verbose:
        if ajustado:
            print(f"[Suporte Dinâmico] Suporte ajustado para {final_support} para garantir pelo menos {alvo_min_nodes} nós (nós obtidos: {nodes}).")
        else:
            print(f"[Suporte Dinâmico] Suporte definido em {final_support} ({motivo}, gerando {nodes} nós no grafo).")

    return final_support

def gerar_tor(
    dados_processados,
    alg="FP-max",
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    output_graphml=None
):
    """
    Generates the Trie of Rules (TOR) from tokenized transactions.
    Optionally saves the resulting graph in .graphml format.
    If min_support is None or target_nodes is provided, calculates support dynamically.
    """
    if min_support is None or target_nodes is not None:
        min_support = calcular_suporte_otimo(
            dados_processados=dados_processados,
            alg=alg,
            target_words=target_words,
            min_nodes=min_nodes,
            target_nodes=target_nodes,
            min_support=min_support,
            verbose=True
        )

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
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    output_pdf="outputs/tor_texto.pdf",
    output_png=None,
    modo_divisao="paragrafos",
    show=False
):
    """
    Direct pipeline for arbitrary text:
    1. Reads text (direct string or file)
    2. Tokenizes into transactions
    3. Calculates optimal support dynamically (if min_support is None or target_nodes is set)
    4. Builds Trie of Rules (TOR)
    5. Renders and saves vector PDF
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

    # 3. Calculate optimal support if not specified or target_nodes is passed
    if min_support is None or target_nodes is not None:
        min_support = calcular_suporte_otimo(
            dados_processados=transacoes,
            alg=alg,
            target_words=target_words,
            min_nodes=min_nodes,
            target_nodes=target_nodes,
            min_support=min_support,
            verbose=True
        )

    # 4. Generate TOR
    tor = gerar_tor(transacoes, alg=alg, min_support=min_support)
    print(f"TOR successfully constructed! Nodes in graph: {tor.trie.number_of_nodes()}")

    # 5. Export visual PDF
    print(f"Drawing graph and generating PDF: {output_pdf}")
    pf.plotar_grafo(
        tor.trie,
        output_pdf=output_pdf,
        output_png=output_png,
        show=show
    )

    return tor, output_pdf
