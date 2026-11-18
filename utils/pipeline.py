"""
Shared orchestration routines used by the news, social media and free-text pipelines.
"""

import csv
import os
import re
import tempfile
from collections import Counter

import nltk
import pandas as pd
import spacy

# Ensure clean temp cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from trieofrules import TrieOfRules
import utils.analize_tor as ator
import utils.other_visualizations as ovi
import utils.processing_file as pf

DEFAULT_DIRECTORIES = [
    "outputs",
    "outputs_clusters",
    "outputs_clusters_csv",
    "ranking_palavras",
    "datasets",
    "process_datasets",
]

# Ensure stopwords are available
try:
    stopwords = set(nltk.corpus.stopwords.words("portuguese"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        stopwords = set(nltk.corpus.stopwords.words("portuguese"))
    except Exception:
        stopwords = set()

# Shared SpaCy model, loaded on first use
_nlp = None


def get_spacy_model():
    """Loads the Portuguese SpaCy model on demand (large, falling back to small)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("pt_core_news_lg")
        except Exception:
            _nlp = spacy.load("pt_core_news_sm")
    return _nlp


def ensure_directories(directories=None):
    """Creates the default project directories if they don't exist."""
    for directory in directories or DEFAULT_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)


def _count_tor_nodes(transactions, alg, support):
    """Builds a TOR with the given support and returns its number of nodes (0 on failure)."""
    try:
        return TrieOfRules(data=transactions, alg=alg, min_support=support).trie.number_of_nodes()
    except Exception:
        return 0


def _round_support(support):
    return round(support, 4 if support >= 0.001 else 5)


def compute_optimal_support(
    transactions,
    alg="FP-max",
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    min_support=None,
    verbose=True,
):
    """
    Computes the ARM minimum support dynamically.

    - target_nodes given: binary search for the support producing ~target_nodes nodes.
    - otherwise the starting support is:
        * min_support, if given explicitly; or
        * the document frequency of the `target_words`-th most frequent word.
      It is then lowered until the trie has at least `min_nodes` nodes.
    """
    n_transactions = len(transactions)
    if n_transactions == 0:
        return 0.05

    doc_freq = Counter()
    for transaction in transactions:
        doc_freq.update(set(transaction))
    if not doc_freq:
        return 0.05

    freqs = sorted(doc_freq.values(), reverse=True)
    min_possible_support = max(0.0001, 1.0 / n_transactions)

    # 1. Binary search for a target number of nodes
    if target_nodes is not None and target_nodes > 0:
        low = min_possible_support
        high = max(min_possible_support, freqs[0] / n_transactions)
        best_support = low
        best_diff = float("inf")

        for _ in range(8):
            mid = (low + high) / 2
            nodes = _count_tor_nodes(transactions, alg, mid)

            diff = abs(nodes - target_nodes)
            if diff < best_diff:
                best_diff = diff
                best_support = mid

            if nodes < target_nodes:
                high = mid  # too few nodes: lower the support
            else:
                low = mid   # too many nodes: raise the support

        final_support = _round_support(best_support)
        if verbose:
            print(f"[Dynamic Support] Target: ~{target_nodes} nodes -> computed support: {final_support}")
        return final_support

    # 2. Starting support from an explicit value or from the most frequent words
    if min_support is not None:
        support = float(min_support)
        reason = f"explicit value ({min_support})"
    else:
        target_w = target_words if target_words and target_words > 0 else 50
        cutoff_freq = freqs[min(target_w, len(freqs)) - 1]
        support = max(min_possible_support, cutoff_freq / n_transactions)
        reason = (f"based on the {min(target_w, len(freqs))} most frequent words "
                  f"(freq={cutoff_freq}/{n_transactions})")

    # Lower the support until the trie reaches min_nodes
    required_nodes = min_nodes if min_nodes and min_nodes > 0 else 4
    nodes = _count_tor_nodes(transactions, alg, support)
    adjusted = False
    attempts = 0

    while nodes < required_nodes and support > min_possible_support and attempts < 8:
        attempts += 1
        adjusted = True
        support = max(min_possible_support, support * 0.7)
        nodes = _count_tor_nodes(transactions, alg, support)

    final_support = _round_support(support)
    if verbose:
        if adjusted:
            print(f"[Dynamic Support] Support adjusted to {final_support} to reach at least "
                  f"{required_nodes} nodes (nodes obtained: {nodes}).")
        else:
            print(f"[Dynamic Support] Support set to {final_support} ({reason}, producing {nodes} nodes).")

    return final_support


def build_tor(
    transactions,
    alg="FP-max",
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    output_graphml=None,
):
    """
    Builds the Trie of Rules (TOR) from tokenized transactions, optionally saving it as .graphml.
    The support is computed dynamically when min_support is None or target_nodes is given.
    """
    if min_support is None or target_nodes is not None:
        min_support = compute_optimal_support(
            transactions=transactions,
            alg=alg,
            target_words=target_words,
            min_nodes=min_nodes,
            target_nodes=target_nodes,
            min_support=min_support,
        )

    print(f"Generating Trie of Rules (TOR) with alg='{alg}' and min_support={min_support}...")
    tor = TrieOfRules(data=transactions, alg=alg, min_support=min_support)

    if output_graphml:
        dir_name = os.path.dirname(output_graphml)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        print(f"Saving graph to: {output_graphml}")
        tor.save_graph(filename=output_graphml)

    return tor


def split_and_save_clusters(graphml_path, file_name, folder="outputs_clusters", show=False):
    """Splits the graph into central core and peripheral clusters and renders both to PDF."""
    os.makedirs(folder, exist_ok=True)
    print("Separating central core from peripheral clusters...")
    core, peripheral = ator.split_core_and_peripheral(graphml_path)

    core_path = os.path.join(folder, f"{file_name}_cluster_central.pdf")
    peripheral_path = os.path.join(folder, f"{file_name}_clusters_perifericos.pdf")

    print("Rendering central core cluster...")
    pf.plot_graph(core, output_pdf=core_path, show=show)
    print("Rendering peripheral clusters...")
    pf.plot_graph(peripheral, output_pdf=peripheral_path, show=show)
    print(f"Clusters saved to:\n - Central Core: {core_path}\n - Peripherals: {peripheral_path}")

    return core, peripheral, core_path, peripheral_path


def save_common_visualizations(transactions, file_name, ranking_folder="ranking_palavras",
                               plot_frequency=False):
    """Exports the word frequency ranking (.csv) and the co-occurrence graph (.gdf)."""
    os.makedirs(ranking_folder, exist_ok=True)

    # 1. Word ranking
    ranking = ovi.rank_words(transactions, remove_duplicates=True)
    ranking_file = os.path.join(ranking_folder, f"{file_name}_ranking_palavras.csv")
    with open(ranking_file, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(ranking)
    print(f"Word ranking saved to: {ranking_file}")

    # 2. Co-occurrence graph (.gdf)
    graph_file = os.path.join(ranking_folder, f"{file_name}_grafo.gdf")
    ovi.generate_graph_file(transactions, graph_file)
    print(f"Co-occurrence graph (.gdf) saved to: {graph_file}")

    df_ranking = pd.read_csv(ranking_file, names=["word", "count"])
    if plot_frequency:
        ovi.plot_word_frequency(df_ranking)

    return df_ranking, ranking_file, graph_file


def split_text(text, split_mode="paragraphs"):
    """Splits raw text into blocks: 'paragraphs', 'sentences' or 'lines'."""
    if split_mode == "lines":
        blocks = text.splitlines()
    elif split_mode == "sentences":
        blocks = re.split(r"[\.\?\!\;\n]+", text)
    else:  # paragraphs (blank-line separated); falls back to lines for single-paragraph text
        blocks = re.split(r"\n\s*\n", text)
        if len([b for b in blocks if b.strip()]) <= 1:
            blocks = text.splitlines()
    return [b.strip() for b in blocks if b.strip()]


def preprocess_free_text(text_or_blocks, split_mode="paragraphs", min_tokens=1):
    """
    Turns raw text (a string or a list of strings) into transactions:
    one list of unique, lowercase, stopword-free tokens per block.
    """
    nlp = get_spacy_model()

    if isinstance(text_or_blocks, list):
        blocks = [str(b).strip() for b in text_or_blocks if str(b).strip()]
    else:
        blocks = split_text(text_or_blocks, split_mode)

    transactions = []
    for doc in nlp.pipe(blocks, batch_size=50):
        tokens = [
            token.text.lower()
            for token in doc
            if token.is_alpha and token.text.lower() not in stopwords and len(token.text) > 2
        ]
        unique_tokens = list(dict.fromkeys(tokens))
        if len(unique_tokens) >= min_tokens:
            transactions.append(unique_tokens)

    return transactions


def run_tor_for_text(
    text_or_file,
    alg="FP-max",
    min_support=None,
    target_words=50,
    min_nodes=4,
    target_nodes=None,
    output_pdf="outputs/tor_texto.pdf",
    output_png=None,
    split_mode="paragraphs",
    show=False,
):
    """
    Free-text pipeline (no clusters):
    read text -> tokenize into transactions -> build TOR -> render PDF/PNG.
    """
    # 1. Read text (file path or the text itself)
    if os.path.exists(text_or_file):
        print(f"Reading content from file: {text_or_file}")
        with open(text_or_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = str(text_or_file)

    if not text.strip():
        raise ValueError("Provided text is empty!")

    # 2. Tokenize into transactions
    print(f"Preprocessing text with split mode: '{split_mode}'...")
    transactions = preprocess_free_text(text, split_mode=split_mode)
    print(f"Total transactions identified: {len(transactions)}")
    if not transactions:
        raise ValueError("No valid transactions found after stopwords removal.")

    # 3. Build TOR (support is computed dynamically when needed)
    tor = build_tor(
        transactions,
        alg=alg,
        min_support=min_support,
        target_words=target_words,
        min_nodes=min_nodes,
        target_nodes=target_nodes,
    )
    print(f"TOR successfully constructed! Nodes in graph: {tor.trie.number_of_nodes()}")

    # 4. Render
    print(f"Drawing graph and generating PDF: {output_pdf}")
    pf.plot_graph(tor.trie, output_pdf=output_pdf, output_png=output_png, show=show)

    return tor, output_pdf
