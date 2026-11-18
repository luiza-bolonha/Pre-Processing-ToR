"""
Analysis on top of a saved TOR graph (.graphml):
cluster extraction, core/periphery split, text-to-cluster mapping,
LLM prompt generation, text recommendation and KWIC concordance.
"""

import re

import networkx as nx
import nltk

try:
    nltk.corpus.stopwords.words("portuguese")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass
stopwords = set(nltk.corpus.stopwords.words("portuguese"))

# In graphs read from .graphml, node ids are strings: the TOR root is "0".
ROOT_ID = "0"


def _load_graph_without_root(graphml_path):
    graph = nx.read_graphml(graphml_path)
    if ROOT_ID in graph:
        graph.remove_node(ROOT_ID)
    return graph


def extract_clusters(graphml_path, show_results=True):
    """
    Splits the TOR (without its root) into clusters: each first-level node with
    descendants becomes a cluster, described by its branches (root-to-leaf paths).
    """
    graph = _load_graph_without_root(graphml_path)

    def label(node):
        return graph.nodes[node].get("value", node)

    clusters = []
    visited = set()

    for root in graph.nodes:
        if root in visited or graph.out_degree(root) == 0:
            continue

        cluster_nodes = {root} | nx.descendants(graph, root)
        visited |= cluster_nodes

        leaves = [n for n in cluster_nodes if graph.out_degree(n) == 0]
        branches = []
        branch_words = set()
        for leaf in leaves:
            for path in nx.all_simple_paths(graph, root, leaf):
                path_labels = [label(n) for n in path]
                branches.append(path_labels)
                branch_words.update(path_labels[1:])

        clusters.append({
            "root": label(root),
            "branches": branches,
            "n_nodes": len(cluster_nodes),
            "ids": cluster_nodes,
            "words": list(branch_words),
        })

    if show_results:
        for idx, cluster in enumerate(clusters, 1):
            print(f"Cluster {idx} — Root: {cluster['root']} — {cluster['n_nodes']} nodes")
            print("Branches:")
            for branch in cluster["branches"]:
                print(" → ".join(branch))
            print("Words:")
            print(", ".join(cluster["words"]))
            print("-" * 80)

    return clusters


def split_core_and_peripheral(graphml_path):
    """
    Splits the TOR into two graphs:
      - core:       first-level items without children (single nodes)
      - peripheral: first-level items that have descendants (subtrees)
    The root node and its edges are kept in both graphs.
    """
    graph = nx.read_graphml(graphml_path)
    core = nx.DiGraph()
    peripheral = nx.DiGraph()

    has_root = ROOT_ID in graph
    first_level = graph.successors(ROOT_ID) if has_root else graph.nodes

    for child in first_level:
        if child == ROOT_ID:
            continue
        subtree = graph.subgraph(nx.descendants(graph, child) | {child}).copy()
        if len(subtree) <= 1:
            core = nx.compose(core, subtree)
        else:
            peripheral = nx.compose(peripheral, subtree)

    if has_root:
        root_attrs = graph.nodes[ROOT_ID]
        root_edges = list(graph.out_edges(ROOT_ID, data=True)) + list(graph.in_edges(ROOT_ID, data=True))
        for subgraph in (core, peripheral):
            subgraph.add_node(ROOT_ID, **root_attrs)
            for u, v, data in root_edges:
                other = v if u == ROOT_ID else u
                if other in subgraph:
                    subgraph.add_edge(u, v, **data)

    return core, peripheral


def map_texts_to_clusters(clusters, tokenized_texts, original_texts):
    """
    For each cluster, collects the texts that contain every word of at least
    one of its branches.
    """
    results = []

    for idx, cluster in enumerate(clusters, 1):
        branch_sets = [set(branch) for branch in cluster["branches"]]
        related = [
            {"tweet": original_texts[i], "index": i}
            for i, tokens in enumerate(tokenized_texts)
            if any(branch.issubset(tokens) for branch in branch_sets)
        ]

        results.append({
            "cluster_idx": idx,
            "root": cluster["root"],
            "n_nodes": cluster["n_nodes"],
            "n_tweets": len(related),
            "tweets": related,
        })
        print(f"Cluster {idx} ({cluster['root']}) — {len(related)} related tweets")

    return results


def build_llm_prompt(cluster):
    """Builds an LLM prompt (in Portuguese, the language of the data) to summarize a cluster."""
    return (
        f"Considere os tweets a seguir, que foram identificados como relacionados ao assunto "
        f"{cluster['root']}. Eles mencionam com frequência palavras como: {', '.join(cluster['words'])}. "
        "Com base nesse conteúdo, gere um pequeno texto de 1 a 2 parágrafos, com uma síntese clara "
        "e objetiva sobre o que está sendo discutido nos tweets. O texto deve ajudar qualquer pessoa "
        "a entender rapidamente o teor das conversas, mesmo que ela não tenha lido os tweets "
        "originais. Retorne somente o resumo."
    )


def tokenize_words(text):
    """Lowercases the text and returns its words (punctuation removed)."""
    return re.findall(r"\b\w+\b", text.lower())


def extract_relevant_words(graphml_path):
    """Returns the sorted, stopword-free words used in the TOR node labels."""
    graph = _load_graph_without_root(graphml_path)

    words = set()
    for _, data in graph.nodes(data=True):
        value = data.get("value")
        if value:
            words.update(value.split())

    return sorted(w for w in words if w not in stopwords)


def recommend_text(texts, graphml_path):
    """
    Returns the text with the highest proportion of TOR words
    (relevant words / distinct words), i.e. the most informative and concise one.
    """
    relevant_words = set(extract_relevant_words(graphml_path))
    best_index = -1
    best_score = 0

    for i, text in enumerate(texts):
        words = tokenize_words(text)
        if not words:
            continue
        relevant = sum(1 for w in words if w in relevant_words)
        score = relevant / len(set(words))
        if score > best_score:
            best_score = score
            best_index = i

    print(f"Best ratio: {round(best_score * 100, 2)}%")
    if best_index == -1:
        return None
    recommended = texts[best_index]
    print(f"Number of words: {len(recommended.split())}")
    return recommended


def concordance(texts, term, width=32, case_sensitive=False):
    """Prints every occurrence of `term` with `width` characters of context (KWIC)."""
    text = " ".join(texts)
    if not case_sensitive:
        text = text.lower()
        term = term.lower()

    lines = []
    for match in re.finditer(re.escape(term), text):
        start = match.start()
        end = start + len(term)
        left = text[max(0, start - width):start].replace("\n", " ")
        right = text[end:min(len(text), end + width)].replace("\n", " ")
        # Left padding aligns the term on the same column in every line
        padding = " " * max(0, width - len(left))
        lines.append(f"{padding}{left}{text[start:end]}{right}")

    print(f"Number of occurrences: {len(lines)}\n")
    for line in lines:
        print(line)
