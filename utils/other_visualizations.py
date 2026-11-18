"""Auxiliary outputs: word rankings, frequency plot and co-occurrence graph (.gdf for Gephi)."""

from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.pyplot as plt


def get_arulespy_rules(transactions, support):
    """Mines rules with R's arules (via arulespy). Requires arulespy and rpy2."""
    try:
        from arulespy.arules import Transactions, apriori, parameters
    except ImportError as e:
        raise ImportError(f"arulespy or rpy2 not available in this environment: {e}")

    clean_transactions = [
        [str(item).strip() for item in transaction if item and str(item).strip()]
        for transaction in transactions
    ]
    item_labels = sorted({item for transaction in clean_transactions for item in transaction})

    arules_transactions = Transactions.from_list(clean_transactions, item_labels)
    return apriori(
        arules_transactions,
        parameter=parameters({"supp": support}),
        control=parameters({"verbose": False}),
    )


def plot_word_frequency(df, top_n=50):
    """Bar chart of the `top_n` most frequent words (expects 'word' and 'count' columns)."""
    df_top = df.sort_values(by="count", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(df_top["word"], df_top["count"], color="green")
    plt.xticks(rotation=90)
    plt.title("Word Frequency")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def generate_graph_file(transactions, output_file, top_n=100):
    """
    Writes a co-occurrence graph in GDF format (Gephi).
    Nodes: the `top_n` most frequent words. Edges: word pairs appearing in the
    same transaction, directed from the more frequent to the less frequent word.
    """
    freq = Counter(token for transaction in transactions for token in transaction)
    top_words = freq.most_common(top_n)
    top_set = {word for word, _ in top_words}

    node_lines = ["nodedef>name VARCHAR,label VARCHAR,weight INT"]
    node_lines += [f"{word},{word},{count}" for word, count in top_words]

    # Count co-occurrences inside each transaction
    pair_counts = Counter()
    for transaction in transactions:
        for pair in combinations(sorted(set(transaction)), 2):
            pair_counts[pair] += 1

    # Group edges by origin (the more frequent word of the pair)
    edges_by_origin = defaultdict(list)
    for (w1, w2), count in pair_counts.items():
        if w1 in top_set or w2 in top_set:
            origin, target = (w1, w2) if freq[w1] >= freq[w2] else (w2, w1)
            edges_by_origin[origin].append((target, count))

    # Edges sorted by origin frequency, then target frequency (both descending)
    edge_lines = ["edgedef>node1 VARCHAR,node2 VARCHAR,weight INT,directed BOOLEAN"]
    for origin in sorted(edges_by_origin, key=lambda w: freq[w], reverse=True):
        for target, count in sorted(edges_by_origin[origin], key=lambda e: freq[e[0]], reverse=True):
            edge_lines.append(f"{origin},{target},{count},True")

    # Self-loops carry each top word's own frequency
    edge_lines += [f"{word},{word},{count},True" for word, count in top_words]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(node_lines + edge_lines) + "\n")


def rank_words(transactions, n=1000, remove_duplicates=True):
    """
    Returns the `n` most common words as [(word, count), ...].
    With remove_duplicates, each word counts at most once per transaction.
    """
    counter = Counter()
    for transaction in transactions:
        counter.update(set(transaction) if remove_duplicates else transaction)
    return counter.most_common(n)
