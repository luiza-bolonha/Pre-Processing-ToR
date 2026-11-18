"""Association Rule Mining (ARM) helpers used to build the Trie of Rules."""

from collections import Counter

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax

ARM_ALGORITHMS = {"FP-max": fpmax, "Apriori": apriori, "FP-growth": fpgrowth}


def mine_frequent_sequences(transactions, min_support, alg, min_len):
    """Returns the frequent itemsets (as sets) with at least `min_len` items."""
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(encoded, columns=encoder.columns_)
    itemsets = ARM_ALGORITHMS[alg](df, min_support, use_colnames=True).itemsets.tolist()
    return [set(itemset) for itemset in itemsets if len(itemset) >= min_len]


def _sorted_by_count(counts):
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def find_frequent_items(transactions, min_support):
    """
    Counts every item and keeps those with support >= min_support.
    Returns {item: count}, sorted by count (descending).
    """
    counts = Counter(item for transaction in transactions for item in transaction)
    total = len(transactions)
    return _sorted_by_count({item: c for item, c in counts.items() if c / total >= min_support})


def count_items_in_sequences(transactions, frequent_sequences):
    """
    Counts, across all transactions, the items that appear in `frequent_sequences`.
    Returns {item: count}, sorted by count (descending).
    """
    counts = {item: 0 for sequence in frequent_sequences for item in sequence}
    for transaction in transactions:
        for item in transaction:
            if item in counts:
                counts[item] += 1
    return _sorted_by_count(counts)


def get_support(items, transactions):
    """Fraction of transactions that contain every item in `items`."""
    item_set = set(items)
    matches = sum(1 for row in transactions if item_set.issubset(row))
    return matches / len(transactions)
