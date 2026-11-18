"""
Trie of Rules (TOR).

A prefix tree built from frequent itemsets, where each path from the root
represents an association rule annotated with support, confidence and lift.
"""

from math import log

import networkx as nx

import trieofrules.arm as arm

ROOT_ID = 0
ROOT_LABEL = "NULL"


class TrieOfRules:
    def __init__(self, data, min_support=0.1, alg="FP-max", frequent_sequences=None,
                 round_value=3, min_len=0):
        """
        Builds the trie from raw transactions (needed to compute the metrics).

        data:               list of transactions (each one a list of items)
        min_support:        minimum support for frequent itemset mining
        alg:                'FP-max', 'Apriori' or 'FP-growth' (via mlxtend)
        frequent_sequences: optional precomputed itemsets (skips mining)
        round_value:        decimal places for the stored metrics
        min_len:            minimum itemset length to keep

        Note: mining uses mlxtend, which may be slow on large datasets.
        """
        self.data = data
        self.min_support = min_support
        self.alg = alg
        self.round_value = round_value

        if frequent_sequences is not None:
            self.frequent_sequences = frequent_sequences
            self.frequent_items = arm.count_items_in_sequences(self.data, self.frequent_sequences)
        else:
            self.frequent_sequences = arm.mine_frequent_sequences(data, min_support, alg, min_len)
            self.frequent_items = arm.find_frequent_items(self.data, self.min_support)

        self.root_node_id = ROOT_ID
        self._build_trie()
        self._add_metrics(self.root_node_id)

    # ---------- Construction ----------
    def _build_trie(self):
        """Inserts every frequent itemset, items ordered by global frequency."""
        self.trie = nx.DiGraph()
        self.trie.add_node(self.root_node_id, value=ROOT_LABEL)

        for sequence in self.frequent_sequences:
            # frequent_items is sorted by frequency, so this orders the itemset
            sorted_items = [item for item in self.frequent_items if item in sequence]
            if sorted_items:
                self._insert(sorted_items)

    def _insert(self, items):
        """Walks down the trie following `items`, creating missing branches."""
        current = self.root_node_id
        for item in items:
            child = self._find_child(current, item)
            if child is None:
                child = self.trie.number_of_nodes()
                self.trie.add_edge(current, child)
                self.trie.nodes[child]["value"] = item
            current = child

    def _find_child(self, node_id, item):
        for child in self.trie.successors(node_id):
            if self.trie.nodes[child]["value"] == item:
                return child
        return None

    def _add_metrics(self, node_id):
        """Recursively annotates nodes with support, confidence and lift."""
        node = self.trie.nodes[node_id]
        if node_id == self.root_node_id:
            node["support"] = 1.0
            node["confidence"] = 1.0
            node["lift"] = 1.0
        else:
            rule = self.get_path(node_id)
            support = arm.get_support(rule, self.data)
            parent_support = self.trie.nodes[self._parent(node_id)]["support"]
            consequent_support = self.frequent_items[rule[-1]] / len(self.data)

            node["support"] = round(support, self.round_value)
            # conf(abc -> d) = supp(abcd) / supp(abc)
            node["confidence"] = round(support / parent_support, self.round_value)
            # lift(abc -> d) = supp(abcd) / (supp(abc) * supp(d))
            node["lift"] = round(support / (parent_support * consequent_support), self.round_value)

        for child in self.trie.successors(node_id):
            self._add_metrics(child)

    # ---------- Queries ----------
    def _parent(self, node_id):
        return next(self.trie.predecessors(node_id))  # a trie node has a single parent

    def get_path(self, node_id):
        """Returns the items on the path from the root to `node_id` (the rule)."""
        rule = []
        while node_id != self.root_node_id:
            rule.append(self.trie.nodes[node_id]["value"])
            node_id = self._parent(node_id)
        return rule[::-1]

    # ---------- Export ----------
    def save_graph(self, filename, fileformat="graphml"):
        """Saves the trie as GraphML, GEXF or GML."""
        writers = {
            "gexf": nx.write_gexf,
            "gml": nx.write_gml,
            "graphml": nx.write_graphml_lxml,
        }
        writer = writers.get(fileformat.lower())
        if writer is None:
            print("Wrong file format. Use:", ", ".join(writers))
            return
        writer(self.trie, filename)

    def save_gexf_with_style(self, filename="tor.gexf"):
        """
        Saves a GEXF file with Gephi 'viz' attributes:
        size from confidence, color from lift (greener = higher lift).
        """
        lifts = [data["lift"] for _, data in self.trie.nodes(data=True)]
        lift_max = max(lifts)

        for _, data in self.trie.nodes(data=True):
            lift = data["lift"]
            green = round(log(lift / lift_max + 1, 2) * 127)
            if lift > 1:
                color = {"a": 0, "r": 190 - green, "g": 255 - green, "b": 190 - green}
            else:
                color = {"a": 1 - lift, "r": 100 + green, "g": 127 + green, "b": 100 + green}
            data["viz"] = {"size": data["confidence"] * 20 + 15, "color": color}

        nx.write_gexf(self.trie, filename)
