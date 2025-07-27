from arulespy.arules import Transactions, apriori, parameters
from arulespy.arulesViz import plot
from rpy2.ipython.ggplot import image_png
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations


def get_rules_to_arulespy(transacoes, support):
    transacoes_sanitizadas = [
        [str(item).strip() for item in trans if item and str(item).strip() != '']
        for trans in transacoes
    ]
    item_labels = sorted({item for trans in transacoes_sanitizadas for item in trans})
    item_labels_set = set(item_labels)
    
    for trans in transacoes_sanitizadas:
        for item in trans:
            if item not in item_labels_set:
                print(f"Item não encontrado em item_labels: {item}")

    trans = Transactions.from_list(transacoes_sanitizadas, item_labels)
    rules = apriori(
        trans,
        parameter=parameters({"supp": support}),
        control=parameters({"verbose": False})
    )
    return rules

def scartter_plot_from_ranking(df):
    df_top = df.sort_values(by='quantidade', ascending=False).head(50)

    plt.figure(figsize=(12, 6))
    plt.bar(df_top['palavra'], df_top['quantidade'], color='green')

    plt.xticks(rotation=90)
    plt.title('Gráfico de Frequência')
    plt.xlabel('Palavras')
    plt.ylabel('Quantidade')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def generate_graph_file(list_of_lists, output_file):
    # Flatten all tokens
    all_tokens = [token for sublist in list_of_lists for token in sublist]
    
    # Count frequencies
    freq_counter = Counter(all_tokens)
    
    # Get top 100 words
    top_100 = freq_counter.most_common(100)
    top_100_words = set([word for word, count in top_100])
    
    # Create node lines
    nodes = ["nodedef>name VARCHAR,label VARCHAR,weight INT"]
    for word, count in top_100:
        nodes.append(f"{word},{word},{count}")
    
    # Count co-occurrences in each sublist
    edge_counter = Counter()
    for sublist in list_of_lists:
        unique_words = set(sublist)
        for w1, w2 in combinations(unique_words, 2):
            pair = tuple(sorted([w1, w2]))
            edge_counter[pair] += 1
    
    # Organize edges by origin node
    edges_by_origin = defaultdict(list)
    for (w1, w2), count in edge_counter.items():
        if w1 in top_100_words or w2 in top_100_words:
            freq1 = freq_counter[w1]
            freq2 = freq_counter[w2]
            if freq1 >= freq2:
                origin, target = w1, w2
            else:
                origin, target = w2, w1
            edges_by_origin[origin].append((origin, target, count))
    
    # Create edge lines sorted by origin node frequency (descending)
    edges = ["edgedef>node1 VARCHAR,node2 VARCHAR,weight INT,directed BOOLEAN"]
    sorted_origins = sorted(edges_by_origin.keys(), key=lambda x: freq_counter[x], reverse=True)
    
    for origin in sorted_origins:
        # Sort targets by their frequency as well
        edges_by_origin[origin].sort(key=lambda x: freq_counter[x[1]], reverse=True)
        for edge in edges_by_origin[origin]:
            origin, target, count = edge
            edges.append(f"{origin},{target},{count},True")
    
    # ➕ Add self-loops for each top 100 word
    for word, count in top_100:
        edges.append(f"{word},{word},{count},True")
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for line in nodes + edges:
            f.write(line + "\n")
'''
def generate_graph_file(list_of_lists, output_file):
    # Count frequencies: uma ocorrência por sublista
    freq_counter = Counter()
    for sublist in list_of_lists:
        unique_tokens = set(sublist)
        freq_counter.update(unique_tokens)
    
    # Get top 100 words
    top_100 = freq_counter.most_common(100)
    top_100_words = set([word for word, count in top_100])
    
    # Create node lines (apenas top100)
    nodes = ["nodedef>name VARCHAR,label VARCHAR,weight INT"]
    for word, count in top_100:
        nodes.append(f"{word},{word},{count}")
    
    # Count co-occurrences: top100 com qualquer palavra (uma ocorrência por sublista)
    edge_counter = Counter()
    for sublist in list_of_lists:
        unique_words = set(sublist)
        for top_word in top_100_words:
            if top_word in unique_words:
                for word in unique_words:
                    pair = (top_word, word)
                    edge_counter[pair] += 1
    
    # Organize edges by origin node (sempre top100 como origem)
    edges_by_origin = defaultdict(list)
    for (origin, target), count in edge_counter.items():
        freq_origin = freq_counter.get(origin, 0)
        freq_target = freq_counter.get(target, 0)
        edges_by_origin[origin].append((origin, target, count))
    
    # Create edge lines sorted by origin node frequency (descending)
    edges = ["edgedef>node1 VARCHAR,node2 VARCHAR,weight INT,directed BOOLEAN"]
    sorted_origins = sorted(edges_by_origin.keys(), key=lambda x: freq_counter[x], reverse=True)
    
    for origin in sorted_origins:
        # Sort targets by their frequency as well
        edges_by_origin[origin].sort(key=lambda x: freq_counter.get(x[1],0), reverse=True)
        for edge in edges_by_origin[origin]:
            origin, target, count = edge
            edges.append(f"{origin},{target},{count},True")
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for line in nodes + edges:
            f.write(line + "\n")
'''

def ranking_words(list_tokens, n=1000, remove_duplicidades=True):
    contador = Counter()
    for sublist in list_tokens:
        if remove_duplicidades:
            unique_tokens = set(sublist)  
            contador.update(unique_tokens)
        else:
            contador.update(sublist)
    return contador.most_common(n)
