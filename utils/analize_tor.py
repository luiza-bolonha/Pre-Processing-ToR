import networkx as nx
import re
import os

def extrair_clusters(graphml_path, show_results=True):
    G = nx.read_graphml(graphml_path)
    if "0" in G:
        G.remove_node("0")

    def nome(n):
        return G.nodes[n].get("value", n)

    componentes = []
    visitados = set()

    for raiz in G.nodes:
        if raiz in visitados or G.out_degree(raiz) == 0:
            continue

        descendentes = nx.descendants(G, raiz)
        cluster_nos = {raiz} | descendentes
        if len(cluster_nos) <= 1:
            continue

        visitados |= cluster_nos

        folhas = [n for n in cluster_nos if G.out_degree(n) == 0]
        ramos = []
        palavras_ramos = set()
        for folha in folhas:
            for caminho in nx.all_simple_paths(G, raiz, folha):
                caminho_nomes = [nome(n) for n in caminho]
                ramos.append(caminho_nomes)
                palavras_ramos.update(caminho_nomes[1:])

        componentes.append({
            "raiz": nome(raiz),
            "ramos": ramos,
            "n_nos": len(cluster_nos),
            "ids": cluster_nos,
            "palavras": list(palavras_ramos)
        })

    if(show_results):
        for idx, c in enumerate(componentes, 1):
            print(f"Cluster {idx} — Raiz: {c['raiz']} — {c['n_nos']} nós")
            print("Ramos: ")
            for ramo in c["ramos"]:
                print(" → ".join(ramo))
            print("Palavras:")
            print(", ".join(c["palavras"]))
            print("-" * 80)

    return componentes

def separar_clusters_por_tamanho(graphml_path):
    G = nx.read_graphml(graphml_path)
    G_central = nx.DiGraph()
    G_perifericos = nx.DiGraph()

    # Armazena os atributos do nó "0" e suas arestas
    origem_presente = "0" in G
    if origem_presente:
        atributos_0 = G.nodes["0"]
        arestas_de_0 = list(G.out_edges("0", data=True))
        arestas_para_0 = list(G.in_edges("0", data=True))

    for vizinho in G.successors("0") if origem_presente else G.nodes:
        if vizinho == "0":
            continue

        subnos = nx.descendants(G, vizinho) | {vizinho}
        subgrafo = G.subgraph(subnos).copy()

        if len(subgrafo.nodes) <= 1:
            G_central = nx.compose(G_central, subgrafo)
        else:
            G_perifericos = nx.compose(G_perifericos, subgrafo)

    # Adiciona "0" e suas arestas aos dois grafos se necessário
    if origem_presente:
        for G_cluster in [G_central, G_perifericos]:
            G_cluster.add_node("0", **atributos_0)
            for u, v, data in arestas_de_0:
                if v in G_cluster:
                    G_cluster.add_edge(u, v, **data)
            for u, v, data in arestas_para_0:
                if u in G_cluster:
                    G_cluster.add_edge(u, v, **data)

    return G_central, G_perifericos

def tweets_clusters(clusters, textos_tokenizados, textos_originais):
    resultados = []

    for idx, c in enumerate(clusters, 1):
        ramos_sets = [set(r) for r in c["ramos"]]
        tweets_relevantes = []

        for i, tokens in enumerate(textos_tokenizados):
            tokens_set = set(tokens)
            if any(ramo.issubset(tokens_set) for ramo in ramos_sets):
                tweets_relevantes.append({
                    "tweet": textos_originais[i],
                    "index": i
                })

        resultados.append({
            "cluster_idx": idx,
            "raiz": c["raiz"],
            "n_nos": c["n_nos"],
            "n_tweets": len(tweets_relevantes),
            "tweets": tweets_relevantes
        })

        print(f"Cluster {idx} ({c['raiz']}) — {len(tweets_relevantes)} tweets relacionados")

    return resultados

def prompt_IA(c):
    prompt = f"Considere os tweets a seguir, que foram identificados como relacionados ao assunto {c['raiz']}. Eles mencionam com frequência palavras como: {', '.join(c['palavras'])}. Com base nesse conteúdo, gere um pequeno texto de 1 a 2 parágrafos, com uma síntese clara e objetiva sobre o que está sendo discutido nos tweets. O texto deve ajudar qualquer pessoa a entender rapidamente o teor das conversas, mesmo que ela não tenha lido os tweets originais. Retorne somente o resumo."
    return prompt

from collections import Counter
import csv

def ranking_words(list_tokens, n=1000):
    contador = Counter()
    for sublist in list_tokens:
        contador.update(sublist)
    return contador.most_common(n)
    
def limpar_texto(texto):
        # Remove pontuação e coloca tudo em minúsculas
        return re.findall(r'\b\w+\b', texto.lower())

def extrair_palavras_relevantes(graphml_path):
    G = nx.read_graphml(graphml_path)

    # Remove nó raiz vazio se existir
    if "0" in G:
        G.remove_node("0")

    palavras = set()
    for n in G.nodes:
        valor = G.nodes[n].get("value")
        if valor:
            palavras.add(valor)

    return sorted(palavras)


def recomendar_resumo(lista_textos, graphml_path):    
    palavras_relevantes = extrair_palavras_relevantes(graphml_path)
    melhor_indice = -1
    melhor_pontuacao = 0
    proporcoes = []

    for i, texto in enumerate(lista_textos):
        palavras = limpar_texto(texto)
        if not palavras:
            continue
        total = len(palavras)
        relevantes = sum(1 for p in palavras if p in palavras_relevantes)
        proporcao = relevantes / total
        proporcoes.append(proporcao)

        if proporcao > melhor_pontuacao:
            melhor_pontuacao = proporcao
            melhor_indice = i
        
    print(f"Melhor proporção: {round((melhor_pontuacao*100),2)}")
    return proporcoes, lista_textos[melhor_indice] if melhor_indice != -1 else None