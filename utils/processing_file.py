import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from fa2_modified import ForceAtlas2

# pip install networkx matplotlib fa2_modified

# ---------- 1. Funções auxiliares ----------
def ranking(valores, minimo, maximo):
    """Igual ao 'Ranking' do Gephi: mapeia o menor valor para `minimo`
    e o maior para `maximo`, de forma linear."""
    v = np.array(valores, dtype=float)
    if v.max() == v.min():
        return np.full_like(v, (minimo + maximo) / 2)
    return minimo + (v - v.min()) / (v.max() - v.min()) * (maximo - minimo)

def rotacionar(pos, graus):
    """Equivalente à 'Rotação' do Gephi."""
    a = np.radians(graus)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    return {n: R @ np.array(p) for n, p in pos.items()}

def plotar_grafo(G):
    
    # ---------- 2. Atributos dos nós ----------
    nos = list(G.nodes())
    suporte   = [float(G.nodes[n]["support"])    for n in nos]
    confianca = [float(G.nodes[n]["confidence"]) for n in nos]
    rotulos   = {n: str(G.nodes[n].get("value", n)) for n in nos}

    # Tamanho: ranking de 10 a 60 (como no Gephi)
    tam_gephi = ranking(suporte, 10, 60)

    # Cor: ranking pela confiança
    norm = Normalize(vmin=min(confianca), vmax=max(confianca))
    cmap = plt.cm.viridis          # troque por plt.cm.YlOrRd, plt.cm.plasma etc.
    cores = [cmap(norm(c)) for c in confianca]

    # ---------- 3. Layout: ForceAtlas2 -> "Yifan Hu" ----------
    fa2 = ForceAtlas2(
        gravity=1.0,
        scalingRatio=2.0,
        outboundAttractionDistribution=True,   # "Dissuade Hubs"
        adjustSizes=True,                      # "Prevent Overlap"
        verbose=False,
    )
    pos = fa2.forceatlas2_networkx_layout(G.to_undirected(), pos=None, iterations=1000)

    # Refinamento parecido com o Yifan Hu (força dirigida, partindo das posições do FA2)
    pos = nx.spring_layout(G, pos=pos, iterations=50, seed=42)

    # Rotação opcional
    # pos = rotacionar(pos, 30)

    # ---------- 4. Desenho ----------
    fig, ax = plt.subplots(figsize=(14, 14))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.4, width=0.8,
        arrows=G.is_directed(),
        arrowsize=10,
        connectionstyle="arc3,rad=0",          # aresta reta (sem curva)
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nos,
        node_size=tam_gephi ** 2 / 2,          # ver explicação abaixo
        node_color=cores,
        edgecolors="black", linewidths=0.5,
    )

    nx.draw_networkx_labels(G, pos, labels=rotulos, font_size=8, ax=ax)

    # Barra de cor para indicar a confiança
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Confiança")

    ax.axis("off")
    plt.savefig("grafo.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("grafo.png", dpi=300, bbox_inches="tight")
    plt.show()