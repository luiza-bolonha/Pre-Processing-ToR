import os
import tempfile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from fa2_modified import ForceAtlas2

# Ensure clean temp cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))


# ---------- 1. Helper Functions ----------
def ranking(values, minimum, maximum):
    """Linearly maps the minimum value to `minimum` and the maximum to `maximum`
    (similar to Gephi's Ranking transform)."""
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return np.array([])
    if v.max() == v.min():
        return np.full_like(v, (minimum + maximum) / 2)
    return minimum + (v - v.min()) / (v.max() - v.min()) * (maximum - minimum)


def rotate(pos, degrees):
    """Rotates layout positions by given degrees (similar to Gephi's Rotation)."""
    a = np.radians(degrees)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    return {n: R @ np.array(p) for n, p in pos.items()}


def plot_graph(G, output_pdf="graph.pdf", output_png=None, show=True, figsize=(14, 14)):
    """
    Renders graph using ForceAtlas2 layout refined by spring_layout
    and exports directly to vector PDF (and optionally PNG).
    """
    if len(G.nodes) == 0:
        print("Empty graph, nothing to render.")
        return

    # ---------- 2. Node Attributes ----------
    nodes = list(G.nodes())
    support    = [float(G.nodes[n].get("support", 0.0))    for n in nodes]
    confidence = [float(G.nodes[n].get("confidence", 0.0)) for n in nodes]
    labels     = {n: str(G.nodes[n].get("value", n)) for n in nodes}

    # Node Size: linear ranking between 10 and 60
    gephi_size = ranking(support, 10, 60)

    # Node Color: mapped by confidence
    c_min = min(confidence) if confidence else 0.0
    c_max = max(confidence) if confidence else 1.0
    if c_max == c_min:
        c_max = c_min + 1.0
    norm = Normalize(vmin=c_min, vmax=c_max)
    cmap = plt.cm.viridis
    colors = [cmap(norm(c)) for c in confidence]

    # ---------- 3. Layout: ForceAtlas2 -> Spring Refinement ----------
    try:
        fa2 = ForceAtlas2(
            gravity=1.0,
            scalingRatio=2.0,
            outboundAttractionDistribution=True,   # "Dissuade Hubs"
            adjustSizes=True,                      # "Prevent Overlap"
            verbose=False,
        )
        pos = fa2.forceatlas2_networkx_layout(G.to_undirected(), pos=None, iterations=1000)
        pos = nx.spring_layout(G, pos=pos, iterations=50, seed=42)
    except Exception as e:
        print(f"Notice in ForceAtlas2 ({e}). Falling back to standard spring_layout.")
        pos = nx.spring_layout(G, seed=42)

    # ---------- 4. Rendering ----------
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.4, width=0.8,
        arrows=G.is_directed(),
        arrowsize=10,
        connectionstyle="arc3,rad=0",
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodes,
        node_size=gephi_size ** 2 / 2,
        node_color=colors,
        edgecolors="black", linewidths=0.5,
    )

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Colorbar indicating confidence
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.5, label="Confidence")

    ax.axis("off")

    if output_pdf:
        dir_name = os.path.dirname(output_pdf)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
        print(f"Graph saved to PDF: {output_pdf}")

    if output_png:
        dir_name = os.path.dirname(output_png)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"Graph saved to PNG: {output_png}")

    if show:
        plt.show()
    plt.close(fig)


# Backwards compatibility aliases
plotar_grafo = plot_graph
rotacionar = rotate
