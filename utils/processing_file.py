"""
Graph rendering for Tries of Rules (TOR).

All distances are in "points" (1 data unit == 1 typographic point), so node
and label sizes can be compared directly with node distances.

The layout pipeline is:
    1. ForceAtlas2 (Gephi-like) for the global structure, refined by spring_layout.
    2. Around the hidden root, single nodes go on a compact inner ring and
       clusters are moved further out (keeping their shape), each in its own
       angular sector so edges of different clusters don't cross.
    3. Each label is placed below, above, right or left of its node, on a side
       none of the node's own edges cross (below when possible).
    4. A collision pass pushes apart any two nodes whose bounding boxes
       (node circle + label) intersect, and pushes nodes off edges that cross
       their box (edges that don't belong to that node).
    5. The figure size is derived from the final layout, so nothing is shrunk
       back into overlap when saving.

The artificial root node ("NULL") is not drawn: its edges are kept, so the
first-level items still converge to the center of the graph.
"""

import os
import tempfile
from functools import lru_cache

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from fa2_modified import ForceAtlas2

# Ensure clean temp cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

ROOT_LABEL = "NULL"

# Visual settings
FONT_SIZE = 8                  # label font size (pt)
NODE_DIAMETER_RANGE = (10, 36) # min/max node diameter (pt), mapped from support
LABEL_GAP = 1.5                # vertical gap between a node and its label (pt)
LABEL_PADDING = 3.0            # extra spacing kept around each node/label box (pt)
LABEL_SIDES = ("below", "above", "right", "left")  # label positions, in order of preference
BASE_CANVAS = 900.0            # initial layout extent (pt) before overlap removal
MARGIN = 30.0                  # blank border around the drawing (pt)
POINTS_PER_INCH = 72.0

# Light colormap: soft blue (low confidence) -> soft green -> soft yellow (high).
# Light tones keep the black labels readable on top of the nodes.
LIGHT_CMAP = LinearSegmentedColormap.from_list(
    "light_confidence",
    ["#c6dbef", "#a8ddb5", "#d9f0a3", "#fee391"],
)


# ---------- Helpers ----------
def rank_scale(values, minimum, maximum):
    """Linearly maps values to [minimum, maximum] (like Gephi's Ranking transform)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    if v.max() == v.min():
        return np.full_like(v, (minimum + maximum) / 2)
    return minimum + (v - v.min()) / (v.max() - v.min()) * (maximum - minimum)


def rotate(pos, degrees):
    """Rotates layout positions by the given angle (like Gephi's Rotation)."""
    a = np.radians(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)],
                         [np.sin(a),  np.cos(a)]])
    return {n: rotation @ np.asarray(p) for n, p in pos.items()}


def find_root_nodes(G):
    """Returns the artificial TOR root(s): nodes labeled 'NULL' without predecessors."""
    roots = []
    for n, data in G.nodes(data=True):
        is_null = str(data.get("value", "")).upper() == ROOT_LABEL
        has_no_parent = G.in_degree(n) == 0 if G.is_directed() else True
        if is_null and has_no_parent:
            roots.append(n)
    return roots


@lru_cache(maxsize=None)
def label_size(text, font_size=FONT_SIZE):
    """Returns the (width, height) of a rendered label, in points."""
    extents = TextPath((0, 0), text, size=font_size, prop=FontProperties()).get_extents()
    return extents.width, font_size


def compute_layout(G):
    """Force-directed layout: ForceAtlas2 refined by spring_layout."""
    try:
        fa2 = ForceAtlas2(
            gravity=1.0,
            scalingRatio=2.0,
            outboundAttractionDistribution=True,   # "Dissuade Hubs"
            adjustSizes=True,                      # "Prevent Overlap"
            verbose=False,
        )
        # Seeded start, so the same graph always produces the same drawing
        initial = nx.random_layout(G, seed=42)
        pos = fa2.forceatlas2_networkx_layout(G.to_undirected(), pos=initial, iterations=1000)
        return nx.spring_layout(G, pos=pos, iterations=50, seed=42)
    except Exception as e:
        print(f"Notice in ForceAtlas2 ({e}). Falling back to standard spring_layout.")
        return nx.spring_layout(G, seed=42)


def scale_to_points(pos, extent=BASE_CANVAS):
    """Centers the layout and scales its largest dimension to `extent` points."""
    coords = np.array(list(pos.values()), dtype=float)
    coords -= coords.mean(axis=0)
    span = np.ptp(coords, axis=0).max()
    if span > 0:
        coords *= extent / span
    return dict(zip(pos.keys(), coords))


# ---------- Arrangement around the center ----------
def node_footprint(diameter, label):
    """Radius (pt) of a circle holding a node and its label placed above or below it."""
    radius = diameter / 2
    text_w, text_h = label_size(label)
    return np.hypot(max(radius, text_w / 2), radius + LABEL_GAP + text_h) + LABEL_PADDING


def _split_counts(total, weights):
    """Splits `total` items into len(weights) integer counts proportional to weights."""
    weights = np.asarray(weights, dtype=float)
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    for i in np.argsort(counts - exact)[: total - counts.sum()]:
        counts[i] += 1
    return counts


CLUSTER_SPACING = 1.3  # edge length inside a cluster, relative to "nodes just touching"


def _compact_cluster_shape(G, members, pos, footprints, root):
    """
    Cluster positions relative to its root, rescaled so a typical edge is just
    long enough for its two nodes not to touch (times CLUSTER_SPACING). Keeps the
    force layout's shape while making cluster sizes consistent across graphs;
    the median makes it robust to nodes the force layout left nearly on top of
    each other (those are separated later by the collision pass).
    """
    rel = {m: pos[m] - pos[root] for m in members}
    ratios = []
    for u, v in G.subgraph(members).edges:
        dist = np.hypot(*(rel[u] - rel[v]))
        if dist > 0:
            ratios.append((footprints[u] + footprints[v] + LABEL_PADDING) / dist)
    if ratios:
        scale = float(np.median(ratios)) * CLUSTER_SPACING
        rel = {m: r * scale for m, r in rel.items()}
    return rel


def arrange_around_center(G, pos, center, footprints):
    """
    Reorganizes the first level of the trie around the (invisible) center, keeping
    the shape the force layout gave to each cluster:

      - clusters (nodes with descendants) get an angular sector proportional to
        their size, and are placed outside the inner ring, just far enough out
        that their sector holds them; each cluster is rotated to face outwards,
        so clusters don't overlap and their edges don't cross;
      - single nodes (no children) fill a compact inner ring, alternating
        between two close radii, spread over the gaps between the clusters' edges.

    pos and footprints are in points. Returns a new pos dict.
    """
    pos = {n: np.asarray(p, dtype=float) for n, p in pos.items()}
    origin = pos[center]
    first_level = list(G.successors(center))
    if not first_level:
        return pos

    def direction(n):
        dx, dy = pos[n] - origin
        return np.arctan2(dy, dx)

    # Keep the force layout's circular order
    first_level.sort(key=direction)
    singles = [n for n in first_level if G.out_degree(n) == 0]
    clusters = [n for n in first_level if G.out_degree(n) > 0]
    max_single = max((footprints[n] for n in singles), default=0.0)
    new_pos = dict(pos)

    # ---- Cluster sectors: angle proportional to each cluster's radius ----
    members, relative, extent = {}, {}, {}
    for c in clusters:
        members[c] = [c] + list(nx.descendants(G, c))
        relative[c] = _compact_cluster_shape(G, members[c], pos, footprints, c)
        extent[c] = max(np.hypot(*relative[c][m]) + footprints[m] for m in members[c])

    angle = {}
    if clusters:
        total_extent = sum(extent.values())
        cursor = direction(clusters[0])
        half_sector = {}
        for c in clusters:
            share = 2 * np.pi * extent[c] / total_extent
            angle[c] = cursor + share / 2  # cluster sits in the middle of its sector
            half_sector[c] = share / 2
            cursor += share
        # Arcs between consecutive cluster edges, where single nodes go
        arcs = [(angle[a], (angle[clusters[(i + 1) % len(clusters)]] - angle[a]) % (2 * np.pi) or 2 * np.pi)
                for i, a in enumerate(clusters)]
    else:
        arcs = [(direction(singles[0]), 2 * np.pi)]

    # ---- Inner ring of single nodes ----
    # Single nodes alternate between two close rings (zigzag), which doubles the
    # room between neighbors on the same ring. The smallest radius where no
    # single node touches another one or another node's edge is found by search.
    ring_step = max_single
    outer_extra = ring_step if len(singles) > 2 else 0.0
    inner_radius = footprints.get(center, 0.0) + max_single + LABEL_PADDING
    if singles:
        counts = _split_counts(len(singles), [length for _, length in arcs])
        single_angles = []
        for (start, length), k in zip(arcs, counts):
            slots = k + 1 if clusters else k  # with clusters, keep a slot for each cluster edge
            for j in range(k):
                offset = (j + 1) if clusters else (j + 0.5)
                single_angles.append(start + length * offset / slots)
        single_angles = np.array(single_angles)
        units = np.stack([np.cos(single_angles), np.sin(single_angles)], axis=1)
        fps = np.array([footprints[n] for n in singles])
        zigzag = np.array([outer_extra * (i % 2) for i in range(len(singles))])
        cluster_units = np.array([[np.cos(angle[c]), np.sin(angle[c])] for c in clusters]).reshape(-1, 2)

        def positions(radius):
            return units * (radius + zigzag)[:, None]

        def clear(radius):
            pts = positions(radius)
            # node vs node
            dist = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
            need = fps[:, None] + fps[None, :] + LABEL_PADDING
            np.fill_diagonal(dist, np.inf)
            if np.any(dist < need):
                return False
            # node vs edges from the center (to other single nodes and to clusters)
            ends = np.vstack([pts, cluster_units * 1e6])
            for i, p in enumerate(pts):
                for j, e in enumerate(ends):
                    if j == i:
                        continue
                    t = np.clip(p @ e / (e @ e), 0, 1)
                    if np.hypot(*(p - t * e)) < fps[i] + LABEL_PADDING:
                        return False
            return True

        low = inner_radius
        if not clear(low):
            high = low * 2
            while not clear(high):
                high *= 2
            for _ in range(30):
                mid = (low + high) / 2
                low, high = (low, mid) if clear(mid) else (mid, high)
            low = high
        inner_radius = low
        for n, p in zip(singles, positions(inner_radius)):
            new_pos[n] = origin + p

    # ---- Clusters outside the inner ring, each inside its sector ----
    for c in clusters:
        # Far enough out to clear the inner ring, and for the cluster to fit in its sector
        radius = inner_radius + outer_extra + max_single + extent[c] + LABEL_PADDING
        if half_sector[c] < np.pi / 2:
            radius = max(radius, extent[c] / np.sin(half_sector[c]))
        # Rotate the cluster around its root so it keeps facing outwards, then move it
        turn = angle[c] - direction(c)
        rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        target = origin + radius * np.array([np.cos(angle[c]), np.sin(angle[c])])
        for m in members[c]:
            new_pos[m] = target + rotation @ relative[c][m]
    return new_pos


def count_edge_crossings(pos, edges):
    """Number of edge pairs that cross (edges sharing an endpoint are ignored)."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    crossings = 0
    for i, (u1, v1) in enumerate(edges):
        p1, p2 = pos[u1], pos[v1]
        for u2, v2 in edges[i + 1:]:
            if {u1, v1} & {u2, v2}:
                continue
            q1, q2 = pos[u2], pos[v2]
            if (orient(p1, p2, q1) * orient(p1, p2, q2) < 0
                    and orient(q1, q2, p1) * orient(q1, q2, p2) < 0):
                crossings += 1
    return crossings


def label_rect(diameter, label, side):
    """
    Rectangle of a label placed on `side` of its node.
    Returns (offset_x, offset_y, half_width, half_height) relative to the node center, in points.
    """
    radius = diameter / 2
    text_w, text_h = label_size(label)
    offsets = {
        "below": (0.0, -(radius + LABEL_GAP + text_h / 2)),
        "above": (0.0, radius + LABEL_GAP + text_h / 2),
        "right": (radius + LABEL_GAP + text_w / 2, 0.0),
        "left": (-(radius + LABEL_GAP + text_w / 2), 0.0),
    }
    return (*offsets[side], text_w / 2, text_h / 2)


def node_box(diameter, label, side="below"):
    """
    Bounding box of a node together with its label (plus padding).
    Returns (half_width, half_height, offset_x, offset_y): the box center is
    offset from the node center by (offset_x, offset_y), in points.
    """
    radius = diameter / 2
    lx, ly, lw, lh = label_rect(diameter, label, side)
    left, right = min(-radius, lx - lw), max(radius, lx + lw)
    bottom, top = min(-radius, ly - lh), max(radius, ly + lh)
    return ((right - left) / 2 + LABEL_PADDING, (top - bottom) / 2 + LABEL_PADDING,
            (right + left) / 2, (top + bottom) / 2)


def _node_overlap_shifts(centers, half, rng):
    """
    Shifts that separate intersecting node boxes, along the axis with the
    smallest penetration (moves nodes as little as possible).
    Returns (shifts, number_of_overlapping_pairs).
    """
    delta = centers[:, None, :] - centers[None, :, :]            # pairwise offsets
    overlap = half[:, None, :] + half[None, :, :] - np.abs(delta)  # > 0 on both axes = intersecting
    intersecting = (overlap[..., 0] > 0) & (overlap[..., 1] > 0)
    np.fill_diagonal(intersecting, False)

    shifts = np.zeros_like(centers)
    pairs = list(zip(*np.nonzero(np.triu(intersecting))))
    for i, j in pairs:
        axis = 0 if overlap[i, j, 0] < overlap[i, j, 1] else 1
        direction = np.sign(delta[i, j, axis])
        if direction == 0:  # identical coordinates: pick a random side
            direction = rng.choice([-1.0, 1.0])
        push = overlap[i, j, axis] / 2 + 0.5
        shifts[i, axis] += direction * push
        shifts[j, axis] -= direction * push
    return shifts, len(pairs)


def _segments_crossing_boxes(starts, ends, centers, half):
    """
    Boolean matrix (n_boxes, n_segments): True where the segment intersects
    the axis-aligned box (Liang-Barsky clipping, vectorized).
    """
    d = (ends - starts)[None, :, :]                       # (1, E, 2)
    lo = (centers - half)[:, None, :] - starts[None]      # (N, E, 2)
    hi = (centers + half)[:, None, :] - starts[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / d, hi / d
    t_enter = np.minimum(t1, t2)
    t_exit = np.maximum(t1, t2)
    # Segment parallel to an axis: inside the slab for every t, or never
    parallel = d == 0
    inside_slab = (lo <= 0) & (hi >= 0)
    t_enter = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), t_enter)
    t_exit = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), t_exit)

    enter = np.maximum(0.0, t_enter.max(axis=-1))
    leave = np.minimum(1.0, t_exit.min(axis=-1))
    return enter < leave


def _edge_crossing_shifts(xy, offsets, half, edges, node_index):
    """
    Shifts that move each node box off the edges crossing it (edges incident
    to the node itself are ignored). The node moves away from the edge along
    the normal; the edge's endpoints move slightly the other way.
    Returns (shifts, number_of_crossings).
    """
    shifts = np.zeros_like(xy)
    if not edges:
        return shifts, 0

    src = np.array([node_index[u] for u, _ in edges])
    dst = np.array([node_index[v] for _, v in edges])
    starts, ends = xy[src], xy[dst]
    centers = xy + offsets

    crossing = _segments_crossing_boxes(starts, ends, centers, half)
    own_edge = (np.arange(len(xy))[:, None] == src[None]) | (np.arange(len(xy))[:, None] == dst[None])
    crossing &= ~own_edge

    pairs = list(zip(*np.nonzero(crossing)))
    for i, e in pairs:
        a, b = starts[e], ends[e]
        seg = b - a
        seg_len2 = seg @ seg
        t = np.clip((centers[i] - a) @ seg / seg_len2, 0, 1) if seg_len2 > 0 else 0.0
        closest = a + t * seg
        normal = centers[i] - closest
        dist = np.hypot(*normal)
        if dist < 1e-9:  # center lies on the edge: push perpendicular to it
            normal = np.array([-seg[1], seg[0]])
            dist = np.hypot(*normal) or 1.0
        normal /= dist
        # Distance the box needs from the line along `normal` to stop touching it
        needed = abs(normal[0]) * half[i, 0] + abs(normal[1]) * half[i, 1]
        push = needed - dist + 1.0
        shifts[i] += normal * push * 0.6
        shifts[src[e]] -= normal * push * 0.2
        shifts[dst[e]] -= normal * push * 0.2
    return shifts, len(pairs)


def choose_label_sides(G, pos, diameters, labels):
    """
    Picks, per node, the label side (LABEL_SIDES order = preference) crossed
    by the fewest of the node's own edges. Edges of other nodes are handled
    later by moving nodes, so they are not considered here.
    """
    sides = {}
    for n in diameters:
        neighbors = list(nx.all_neighbors(G, n))
        if not neighbors:
            sides[n] = LABEL_SIDES[0]
            continue
        node_xy = np.asarray(pos[n], dtype=float)
        starts = np.repeat(node_xy[None], len(neighbors), axis=0)
        ends = np.array([pos[m] for m in neighbors], dtype=float)

        def crossings(side):
            ox, oy, hw, hh = label_rect(diameters[n], labels[n], side)
            center = node_xy[None] + [ox, oy]
            return int(_segments_crossing_boxes(starts, ends, center, np.array([[hw, hh]])).sum())

        sides[n] = min(LABEL_SIDES, key=crossings)  # min() keeps the first side on ties
    return sides


def resolve_collisions(pos, boxes, edges, max_iterations=600):
    """
    Iteratively removes node/node overlaps and node/edge crossings.

    pos:   {node: (x, y)} in points
    boxes: {node: (half_width, half_height, offset_x, offset_y)} in points (see node_box)
    edges: list of (u, v) drawn as straight segments between node centers
    Prints a notice if some crossings could not be removed.
    """
    nodes = list(pos.keys())
    if len(nodes) < 2:
        return pos

    node_index = {n: i for i, n in enumerate(nodes)}
    xy = np.array([pos[n] for n in nodes], dtype=float)
    offsets = np.array([boxes[n][2:] for n in nodes], dtype=float)
    half = np.array([boxes[n][:2] for n in nodes], dtype=float)
    rng = np.random.default_rng(42)

    node_overlaps = edge_crossings = 0
    for _ in range(max_iterations):
        node_shift, node_overlaps = _node_overlap_shifts(xy + offsets, half, rng)
        edge_shift, edge_crossings = _edge_crossing_shifts(xy, offsets, half, edges, node_index)
        if node_overlaps == 0 and edge_crossings == 0:
            break
        xy += node_shift + edge_shift

    if node_overlaps or edge_crossings:
        print(f"Notice: layout kept {node_overlaps} node overlap(s) and "
              f"{edge_crossings} edge crossing(s) after {max_iterations} iterations.")
    return dict(zip(nodes, xy))


# ---------- Main entry point ----------
def plot_graph(G, output_pdf="graph.pdf", output_png=None, show=True, figsize=None):
    """
    Renders a TOR graph and exports it to vector PDF (and optionally PNG).

    Node size  -> support (linear ranking)
    Node color -> confidence (light colormap)
    `figsize` is optional: by default it is derived from the layout so that
    nodes and labels never overlap.
    """
    if len(G.nodes) == 0:
        print("Empty graph, nothing to render.")
        return

    roots = set(find_root_nodes(G))
    visible = [n for n in G.nodes if n not in roots]
    if not visible:
        print("Graph only contains the root node, nothing to render.")
        return

    # ---------- Node attributes (root excluded so it doesn't skew the scales) ----------
    support = [float(G.nodes[n].get("support", 0.0)) for n in visible]
    confidence = [float(G.nodes[n].get("confidence", 0.0)) for n in visible]
    labels = {n: str(G.nodes[n].get("value", n)) for n in visible}

    diameters = dict(zip(visible, rank_scale(support, *NODE_DIAMETER_RANGE)))

    c_min, c_max = min(confidence), max(confidence)
    if c_max == c_min:
        c_max = c_min + 1.0
    norm = Normalize(vmin=c_min, vmax=c_max)
    colors = [LIGHT_CMAP(norm(c)) for c in confidence]

    # ---------- Layout + overlap removal ----------
    pos = scale_to_points(compute_layout(G))
    # TOR: single nodes close to the center, clusters further out
    if len(roots) == 1:
        center = next(iter(roots))
        footprints = {n: node_footprint(diameters[n], labels[n]) for n in visible}
        footprints[center] = 0.0
        pos = arrange_around_center(G, pos, center, footprints)

    # Each label goes on the side its own edges leave free. Edge directions shift
    # slightly while resolving collisions, so sides are re-checked until stable.
    edges = list(G.edges)
    label_sides = {}
    for _ in range(3):
        new_sides = choose_label_sides(G, pos, diameters, labels)
        if new_sides == label_sides:
            break
        label_sides = new_sides
        # The invisible root only needs a tiny footprint
        boxes = {n: (1.0, 1.0, 0.0, 0.0) if n in roots
                 else node_box(diameters[n], labels[n], label_sides[n])
                 for n in G.nodes}
        pos = resolve_collisions(pos, boxes, edges)

    remaining = count_edge_crossings(pos, edges)
    if remaining:
        print(f"Notice: {remaining} edge-edge crossing(s) could not be avoided.")

    # ---------- Figure sized to the layout (1 data unit == 1 pt) ----------
    centers = np.array([pos[n] + np.array(boxes[n][2:]) for n in pos])
    half_arr = np.array([boxes[n][:2] for n in pos])
    x_min, y_min = (centers - half_arr).min(axis=0) - MARGIN
    x_max, y_max = (centers + half_arr).max(axis=0) + MARGIN
    width_pt, height_pt = x_max - x_min, y_max - y_min

    colorbar_pt = 80.0
    if figsize is None:
        figsize = ((width_pt + colorbar_pt) / POINTS_PER_INCH, height_pt / POINTS_PER_INCH)
    fig = plt.figure(figsize=figsize)
    graph_frac = width_pt / (width_pt + colorbar_pt)
    ax = fig.add_axes([0, 0, graph_frac, 1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- Drawing ----------
    # node_size (pt^2) is passed for every node so arrows stop at the node border;
    # the root gets size 0, so its edges meet at a single invisible point.
    edge_node_sizes = [0 if n in roots else diameters[n] ** 2 for n in G.nodes]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        nodelist=list(G.nodes),
        node_size=edge_node_sizes,
        edge_color="#9e9e9e", alpha=0.6, width=0.8,
        arrows=G.is_directed(), arrowsize=8,
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=visible,
        node_size=[diameters[n] ** 2 for n in visible],
        node_color=colors,
        edgecolors="#7a7a7a", linewidths=0.5,
    )

    # Labels are centered on their chosen rectangle; a thin white halo keeps them readable
    halo = [path_effects.withStroke(linewidth=2.5, foreground="white")]
    for n in visible:
        ox, oy, _, _ = label_rect(diameters[n], labels[n], label_sides[n])
        ax.text(pos[n][0] + ox, pos[n][1] + oy, labels[n], ha="center", va="center",
                fontsize=FONT_SIZE, color="#1a1a1a", path_effects=halo, zorder=3)

    colorbar_ax = fig.add_axes([graph_frac + 0.15 * (1 - graph_frac), 0.3,
                                0.2 * (1 - graph_frac), 0.4])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=LIGHT_CMAP),
                 cax=colorbar_ax, label="Confidence")

    # ---------- Export ----------
    for path, kwargs in ((output_pdf, {"format": "pdf"}), (output_png, {"dpi": 300})):
        if not path:
            continue
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(path, **kwargs)
        print(f"Graph saved to: {path}")

    if show:
        plt.show()
    plt.close(fig)
