"""Tests for ``_merge_close_nodes`` in skeleton.py.

The function used to rescan every node pair from scratch after each merge
(``while changed: ... break  # restart after modification``), making it
O(merges x n^2). On a real 3.9k-node wing skeleton it burned ~568s of the
571s skeleton "finish" half — the half whose entire purpose is to be the
cheap, cacheable path for the live-tuning preview.

It now finds close pairs once with a KD-tree. That is sound because a merge
only ever REMOVES a node — the kept node keeps its own coordinates — so no
node ever moves and the close-pair set can only shrink.

These tests pin the behavior the rewrite had to preserve: which node survives,
what happens to the dropped node's edges, and the strict ``<`` boundary.
"""

from __future__ import annotations

import networkx as nx
from identify_features.models.skeleton import _merge_close_nodes


def _g(*nodes: tuple[int, float, float]) -> nx.Graph:
    G = nx.Graph()
    for n, x, y in nodes:
        G.add_node(n, x=x, y=y)
    return G


def test_merges_pair_closer_than_min_dist():
    G = _g((0, 0.0, 0.0), (1, 3.0, 4.0))  # 3-4-5 triangle -> dist 5
    _merge_close_nodes(G, min_dist=10.0)
    assert G.number_of_nodes() == 1


def test_leaves_pair_at_exactly_min_dist():
    """The test is strict ``<``: a pair exactly min_dist apart must survive."""
    G = _g((0, 0.0, 0.0), (1, 10.0, 0.0))
    _merge_close_nodes(G, min_dist=10.0)
    assert G.number_of_nodes() == 2


def test_leaves_distant_pair():
    G = _g((0, 0.0, 0.0), (1, 100.0, 0.0))
    _merge_close_nodes(G, min_dist=10.0)
    assert G.number_of_nodes() == 2


def test_keeps_higher_degree_node():
    """Ties go to the first node; otherwise the busier junction survives."""
    G = _g((0, 0.0, 0.0), (1, 1.0, 0.0), (2, 50.0, 0.0), (3, 60.0, 0.0))
    G.add_edge(1, 2)
    G.add_edge(1, 3)  # node 1 has degree 2, node 0 has degree 0
    _merge_close_nodes(G, min_dist=5.0)
    assert 1 in G and 0 not in G


def test_dropped_nodes_edges_transfer_to_kept():
    G = _g((0, 0.0, 0.0), (1, 1.0, 0.0), (2, 80.0, 0.0))
    G.add_edge(0, 2, length_px=80.0)  # node 0 is dropped (degree tie -> keeps 0? no: 0 has deg 1)
    _merge_close_nodes(G, min_dist=5.0)
    survivor = 0 if 0 in G else 1
    assert G.has_edge(survivor, 2), "the dropped node's neighbor must be reconnected"
    assert G.number_of_nodes() == 2


def test_direct_edge_between_merged_pair_is_dropped():
    """A tiny connecting segment between the pair is removed, not self-looped."""
    G = _g((0, 0.0, 0.0), (1, 1.0, 0.0))
    G.add_edge(0, 1, length_px=1.0)
    _merge_close_nodes(G, min_dist=5.0)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0


def test_chain_does_not_collapse_transitively():
    """Merging does NOT chain through the survivor's position.

    0-4-8 with min_dist=5: (0,1) and (1,2) are each 4 apart, but merging 1
    into 0 keeps node 0's coordinates, leaving node 2 eight away — so it
    survives. Pinned because it is the exact property that lets the close-pair
    set be computed once up front: a merge removes a node, it never moves one,
    so no merge can create a new close pair.
    """
    G = _g((0, 0.0, 0.0), (1, 4.0, 0.0), (2, 8.0, 0.0))
    _merge_close_nodes(G, min_dist=5.0)
    assert sorted(G.nodes()) == [0, 2]


def test_degenerate_inputs_are_noops():
    for min_dist in (0.0, -5.0):
        G = _g((0, 0.0, 0.0), (1, 1.0, 0.0))
        _merge_close_nodes(G, min_dist=min_dist)
        assert G.number_of_nodes() == 2, f"min_dist={min_dist} must not merge"

    empty = nx.Graph()
    _merge_close_nodes(empty, min_dist=10.0)
    assert empty.number_of_nodes() == 0

    single = _g((0, 0.0, 0.0))
    _merge_close_nodes(single, min_dist=10.0)
    assert single.number_of_nodes() == 1


def test_nodes_without_coordinates_are_ignored():
    """Defensive: a node missing x/y must not raise."""
    G = _g((0, 0.0, 0.0), (1, 1.0, 0.0))
    G.add_node(99)  # no x/y
    _merge_close_nodes(G, min_dist=5.0)
    assert 99 in G


def test_scales_to_a_large_graph():
    """Guards the complexity regression: this took minutes with the old loop."""
    import time

    G = nx.Graph()
    for i in range(4000):
        G.add_node(i, x=float(i % 200) * 3.0, y=float(i // 200) * 3.0)
    start = time.perf_counter()
    _merge_close_nodes(G, min_dist=4.0)
    assert time.perf_counter() - start < 10.0, "merge_close_nodes regressed to a rescan-per-merge loop"
