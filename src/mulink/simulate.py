"""Simulate artificial mudata with explicit feature relationship"""

from itertools import product
from typing import Literal

import anndata as ad
import mudata as md
import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import Generator

from mulink.basic import MuLink


def _generate_dag(
    n_level: int = 3,
    n_vertices: int = 2,
    min_edges: int = 2,
    *,
    extra_edge_probability: float | None = 0.2,
    extra_edge_levels: list[int] | None = None,
    transitive_closure: bool = False,
    rng: Generator | None = None,
) -> tuple[nx.DiGraph, dict[int, int]]:
    """Generate a hierarchical directed acyclic graph (DAG), starting from a tree.

    Generates a directed acyclic graph, starting from a balanced tree with the desired number of levels.
    Additional edges can be randomly added to the DAG.
    Edges between reachable nodes in the graph can be optionally added explicitly (default).

    Parameters
    ----------
    n_level
        Number of levels
    n_vertices
        Number of vertices at lowest level
    min_edges
        Minimum number of edges between vertices
    extra_edge_probability
        Probability of drawing an extra edge between vertices from two adjacent topological generations.
        If `None`, does not add extra edges.
    extra_edge_levels
        Topological generations where extra edges are allowed.
        If None, all levels are eligible.
    transitive_closure
        Whether to return a graph where vertices that are indirectly linked are connected with an
        explicit edge.
        e.g. for the graph A --> B --> C, the connection A --> C will also be added
    rng
        Random number generator instance.

    Returns
    -------
    dag
        A directed acyclic graph
    n_nodes_per_level
        Mapping between feature level and number of nodes in this level
    """
    rng = rng if rng is not None else np.random.default_rng()

    dag = nx.DiGraph()
    subtrees = []
    for _ in range(n_vertices):
        subtree = nx.balanced_tree(r=min_edges, h=(n_level - 1), create_using=nx.DiGraph)
        subtrees.append(subtree)

    dag = nx.union_all(subtrees, rename=(f"{idx}-" for idx in range(n_vertices)))

    level_to_nodes = dict(enumerate(nx.topological_generations(dag)))

    if extra_edge_probability is not None:
        # Last level does not have any connections
        extra_edge_levels = list(range(len(level_to_nodes) - 1)) if extra_edge_levels is None else extra_edge_levels

        for level in extra_edge_levels:
            for u, v in product(level_to_nodes[level], level_to_nodes[level + 1]):
                if rng.random() < extra_edge_probability:
                    dag.add_edge(u, v)

    dag = dag if not transitive_closure else nx.transitive_closure(dag)
    n_nodes_per_level = {level: len(nodes) for level, nodes in level_to_nodes.items()}

    return dag, n_nodes_per_level


def _generate_anndata(
    n_obs: int,
    n_var: int,
    var_prefix: str | None = None,
    rng: Generator | None = None,
    *,
    obs_prefix: str | None = None,
) -> ad.AnnData:
    """Generate an anndata object with random values"""
    rng = rng if rng is not None else np.random.default_rng()

    # Prefixes make names unique across modalities
    obs = pd.DataFrame(index=[f"{obs_prefix}{idx}" for idx in range(n_obs)]) if obs_prefix is not None else None
    var = pd.DataFrame(index=[f"{var_prefix}{idx}" for idx in range(n_var)]) if var_prefix is not None else None

    return ad.AnnData(X=rng.random(size=(n_obs, n_var)), obs=obs, var=var)


def hierarchical_mudata(
    n_mod: int,
    *,
    axis: Literal[0, 1] = 0,
    n_obs: int = 5,
    n_var: int = 5,
    n_vertices: int = 2,
    min_edges=2,
    extra_edge_probability: float | None = 0.2,
    extra_edge_levels: list[int] | None = None,
    transitive_closure: bool = False,
    key: str = "feature_mapping",
    random_state: int = 42,
) -> md.MuData:
    """Generate a mudata object with hierarchical relationship between the elements of its modalities

    Parameters
    ----------
    n_mod
        Number of modalities (levels) in the object
    axis
        Shared axis of the returned :class:`mudata.MuData` object.
        - If `0`, observations are shared and variables are linked.
        - If `1`, variables are shared and observations are linked.
    n_obs
        Number of shared observations in the object. Only used if `axis=0`.
    n_var
        Number of shared variables in the object. Only used if `axis=1`.
    n_vertices
        Number of vertices in the level with the lowest cardinality
    min_edges
        Minimum number of vertices between adjacent levels
    extra_edge_probability
        Probability of adding an additional edge between vertices of adjacent levels.
        If `None`, the feature relationship between different levels is represented by a tree.
        This means that a feature from level n+1 maps to exactly one feature in level n
    extra_edge_levels
        Constrain the addition of extra edges to these levels of the final mudata object (starting at 0).
        Must not contain the highest level, as these features do not contain additional connections.
    transitive_closure
        Whether to return a graph where features that are indirectly linked are connected with an
        explicit edge.
        For example, for the feature mapping A --> B --> C, the connection A --> C will also be added
    key
        Name of the link in the `.varp` (`axis=0`) or `.obsp` (`axis=1`) attribute of the returned
        :class:`mudata.MuData` object
    random_state
        Random state for the simulation


    Returns
    -------
    A :class:`mudata.MuData` object with explicit link. The individual modalities/levels are indicated
    with `mod{idx}`. The link is added as adjacency matrix:

        - If `axis=0`: To the `.varp` attribute
        - If `axis=1`: To the `.obsp` attribute

    Entry (i, j) in the matrix corresponds to a directed edge from element i to element j.

    Raises
    ------
    ValueError
        If `extra_edge_levels` contains invalid levels or `axis` is not supported.
    """
    if axis not in (0, 1):
        raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")
    if extra_edge_levels is not None and any(edge_level >= n_mod - 1 for edge_level in extra_edge_levels):
        raise ValueError("extra_edge_levels must only contain levels from (0..n_mod-2)")

    rng = np.random.default_rng(seed=random_state)

    dag, n_nodes_per_level = _generate_dag(
        n_level=n_mod,
        n_vertices=n_vertices,
        min_edges=min_edges,
        extra_edge_probability=extra_edge_probability,
        extra_edge_levels=extra_edge_levels,
        transitive_closure=transitive_closure,
        rng=rng,
    )

    if axis == 0:
        modalities = {
            f"mod{mod}": _generate_anndata(n_obs=n_obs, n_var=n_nodes_per_level[mod], var_prefix=f"mod{mod}-", rng=rng)
            for mod in range(n_mod)
        }
    else:
        modalities = {
            f"mod{mod}": _generate_anndata(n_obs=n_nodes_per_level[mod], n_var=n_var, obs_prefix=f"mod{mod}-", rng=rng)
            for mod in range(n_mod)
        }

    mdata = md.MuData(data=modalities, axis=axis)

    # Ensure that the adjacency matrix is stored in topological order, so that nodes belonging to highest priority mod
    # are coming up first, etc.
    adjacency_matrix = nx.adjacency_matrix(dag, nodelist=list(nx.topological_sort(dag)))

    mlink = MuLink(mdata=mdata)
    mlink.add_link(link=adjacency_matrix, key=key, semantics="aggregates_to")

    return mlink._obj
