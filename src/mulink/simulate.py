"""Simulate artificial mudata with explicit feature relationship"""

from itertools import product

import anndata as ad
import mudata as md
import networkx as nx
import numpy as np
from numpy.random import Generator


def _generate_dag(
    n_level: int = 3,
    n_vertices: int = 2,
    *,
    extra_edge_probability: float | None = 0.2,
    extra_edge_levels: list[int] | None = None,
    transitive_closure: bool = True,
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
        Number of vertices in lowest level
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
    seed
        Random seed (for edge addition).

    Returns
    -------
    dag
        A directed acyclic graph
    n_nodes_per_level
        Mapping between feature level and number of nodes in this level
    """
    rng = rng if rng is not None else np.random.default_rng()

    dag = nx.balanced_tree(r=n_vertices, h=n_level, create_using=nx.DiGraph)
    # Remove root node (extra level)
    dag.remove_node(0)

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


def _generate_anndata(n_obs: int, n_var: int, rng: Generator | None = None) -> ad.AnnData:
    """Generate an anndata object with random values"""
    rng = rng if rng is not None else np.random.default_rng()

    return ad.AnnData(X=rng.random(size=(n_obs, n_var)))


def hierarchical_mudata(
    n_mod: int,
    *,
    n_obs: int = 5,
    n_vertices: int = 2,
    extra_edge_probability: float | None = 0.2,
    extra_edge_levels: list[int] | None = None,
    transitive_closure: bool = True,
    varp_key: str = "feature_mapping",
    random_state: int = 42,
):
    """Generate a mudata object with hierarchical feature relationship

    Parameters
    ----------
    n_mod
        Number of modalities (levels) in the object
    n_obs
        Number of observations in the object
    n_vertices
        Number of vertices in the level with the lowest cardinality
    extra_edge_probability
        Probability of adding an additional edge between vertices of adjacent levels.
        If `None`, the feature relationship between different levels is represented by a tree.
        This means that a feature from level n+1 maps to exactly one feature in level n
    extra_edge_levels
        Constrain the addition of extra edges to these levels of the final mudata object.
        Must not contain the highest level, as these features do not contain additional connections.
    transitive_closure
        Whether to return a graph where features that are indirectly linked are connected with an
        explicit edge.
        For example, for the feature mapping A --> B --> C, the connection A --> C will also be added
    varp_key
        Name of key that contains the feature mapping in `.varp` attribute of the returned :class:`mudata.MuData`
        object
    random_state
        Random state for the simulation


    Returns
    -------
    A :class:`mudata.MuData` object with explicit feature mapping. The individual modalities/feature-levels are indicated
    with `mod{idx}`. The feature mapping is added as adjacency matrix with in the `.varp` attribute as `varp_key`.
    In the matrix, entry (i, j) corresponds to a directed edge from feature i to feature j.
    """
    if any(edge_level > n_mod - 1 for edge_level in extra_edge_levels):
        raise ValueError("extra_edge_levels must only contain levels from (0..n_mod-1)")

    rng = np.random.default_rng(seed=random_state)

    dag, n_nodes_per_level = _generate_dag(
        n_level=n_mod,
        n_vertices=n_vertices,
        extra_edge_probability=extra_edge_probability,
        extra_edge_levels=extra_edge_levels,
        transitive_closure=transitive_closure,
        rng=rng,
    )

    mdata = md.MuData(
        data={
            f"mod{mod}": _generate_anndata(n_obs=n_obs, n_var=n_nodes_per_level[mod], rng=rng) for mod in range(n_mod)
        },
    )

    mdata.varp[varp_key] = nx.adjacency_matrix(dag)

    return mdata
