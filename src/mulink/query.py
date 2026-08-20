"""Query an adjacency matrix"""

from collections.abc import Callable, Iterable

import mudata as md
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order


def get_descendants(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all descendants for a feature or a list of features

    Descendants represent vertices that can be reached from a node along the
    edge directionality, including self.

    Parameters
    ----------
    vertices
        List of vertices for which the descendants should be queried.
    adjacency_matrix
        Adjacency matrix which indicates that u -> v (u maps to v)
        if (u, v) is nonzero.

    Returns
    -------
    List of direct successors of the provided vertices
    """
    vertices = [vertices] if isinstance(vertices, int) else vertices

    descendants = np.concatenate(
        [
            breadth_first_order(adjacency_matrix, i_start=vertix, directed=True, return_predecessors=False)
            for vertix in vertices
        ]
    )

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(descendants)


def get_ancestors(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all ancestors for a feature or a list of features

    A direct ancestors represents a vertix that can be reached by a single hop against
    edge directionality, including self.

    Returns
    -------
    List of direct ancestors of the provided vertices
    """
    vertices = [vertices] if isinstance(vertices, int) else vertices

    # Transpose adjacency matrix so that edge directions become inverted
    inverted_adjacency_matrix = adjacency_matrix.T
    ancestors = np.concatenate(
        [
            breadth_first_order(inverted_adjacency_matrix, i_start=vertix, directed=True, return_predecessors=False)
            for vertix in vertices
        ]
    )

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(ancestors)


class QueryAccessor:
    """Query functionality for mulink"""

    def __init__(self, link):
        self._link = link
        self._mdata = self._link._obj

    def _query(
        self,
        query_func: Callable[[np.ndarray, csr_matrix], np.ndarray],
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
    ) -> md.MuData:
        adjacency_matrix = self._mdata.varp[key]

        features = [features] if isinstance(features, str) else features
        query_indices = self._mdata.var_names.get_indexer(features)

        result_indices = query_func(vertices=query_indices, adjacency_matrix=adjacency_matrix)

        if not include_self:
            # Drop self
            result_indices = result_indices[~np.isin(result_indices, query_indices)]

        return self._mdata[:, self._mdata.var_names[result_indices]]

    def descendants(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
    ) -> md.MuData:
        """Get descendants of features

        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query.descendants(features="mod0-0")
            mdata.link.query.descendants(features=["mod0-0", "mod0-1"])

        """
        return self._query(
            query_func=get_descendants,
            features=features,
            key=key,
            include_self=include_self,
        )

    def ancestors(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
    ) -> md.MuData:
        """Get ancestors of features

        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query.ancestors(features="mod2-0")
            mdata.link.query.ancestors(features=["mod2-0", "mod2-1"])

        """
        return self._query(
            query_func=get_ancestors,
            features=features,
            key=key,
            include_self=include_self,
        )
