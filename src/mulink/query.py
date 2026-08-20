"""Query an adjacency matrix"""

from collections.abc import Callable, Iterable, Mapping

import mudata as md
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order


def get_descendants(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all descendants for a feature or a list of features

    Descendants represent vertices that can be reached from a node along the
    edge directionality.

    Parameters
    ----------
    vertices
        List of vertices for which the descendants should be queried.
    adjacency_matrix
        Adjacency matrix which indicates that u -> v (u maps to v)
        if (u, v) is nonzero.

    Returns
    -------
    List of successors of the provided vertices
    """
    vertices = [vertices] if isinstance(vertices, int) else vertices

    # Exclude self (first node in results as results represent a tree)
    # This is necessary to allow for the option to exclude self from queries
    descendants = np.concatenate(
        [
            breadth_first_order(adjacency_matrix, i_start=vertix, directed=True, return_predecessors=False)[1:]
            for vertix in vertices
        ]
    )

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(descendants)


def get_ancestors(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all ancestors for a feature or a list of features

    A direct ancestors represents a vertix that can be reached from a node against the
    edge directionality.

    Returns
    -------
    List of ancestors of the provided vertices
    """
    vertices = [vertices] if isinstance(vertices, int) else vertices

    # Transpose adjacency matrix so that edge directions become inverted.
    # scipy converts to CSR in `breadth_first_order`, so this prevents repetitive conversions
    # Exclude self (first node in results as results represent a tree)
    # This is necessary to allow for the option to exclude self from queries
    inverted_adjacency_matrix = csr_matrix(adjacency_matrix.T)
    ancestors = np.concatenate(
        [
            breadth_first_order(inverted_adjacency_matrix, i_start=vertix, directed=True, return_predecessors=False)[1:]
            for vertix in vertices
        ]
    )

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(ancestors)


def filter_modality_members(vertices: Iterable[int], varmap: Mapping, mods: Iterable[str]) -> np.ndarray:
    """Filter for all vertices that are member in a modality

    Parameters
    ----------
    vertices
        Global integer position of vertix in mdata object
    varmap
        mudata varmap
    modalities
        Modalities to consider

    Returns
    -------
    Vertices that are members of the provided modalities
    """
    # Flatten as varmap is a 1d array
    allowed_indices = np.concatenate([np.flatnonzero(varmap[mod]) for mod in mods])

    return vertices[np.isin(vertices, allowed_indices)]


class QueryAccessor:
    """Query functionality for mulink"""

    def __init__(self, link) -> None:
        self._link = link
        self._mdata = self._link._obj

    def _query(
        self,
        query_func: Callable[[np.ndarray, csr_matrix], np.ndarray],
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
        mods: str | list[str] | None = None,
    ) -> md.MuData:
        adjacency_matrix = self._mdata.varp[key]

        features = [features] if isinstance(features, str) else features
        query_indices = self._mdata.var_names.get_indexer(features)

        result_indices = query_func(vertices=query_indices, adjacency_matrix=adjacency_matrix)

        if include_self:
            result_indices = np.union1d(result_indices, query_indices)

        if mods is not None:
            mods = [mods] if isinstance(mods, str) else mods
            result_indices = filter_modality_members(vertices=result_indices, varmap=self._mdata.varmap, mods=mods)

        return self._mdata[:, self._mdata.var_names[result_indices]]

    def descendants(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
        include_mods: str | list[str] | None = None,
    ) -> md.MuData:
        """Get descendants of features

        Parameters
        ----------
        features
            Features to query for
        key
            Key in `mdata.varm` that represents the mulink graph
        include_self
            Whether to include the query features in the results.
        include_mods
            Only include features from the provided `mdata.mods` in the results.
            If `None`, includes members of all modalities.


        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query.descendants(features="mod0-0")
            mdata.link.query.descendants(features=["mod0-0", "mod0-1"])

        """
        return self._query(
            query_func=get_descendants, features=features, key=key, include_self=include_self, mods=include_mods
        )

    def ancestors(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
        include_mods: str | list[str] | None = None,
    ) -> md.MuData:
        """Get ancestors of features

        Parameters
        ----------
        features
            Features to query for
        key
            Key in `mdata.varm` that represents the mulink graph
        include_self
            Whether to include the query features in the results.
        include_mods
            Only include features from the provided `mdata.mods` in the results.
            If `None`, includes members of all modalities.

        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query.ancestors(features="mod2-0")
            mdata.link.query.ancestors(features=["mod2-0", "mod2-1"])

        """
        return self._query(
            query_func=get_ancestors, features=features, key=key, include_self=include_self, mods=include_mods
        )
