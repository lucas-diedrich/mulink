"""Main entrypoint for mulink"""

from collections.abc import Callable
from typing import Literal

import mudata as md
import numpy as np
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_matrix

from .query import get_ancestors, get_descendants


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata"""

    _PAIRWISE = {0: "varp", 1: "obsp"}
    _INDICES = {0: "var_names", 1: "obs_names"}

    def __init__(self, mdata: md.MuData):
        self._obj = mdata

    def _attrs(self, axis: Literal[0, 1]) -> tuple[str, str]:
        """Resolve axis to (pairwise_attr, names_attr) on the underlying MuData.

        Follows https://mudata.readthedocs.io/stable/notebooks/axes.html#axes-in-mudata:
        - `axis=0` → observations are shared, features are mapped (`.varp` / `.var_names`).
        - `axis=1` → features are shared, observations are mapped (`.obsp` / `.obs_names`).
        """
        if axis not in (0, 1):
            raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")
        return self._PAIRWISE[axis], self._INDICES[axis]

    def _get_link(self, key: str = "feature_mapping", axis: Literal[0, 1] = 0):
        """Get the linking matrix for an axis."""
        pairwise, _ = self._attrs(axis)
        return getattr(self._obj, pairwise)[key]

    def _get_link_indices(self, axis: Literal[0, 1] = 0):
        """Get the indices of the matrix for an axis."""
        _, names = self._attrs(axis)
        return getattr(self._obj, names)

    def _query(
        self,
        query_func: Callable[[np.ndarray, csr_matrix], np.ndarray],
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        axis: Literal[0, 1] = 0,
        include_self: bool = True,
    ) -> md.MuData:
        adjacency_matrix = self._get_link(key=key, axis=axis)

        features = [features] if isinstance(features, str) else features
        query_indices = self._get_link_indices(axis=axis)

        query_indexer = query_indices.get_indexer(features)
        result_indices = query_func(vertices=query_indexer, adjacency_matrix=adjacency_matrix)

        if include_self:
            result_indices = np.concatenate([query_indexer, result_indices])

        selection = query_indices[result_indices]
        slicer = (slice(None), selection) if axis == 0 else (selection, slice(None))
        return self._obj[slicer]

    def query_descendants(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        axis: Literal[0, 1] = 0,
        include_self: bool = True,
    ) -> md.MuData:
        """Get direct descendants of features

        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query_descendants(features="mod0-0")
            mdata.link.query_descendants(features=["mod0-0", "mod0-1"])

        """
        return self._query(
            query_func=get_descendants,
            features=features,
            key=key,
            axis=axis,
            include_self=include_self,
        )

    def query_ancestors(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        axis: Literal[0, 1] = 0,
        include_self: bool = True,
    ) -> md.MuData:
        """Get direct ancestors of features

        Examples
        --------

        .. code-block:: python

            mdata = mulink.simulate.hierarchical_mudata(n_mod=3)

            mdata.link.query_ancestors(features="mod2-0")
            mdata.link.query_ancestors(features=["mod2-0", "mod2-1"])

        """
        return self._query(
            query_func=get_ancestors,
            features=features,
            key=key,
            axis=axis,
            include_self=include_self,
        )

    def add_link(self, link: csr_matrix, *, key: str = "feature_mapping", axis: Literal[0, 1] = 0) -> None:
        """Add a link to mudata (on `.varp` for `axis=0`, `.obsp` for `axis=1`)."""
        pairwise, _ = self._attrs(axis)
        getattr(self._obj, pairwise)[key] = csr_matrix(link)

    def link(self, key: str = "feature_mapping", axis: Literal[0, 1] = 0) -> pd.DataFrame:
        """Returns the linking matrix of the current object

        Parameters
        ----------
        key
            Key of the linking matrix in `.varp`/`.obsp` to use.
        axis
            Axis/Dimension which is shared in the mudata object.
            - `axis=0` indicates that observations are shared and features are mapping to one another.
            - `axis=1` indicates that features are shared and observations are mapping to one another.
        """
        mapping = self._get_link(key=key, axis=axis)
        indices = self._get_link_indices(axis=axis)

        return pd.DataFrame.sparse.from_spmatrix(mapping, index=indices, columns=indices)
