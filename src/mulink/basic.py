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

    def __init__(self, mdata: md.MuData):
        self._obj = mdata

    def _get_link(self, key: str = "feature_mapping", axis: Literal[0, 1] = 0):
        """Get the linking matrix for an axis

        Parameters
        ----------
        key
            key in mdata.varp (axis=0)/mdata.obsp (axis=1)
        axis
            Axis/Dimension which is shared in the mudata object.
            - `axis=0` indicates that observations are shared and features are mapping to one another.
            - `axis=1` indicates that features are shared and observations are mapping to one another.

        Notes
        -----
        Follows the syntax in https://mudata.readthedocs.io/stable/notebooks/axes.html#axes-in-mudata.
        """
        if axis == 0:
            return self._obj.varp[key]
        elif axis == 1:
            return self._obj.obsp[key]
        else:
            raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")

    def _get_link_indices(self, axis: Literal[0, 1] = 0):
        """Get the indices of the matrix for an axis

        Parameters
        ----------
        axis
            Axis/Dimension which is shared in the mudata object.
            - `axis=0` indicates that observations are shared and features are mapping to one another.
            - `axis=1` indicates that features are shared and observations are mapping to one another.

        Notes
        -----
        Follows the syntax in https://mudata.readthedocs.io/stable/notebooks/axes.html#axes-in-mudata.
        """
        if axis == 0:
            return self._obj.var_names
        elif axis == 1:
            return self._obj.obs_names
        else:
            raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")

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

        if axis == 0:
            return self._obj[:, query_indices[result_indices]]
        elif axis == 1:
            return self._obj[query_indices[result_indices], :]

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
        """Add a feature link to mudata"""
        if axis == 0:
            self._obj.varp[key] = csr_matrix(link)
        elif axis == 1:
            self._obj.obsp[key] = csr_matrix(link)
        else:
            raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")

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
