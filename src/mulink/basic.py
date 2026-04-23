"""Main entrypoint for mulink"""

from collections.abc import Callable

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

    def _query(
        self,
        query_func: Callable[[np.ndarray, csr_matrix], np.ndarray],
        features: str | list[str],
        *,
        key: str = "feature_mapping",
        include_self: bool = True,
    ) -> md.MuData:
        adjacency_matrix = self._obj.varp[key]

        features = [features] if isinstance(features, str) else features
        query_indices = self._obj.var_names.get_indexer(features)

        result_indices = query_func(vertices=query_indices, adjacency_matrix=adjacency_matrix)

        if include_self:
            result_indices = np.concatenate([query_indices, result_indices])

        return self._obj[:, self._obj.var_names[result_indices]]

    def query_descendants(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
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
            include_self=include_self,
        )

    def query_ancestors(
        self,
        features: str | list[str],
        *,
        key: str = "feature_mapping",
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
            include_self=include_self,
        )

    def add_link(self, link: csr_matrix, *, key: str = "feature_mapping") -> None:
        """Add a feature link to mudata"""
        self._obj.varp[key] = csr_matrix(link)

    def link(self, key: str = "feature_mapping") -> pd.DataFrame:
        """Returns the feature mapping matrix of the current object

        Parameters
        ----------
        key
            Key of feature mapping matrix in `.varp` to use.
        """
        return pd.DataFrame.sparse.from_spmatrix(
            self._obj.varp[key], index=self._obj.var_names, columns=self._obj.var_names
        )
