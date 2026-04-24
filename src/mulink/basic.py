"""Main entrypoint for mulink"""

from typing import Literal

import mudata as md
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_matrix


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata

    Functionalities are extended via accessors:

        - query: Querying functionalities
        - pl: Plotting functionalities
    """

    _PAIRWISE = {0: "varp", 1: "obsp"}
    _INDICES = {0: "var_names", 1: "obs_names"}

    def __init__(self, mdata: md.MuData):
        self._obj = mdata

    def _attrs(self, axis: Literal[0, 1]) -> tuple[str, str]:
        """Resolve axis to (pairwise_attr, names_attr) on the underlying MuData.

        Follows https://mudata.readthedocs.io/stable/notebooks/axes.html#axes-in-mudata:
        - `axis=0`: observations are shared, features are mapped (`.varp` / `.var_names`).
        - `axis=1`: features are shared, observations are mapped (`.obsp` / `.obs_names`).
        """
        if axis not in (0, 1):
            raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")
        return self._PAIRWISE[axis], self._INDICES[axis]

    def _get_link(self, key: str = "feature_mapping", axis: Literal[0, 1] = 0):
        """Get the linking matrix for an axis."""
        pairwise, _ = self._attrs(axis)
        return getattr(self._obj, pairwise)[key]

    def _get_link_indices(self, axis: Literal[0, 1] = 0, mod: str | None = None):
        """Get the indices of the matrix for an axis.

        Parameters
        ----------
        mod
            Modality. If `None`, uses index of full object
        axis
            Axis/Dimension which is shared in the mudata object.
            - `axis=0`: indicates that observations are shared and features are mapping to one another.
            - `axis=1`: indicates that features are shared and observations are mapping to one another.
        """
        _, names = self._attrs(axis)
        if mod is None:
            return getattr(self._obj, names)
        else:
            return getattr(self._obj[mod], names)

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
            - `axis=0`: indicates that observations are shared and features are mapping to one another.
            - `axis=1`: indicates that features are shared and observations are mapping to one another.
        """
        mapping = self._get_link(key=key, axis=axis)
        indices = self._get_link_indices(axis=axis)

        return pd.DataFrame.sparse.from_spmatrix(mapping, index=indices, columns=indices)

    @property
    def query(self):
        """Querying functionality"""
        from .query import QueryAccessor

        return QueryAccessor(self)

    @property
    def pl(self):
        """Plotting functionality"""
        from .plotting import PlotAccessor

        return PlotAccessor(self)
