"""Main entrypoint for mulink"""

from collections.abc import KeysView

import mudata as md
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_array

from mulink.schema import LinkSchema, MuLinkSchema

_MULINK_SUPPORTED_AXES = (0,)
"""Supported axis of mudata objects"""


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata

    Functionalities are extended via accessors:

        - query: Querying functionalities
        - pl: Plotting functionalities
    """

    def __init__(self, mdata: md.MuData) -> None:
        self._obj = mdata

    def add_link(self, link: csr_array, *, key: str = "feature_mapping", **link_properties) -> None:
        """Add a feature link to mudata

        Parameters
        ----------
        link
            Adjacency matrix defining the link graph
        key
            Name of slot in `mudata.varp`.
        link_properties
            Annotation of link.
            Passed to :class:`mulink.schema.LinkSchema` and added to `mdata.uns["mulink"]`
        """
        if self._obj.axis not in (_MULINK_SUPPORTED_AXES):
            raise NotImplementedError(f"Shared axis {self._obj.axis} in mudata object is not supported.")

        annotation = LinkSchema(**link_properties)

        if self._obj.uns.get(MuLinkSchema._MULINK_KEY) is None:
            schema = MuLinkSchema(links={key: annotation})
        else:
            schema = MuLinkSchema.from_mudata(self._obj)
            schema.add_link(key=key, link=annotation)

        self._obj.varp[key] = csr_array(link)
        schema.to_mudata(mdata=self._obj)

    @property
    def link_keys(self) -> KeysView:
        """Names of available keys in object"""
        return MuLinkSchema.from_mudata(self._obj).keys()

    def link(self, key: str | None = None) -> pd.DataFrame:
        """Returns the feature mapping matrix of the current object

        Parameters
        ----------
        key
            Key of feature mapping matrix in `.varp` to use.
        """
        if key is None:
            keys = list(self.link_keys)

            # For multiple keys, the order is not preserved in I/O wround trips
            # Make intent explicit for ambiguous scenarios
            if len(keys) > 1:
                raise ValueError("More than 1 available key, please explicitly set the ``key`` argument")
            key = keys[0]

        matrix = self._obj.varp[key]
        return pd.DataFrame.sparse.from_spmatrix(matrix, index=self._obj.var_names, columns=self._obj.var_names).astype(
            pd.SparseDtype(matrix.dtype, fill_value=0)
        )

    def meta(self, key: str | None = None) -> LinkSchema:
        """Returns the link metadata from the current object

        Parameters
        ----------
        key
            Name of the link. If `None`, uses first defined key.

        Returns
        -------
        The LinkSchema of the link `key`
        """
        if key is None:
            keys = list(self.link_keys)
            # For multiple keys, the order is not preserved in I/O wround trips
            # Make intent explicit for ambiguous scenarios
            if len(keys) > 1:
                raise ValueError("More than 1 available key, please explicitly set the ``key`` argument")
            key = keys[0]

        schema = MuLinkSchema.from_mudata(self._obj)
        return schema[key]

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
