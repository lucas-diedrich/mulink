"""Main entrypoint for mulink"""

import mudata as md
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_matrix

from mulink.schema import LinkSchema, MuLinkSchema


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata

    Functionalities are extended via accessors:

        - query: Querying functionalities
        - pl: Plotting functionalities
    """

    def __init__(self, mdata: md.MuData) -> None:
        self._obj = mdata

    def add_link(self, link: csr_matrix, *, key: str = "feature_mapping", **link_properties) -> None:
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
        self._obj.varp[key] = csr_matrix(link)

        # TODO: currently only `var` axis supported
        annotation = LinkSchema(key=key, axis="var", **link_properties)

        if self._obj.uns.get(MuLinkSchema._MULINK_KEY) is None:
            schema = MuLinkSchema(links=[annotation])
        else:
            schema = MuLinkSchema.from_mudata(self._obj).add_link(link=annotation)

        schema.to_mudata(mdata=self._obj)

    def link(self, key: str = "feature_mapping") -> pd.DataFrame:
        """Returns the feature mapping matrix of the current object

        Parameters
        ----------
        key
            Key of feature mapping matrix in `.varp` to use.
        """
        matrix = self._obj.varp[key]
        return pd.DataFrame.sparse.from_spmatrix(matrix, index=self._obj.var_names, columns=self._obj.var_names).astype(
            pd.SparseDtype(matrix.dtype, fill_value=0)
        )

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
