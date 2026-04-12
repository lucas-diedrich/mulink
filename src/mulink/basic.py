"""Main entrypoint for mulink"""

import mudata as md
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_matrix


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata"""

    def __init__(self, mdata: md.MuData):
        self._obj = mdata

    def add_link(self, link: csr_matrix, *, key: str = "feature_mapping") -> None:
        """Add a feature link to mudata"""
        self._obj.varp[key] = csr_matrix(link)

    @property
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
