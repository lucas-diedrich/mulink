"""Resolve the MuData attributes that hold links

Links connect the elements of the non-shared axis of a MuData object, i.e. the axis along which modalities are
concatenated (see https://mudata.readthedocs.io/stable/notebooks/axes.html#axes-in-mudata):

- `axis=0`: observations are shared, links connect variables (`.varp`, `.var_names`, `.varmap`)
- `axis=1`: variables are shared, links connect observations (`.obsp`, `.obs_names`, `.obsmap`)
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata as ad
    import mudata as md
    import numpy as np
    import pandas as pd


@dataclass(frozen=True)
class LinkAxis:
    """Attribute names of the MuData axis that holds links"""

    axis: int
    pairwise: str
    names: str
    modality_map: str

    def get_pairwise(self, mdata: "md.MuData") -> Mapping[str, Any]:
        """Pairwise attribute (`.varp`/`.obsp`) that stores the link matrices"""
        return getattr(mdata, self.pairwise)

    def get_names(self, data: "md.MuData | ad.AnnData") -> "pd.Index":
        """Names (`.var_names`/`.obs_names`) of the linked elements of a MuData object or one of its modalities"""
        return getattr(data, self.names)

    def get_modality_map(self, mdata: "md.MuData") -> Mapping[str, "np.ndarray"]:
        """Mapping (`.varmap`/`.obsmap`) from modality name to the positions of its elements in the MuData object"""
        return getattr(mdata, self.modality_map)

    def slicer(self, selection: "pd.Index") -> tuple:
        """Indexer that subsets a MuData object to `selection` along the linked axis"""
        return (slice(None), selection) if self.axis == 0 else (selection, slice(None))


_LINK_AXES: dict[int, LinkAxis] = {
    0: LinkAxis(axis=0, pairwise="varp", names="var_names", modality_map="varmap"),
    1: LinkAxis(axis=1, pairwise="obsp", names="obs_names", modality_map="obsmap"),
}


def link_axis(mdata: "md.MuData") -> LinkAxis:
    """Linked axis of a MuData object, derived from its shared axis `mdata.axis`

    Raises
    ------
    NotImplementedError
        If the shared axis of the MuData object does not support links (e.g. `axis=-1`, where both axes are shared).
    """
    if mdata.axis not in _LINK_AXES:
        raise NotImplementedError(f"Shared axis {mdata.axis} in mudata object is not supported.")
    return _LINK_AXES[mdata.axis]
