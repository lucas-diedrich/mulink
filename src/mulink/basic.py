"""Main entrypoint for mulink"""

from collections.abc import Iterable, KeysView

import mudata as md
import pandas as pd
from mudata import register_mudata_namespace
from scipy.sparse import csr_array

from mulink._axis import LinkAxis, link_axis
from mulink.schema import LinkSchema, MuLinkSchema
from mulink.validate import validate_link


@register_mudata_namespace("link")
class MuLink:
    """Link between modalities in mudata

    Links connect the elements of the non-shared axis of the MuData object:
        - variables (`.varp`) if observations are shared (`mdata.axis==0`)
        - observations (`.obsp`) if variables are shared (`mdata.axis == 1`)

    Functionalities are extended via accessors:

        - query: Querying functionalities
        - pl: Plotting functionalities
    """

    def __init__(self, mdata: md.MuData) -> None:
        self._obj = mdata

    def add_link(
        self, link: csr_array, *, key: str = "feature_mapping", validate: bool = True, **link_properties
    ) -> None:
        """Add a link to mudata

        Parameters
        ----------
        link
            Adjacency matrix defining the link graph
        key
            Name of slot in `mudata.varp` (`mdata.axis == 0`) or `mudata.obsp` (`mdata.axis == 1`).
        validate
            Whether to validate the link graph before adding it, see :meth:`validate`.
        link_properties
            Annotation of link.
            Passed to :class:`mulink.schema.LinkSchema` and added to `mdata.uns["mulink"]`

        Raises
        ------
        NotImplementedError
            If the shared axis of the MuData object does not support links.
        mulink.validate.LinkValidationError
            If `validate` is `True` and the link graph is invalid. The object is not modified.
        """
        annotation = LinkSchema(**link_properties)
        link = csr_array(link)

        if validate:
            validate_link(link, self._axis.get_modality_map(self._obj), annotation.constraints)

        if self._obj.uns.get(MuLinkSchema._MULINK_KEY) is None:
            schema = MuLinkSchema(links={key: annotation})
        else:
            schema = MuLinkSchema.from_mudata(self._obj)
            schema.add_link(key=key, link=annotation)

        self._axis.get_pairwise(self._obj)[key] = link
        schema.to_mudata(mdata=self._obj)

    def link(self, key: str | None = None) -> pd.DataFrame:
        """Returns the link matrix of the current object, indexed by the names of the linked axis

        Parameters
        ----------
        key
            Name of the link. Required if more than one link exists.

        Raises
        ------
        ValueError
            If `key` is `None` and the object does not contain exactly one link.
        """
        key = self._resolve_key(key)
        matrix = self._get_link(key)
        names = self._axis.get_names(self._obj)
        return pd.DataFrame.sparse.from_spmatrix(matrix, index=names, columns=names).astype(
            pd.SparseDtype(matrix.dtype, fill_value=0)
        )

    def meta(self, key: str | None = None) -> LinkSchema:
        """Returns the link metadata from the current object

        Parameters
        ----------
        key
            Name of the link. Required if more than one link exists.

        Returns
        -------
        The LinkSchema of the link `key`

        Raises
        ------
        ValueError
            If `key` is `None` and the object does not contain exactly one link.
        """
        key = self._resolve_key(key)
        schema = MuLinkSchema.from_mudata(self._obj)
        return schema[key]

    @property
    def link_keys(self) -> KeysView[str]:
        """Names of available keys in object"""
        return MuLinkSchema.from_mudata(self._obj).keys()

    @property
    def _axis(self) -> LinkAxis:
        return link_axis(self._obj)

    def _get_link(self, key: str) -> csr_array:
        return self._axis.get_pairwise(self._obj)[key]

    def _resolve_key(self, key: str | None) -> str:
        if key is not None:
            return key
        keys = list(self.link_keys)
        # Dict order in `uns` is not preserved across h5mu/zarr I/O (scverse/anndata#610),
        # so picking "the first" key would be non-deterministic.
        if len(keys) != 1:
            raise ValueError(f"`key` must be given explicitly when the object has {len(keys)} links. Available: {keys}")
        return keys[0]

    @property
    def query(self):
        """Querying functionality

        See Also
        --------
        :class:`mulink.plotting.QueryAccessor`
        """
        from .query import QueryAccessor

        return QueryAccessor(self)

    @property
    def pl(self):
        """Plotting functionality

        See Also
        --------
        :class:`mulink.plotting.PlotAccessor`
        """
        from .plotting import PlotAccessor

        return PlotAccessor(self)

    def validate(self, keys: str | Iterable[str] | None = None) -> None:
        """Validate links against the structural rules of the format and their stored constraints

        Parameters
        ----------
        keys
            Names of the links to validate. Defaults to all links.

        Raises
        ------
        mulink.validate.LinkValidationError
            If a link graph is invalid. The subclass indicates the violated property.
        """
        schema = MuLinkSchema.from_mudata(self._obj)
        keys = schema.keys() if keys is None else [keys] if isinstance(keys, str) else keys

        modality_map = self._axis.get_modality_map(self._obj)
        for key in keys:
            validate_link(self._get_link(key), modality_map, schema[key].constraints)
