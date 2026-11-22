from collections.abc import KeysView
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, Field, NonNegativeInt

from mulink._axis import link_axis

if TYPE_CHECKING:
    import mudata as md


_MULINK_VERSION_REGEX: str = r"^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$"
"""Version of mulink schema (SchemaVer versioning <MODEL>-<REVISION>-<ADDITION>)

MODEL when you make a breaking schema change which will prevent interaction with any historical data
REVISION when you make a schema change which may prevent interaction with some historical data
ADDITION when you make a schema change that is compatible with all historical data
"""

MULINK_SCHEMA_VERSION: str = "1-0-0"
"""Schema version written by this version of mulink"""

_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({MULINK_SCHEMA_VERSION})
"""Schema versions this version of mulink can read"""


def supported_schema_version(version: str) -> str:
    """Validate that a schema version can be read by this version of mulink

    Raises
    ------
    ValueError
        If the version is not supported.
    """
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported mulink schema version '{version}'. Supported versions: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}. "
            "Data written with a newer schema version requires a newer version of mulink."
        )
    return version


class GraphConstraints(BaseModel):
    """Optional constraints on the link graph

    Acyclicity and the absence of edges within a modality are part of the link format and always enforced,
    hence they are not configurable here.
    """

    model_config = {"extra": "forbid"}

    max_outgoing_edges: NonNegativeInt | None = Field(
        None, description="Maximum number of edges pointing away from any single feature."
    )
    max_incoming_edges: NonNegativeInt | None = Field(
        None, description="Maximum number of edges pointing to any single feature."
    )
    allow_isolated_features: bool = Field(
        True, description="Whether features without any incoming or outgoing edge are allowed."
    )
    allowed_targets: dict[str, list[str]] | None = Field(
        None,
        description=(
            "Mapping from a source modality to the modalities its features may link to. "
            "Modalities that are not listed as keys are unrestricted."
        ),
    )


class LinkSchema(BaseModel):
    """Metadata of a single link"""

    model_config = {"extra": "forbid"}

    semantics: str | None = Field(None, description="Semantic meaning of link.")
    constraints: GraphConstraints = Field(default_factory=GraphConstraints, description="Constraints on link graph.")


class MuLinkSchema(BaseModel):
    """Versioned metadata schema for MuLink links

    Supports dict-style access to links by name: `schema.keys()`, `schema[key]` and `key in schema`.
    """

    model_config = {"extra": "forbid"}

    _MULINK_KEY: ClassVar = "mulink"

    version: Annotated[str, AfterValidator(supported_schema_version)] = Field(
        MULINK_SCHEMA_VERSION,
        description="Version of schema",
        pattern=_MULINK_VERSION_REGEX,
    )
    links: dict[str, LinkSchema] = Field(
        ...,
        description="Link definitions, corresponding to the link-name in the pairwise attribute (e.g. `mdata.varp`).",
    )

    def keys(self) -> KeysView[str]:
        """Names of all registered links"""
        return self.links.keys()

    def __getitem__(self, key: str) -> LinkSchema:
        return self.links[key]

    def __contains__(self, key: str) -> bool:
        return key in self.links

    def validate_pkeys_exist(self, mdata: "md.MuData") -> None:
        """Validate that every link exists in the pairwise attribute of the linked axis (`mdata.varp`/`mdata.obsp`)

        Raises
        ------
        KeyError
            If a link name is missing in the pairwise attribute.
        NotImplementedError
            If the shared axis of the MuData object does not support links.
        """
        # Links always live on the non-shared axis of the MuData object, so the axis is not stored per link
        # but derived from `mdata.axis`.
        axis = link_axis(mdata)
        pairwise = axis.get_pairwise(mdata)
        for key in self.links:
            if key not in pairwise:
                raise KeyError(f"Link `{key}` does not exist in `mdata.{axis.pairwise}`")

    def add_link(self, key: str, link: LinkSchema) -> None:
        """Add a link to the model

        Parameters
        ----------
        key
            Name of the link, identical to its key in the pairwise attribute.
        link
            Metadata of the link.

        Raises
        ------
        ValueError
            If a link with the same name already exists.
        """
        if key in self.links:
            raise ValueError(f"Adding redundant key. `key={key}` is already taken.")

        self.links[key] = LinkSchema.model_validate(link)

    def to_mudata(self, mdata: "md.MuData") -> None:
        """Add model to mudata object inplace"""
        self.validate_pkeys_exist(mdata)

        mdata.uns[self._MULINK_KEY] = self.model_dump(mode="json")

    @classmethod
    def from_mudata(cls, mdata: "md.MuData") -> "MuLinkSchema":
        """Initialize from mudata object"""
        if cls._MULINK_KEY not in mdata.uns:
            raise KeyError(f"Expected link metadata in `mdata.uns['{cls._MULINK_KEY}']` but the key does not exist")

        return cls.model_validate(mdata.uns[cls._MULINK_KEY])
