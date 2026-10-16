from collections.abc import KeysView
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, Field

if TYPE_CHECKING:
    import mudata as md


_MULINK_VERSION_REGEX: str = r"^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$"
"""Version of mulink schema (SchemaVer versioning <MODEL>-<REVISION>-<ADDITION>)

MODEL when you make a breaking schema change which will prevent interaction with any historical data
REVISION when you make a schema change which may prevent interaction with some historical data
ADDITION when you make a schema change that is compatible with all historical data
"""

_MULINK_SUPPORTED_AXES = ("var",)


def mudata_axis(axis: str) -> str:
    """Axis in MuData object"""
    if axis not in _MULINK_SUPPORTED_AXES:
        raise ValueError(f"`axis` must be 'var', not '{axis}'")

    return axis


# TODO: Versioning
class LinkSchema(BaseModel):
    """Metadata of a single link

    The link's name is not part of the model, it is the key of the link in :attr:`MuLinkSchema.links`.
    """

    axis: Annotated[Literal["var"], AfterValidator(mudata_axis)] = Field(
        ..., description="Axis of MuData object to which the link applies. One of `obs` or `var`."
    )
    semantics: str | None = Field(None, description="Semantic meaning of link.")


# TODO: Versioning
class MuLinkSchema(BaseModel):
    """Metadata schema for MuLink links

    Supports dict-style access to links by name: `schema.keys()`, `schema[key]` and `key in schema`.
    """

    model_config = {"extra": "forbid"}

    _MULINK_KEY: ClassVar = "mulink"

    version: str = Field(
        "1-0-0",
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
        """Validate that every link exists in the pairwise attribute of its axis

        Raises
        ------
        KeyError
            If a link name is missing in the pairwise attribute.
        """
        for key, link in self.links.items():
            pairwise_attribute = f"{link.axis}p"
            if key not in getattr(mdata, pairwise_attribute):
                raise KeyError(f"Link `{key}` does not exist in `mdata.{pairwise_attribute}`")

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
            raise KeyError("Expected link metadata in `mdata.uns['{cls._MULINK_KEY}']` but the key does not exist")

        return cls.model_validate(mdata.uns[cls._MULINK_KEY])
