from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

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
    """Definition of a single Link"""

    key: str
    axis: Annotated[Literal["var"], AfterValidator(mudata_axis)] = Field(
        ..., description="Axis of MuData object to which the link applies. One of `obs` or `var`."
    )
    semantics: str | None = Field(None, description="Semantic meaning of link.")

    def validate_pkey_exists(self, mdata: "md.MuData") -> None:
        """Validate that key exists in pairwise mapping of mudata object"""
        if self.key not in mdata.varp:
            raise KeyError(f"Passed `key={self.key}` does not exist")


# TODO: Versioning
class MuLinkSchema(BaseModel):
    """Metadata schema for MuLink links"""

    model_config = {"extra": "forbid"}

    _MULINK_KEY: ClassVar = "mulink"

    version: str = Field(
        "1-0-0",
        description="Version of schema",
        pattern=_MULINK_VERSION_REGEX,
    )
    links: list[LinkSchema] = Field(..., description="Link definitions.")

    @field_validator("links")
    def link_keys_unique(cls, links: list[LinkSchema]) -> list[LinkSchema]:
        """MuLink keys must be unique in the object"""
        # TODO: Evaluate whether obs/var should be separately enforced
        if len({link.key for link in links}) != len(links):
            raise ValueError("Redundant links")
        return links

    def validate_pkeys_exist(self, mdata: "md.MuData") -> None:
        """Validate that added modalities exist in mudata object"""
        for link in self.links:
            link.validate_pkey_exists(mdata)

    def add_link(self, link: LinkSchema) -> None:
        """Add a link to the model"""
        new_link = LinkSchema.model_validate(link)

        if any(new_link.key == link.key for link in self.links):
            raise ValueError(f"Adding redundant key. `key={new_link.key}` is already taken.")

        self.links.append(new_link)

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
