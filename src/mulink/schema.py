import re
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator

if TYPE_CHECKING:
    import mudata as md


def mulink_schema_version(version: str):
    """Semantic vesioning with <major>.<minor>.<patch> syntax

    Notes
    -----
    Update <major> if the schema structure significantly changes (e.g. nesting structure).
    Update <minor> if downstream incompatible changes occur (e.g. key renaming) occur.
    Update <patch> if downstream compatible changes (new keys) occur.
    """
    VERSION_REGEX = r"^(?!0\.0\.0$)(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"

    if not re.match(VERSION_REGEX, version):
        raise ValueError(f"{version} did not match expected structure: {VERSION_REGEX}")
    return version


def mudata_axis(axis: str) -> str:
    """Axis in MuData object"""
    if axis not in ("var",):
        raise ValueError(f"`axis` must be 'var', not '{axis}'")

    return axis


# TODO: Versioning
class LinkSchema(BaseModel):
    """Definition of a single Link"""

    key: str
    axis: Annotated[str, AfterValidator(mudata_axis)] = Field(
        ..., description="Axis of MuData object to which the link applies. One of `obs` or `var`."
    )
    mods: Literal["all"] | list[str] = Field("all", description="Included modalities (`.mod`) in link.")
    semantic: str | None = Field(None, description="Semantic meaning of link.")

    def validate_modalities_exist(self, mdata: "md.MuData") -> None:
        """Validate that added modalities exist in mudata object"""
        if len(mdata.mod_names) == 0:
            raise KeyError("No modalities in mudata object. Cannot add links.")
        if self.mods == "all":
            return
        for key in self.mods:
            if key not in mdata.mod_names:
                raise KeyError(f"Key {key} does not exist")


# TODO: Versioning
class MuLinkSchema(BaseModel):
    """Schema for MuLink"""

    _MULINK_KEY: ClassVar = "mulink"

    version: Annotated[str, AfterValidator(mulink_schema_version)] = Field("v0.0.1", description="Version of schema")
    links: list[LinkSchema] = Field(..., description="Link definitions.")

    @field_validator("links")
    def link_keys_unique(cls, links: list[LinkSchema]) -> list[LinkSchema]:
        """MuLink keys must be unique in the object"""
        # TODO: Evaluate whether obs/var should be separately enforced
        if len({link.key for link in links}) != len(links):
            raise ValueError("Redundant links")
        return links

    def validate_modalities_exist(self, mdata: "md.MuData") -> None:
        """Validate that added modalities exist in mudata object"""
        for link in self.links:
            link.validate_modalities_exist(mdata)

    def add_link(self, link: LinkSchema) -> None:
        """Add a link to the model"""
        new_link = LinkSchema.model_validate(link)

        if any(new_link.key == link.key for link in self.links):
            raise ValueError(f"Adding redundant key. `key={new_link.key}` is already taken.")

        self.links.append(new_link)

    def to_mudata(self, mdata: "md.MuData") -> None:
        """Add model to mudata object inplace"""
        self.validate_modalities_exist(mdata)
        mdata.uns[self._MULINK_KEY] = self.model_dump(mode="python")

    @classmethod
    def from_mudata(cls, mdata: "md.MuData") -> "MuLinkSchema":
        """Initialize from mudata object"""
        return cls(**mdata.uns[cls._MULINK_KEY])
