"""Test mulink schema definition"""

import anndata as ad
import mudata as md
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.sparse import csr_matrix

from mulink.schema import LinkSchema, MuLinkSchema


@pytest.fixture
def mdata() -> md.MuData:
    """Minimal mudata object"""

    modalities = [
        {"name": "mod1", "shape": (3, 2)},
        {"name": "mod1", "shape": (3, 4)},
        {"name": "mod3", "shape": (3, 6)},
    ]
    mdata = md.MuData({mod["name"]: ad.AnnData(X=np.zeros(shape=mod["shape"])) for mod in modalities})
    mdata.var_names_make_unique()
    return mdata


@pytest.fixture
def adjacency_matrix(mdata) -> csr_matrix:
    """Minimal adjacency matrix for mudata object with no connections"""

    return csr_matrix(np.zeros(shape=(mdata.n_vars, mdata.n_vars)))


@pytest.fixture
def mdata_with_varp(mdata: md.MuData, adjacency_matrix: csr_matrix) -> md.MuData:
    """Create mdata with varp mapping"""
    _mdata = mdata.copy()
    _mdata.varp["feature_mapping"] = adjacency_matrix

    return _mdata


class TestLinkSchema:
    """Test LinkSchema"""

    @pytest.mark.parametrize(
        "_input",
        [
            {"axis": "var", "semantics": "aggregates_to"},
            {"axis": "var", "semantics": None},
        ],
        ids=("complete", "missing_semantics"),
    )
    def test_linkschema_validate__succeeds(self, _input) -> None:
        LinkSchema.model_validate(_input)

    @pytest.mark.parametrize(
        "axis",
        ["unsupported_axis", 1],
        ids=("unsupported", "integer_value"),
    )
    def test_link_schema_validate__fails_incorrect_axis(self, axis) -> None:
        """Check that the LinkSchema fails if an incorrect axis is specified"""

        with pytest.raises(ValidationError):
            LinkSchema.model_validate({"axis": axis, "semantics": None})


class TestMuLinkSchema:
    """Test MuLinkSchema"""

    @pytest.fixture(params=[{"axis": "var"}])
    def correct_link_schema(self, request):
        """Correct schema"""
        return request.param

    @pytest.fixture
    def link(self, correct_link_schema) -> dict:
        return LinkSchema.model_validate(correct_link_schema).model_dump(mode="python")

    @pytest.fixture
    def links(self, link) -> dict[str, dict]:
        return {"feature_mapping": link}

    @pytest.fixture
    def schema(self, links) -> MuLinkSchema:
        return MuLinkSchema(version="1-0-0", links=links)

    def test_mulinkschema_dict_access(self, schema) -> None:
        assert set(schema.keys()) == {"feature_mapping"}
        assert "feature_mapping" in schema
        assert schema["feature_mapping"] == LinkSchema(axis="var")

    def test_mulinkschema_add_link__succeeds(self, schema) -> None:
        schema.add_link(key="new_link", link=LinkSchema(axis="var", semantics="maps_to"))

        assert schema["new_link"].semantics == "maps_to"

    def test_mulinkschema_add_link__fails_redundant_key(self, schema) -> None:
        with pytest.raises(ValueError, match="already taken"):
            schema.add_link(key="feature_mapping", link=LinkSchema(axis="var"))

    def test_validate_pkeys_exist__succeeds(self, schema, mdata_with_varp) -> None:
        schema.validate_pkeys_exist(mdata_with_varp)

    def test_validate_pkeys_exist__fails_missing_pkey(self, mdata_with_varp) -> None:
        schema = MuLinkSchema(links={"missing_key": LinkSchema(axis="var")})

        with pytest.raises(KeyError, match="missing_key"):
            schema.validate_pkeys_exist(mdata_with_varp)

    @pytest.mark.parametrize(
        "version",
        ["0", "0-0", "0-0-1a", "0-01-0", "1.0.0"],
        ids=("model-only", "model-revision-only", "added-suffix", "prefix-zero", "dotted-version"),
    )
    def test_mulinkschema_validate__fails_incorrect_version(self, version: str, links) -> None:
        """Raise ValidationError when incorrect versions are passed"""

        with pytest.raises(ValidationError):
            MuLinkSchema.model_validate({"version": version, "links": links})

    def test_mulinkschema_validate__fails_single_link(self, link: LinkSchema) -> None:
        """Raise ValidationError when only a single link is passed"""

        with pytest.raises(ValidationError):
            MuLinkSchema.model_validate({"version": "1-0-0", "links": link})

    def test_mulinkschema_from_mudata__succeeds(self, mdata_with_varp, links) -> None:
        mdata = mdata_with_varp.copy()

        schema = MuLinkSchema(version="1-0-0", links=links)
        mdata.uns["mulink"] = schema.model_dump(mode="json")

        inferred_schema = MuLinkSchema.from_mudata(mdata)

        assert schema == inferred_schema

    def test_mulinkschema_from_mudata__missing_key(self, mdata_with_varp) -> None:
        """No mulink metadata available"""

        with pytest.raises(KeyError, match="key does not exist"):
            MuLinkSchema.from_mudata(mdata_with_varp)

    @pytest.mark.parametrize(
        "link_metadata",
        [
            "incorrect_schema",
            {"key": "incorrect_schema"},
            {"version": "1.0.0", "links": {"feature_mapping": {"axis": "var"}}},
            {"version": "1-0-0", "links": [{"key": "feature_mapping", "axis": "var"}]},
        ],
        ids=("string", "incorrect-dict", "wrong-version-schema", "list-of-links"),
    )
    def test_mulinkschema_from_mudata__validation_fails(self, mdata_with_varp, link_metadata) -> None:
        """No mulink metadata available"""

        mdata = mdata_with_varp.copy()
        mdata.uns[MuLinkSchema._MULINK_KEY] = link_metadata

        with pytest.raises(ValidationError):
            MuLinkSchema.from_mudata(mdata)

    def test_mulinkschema_to_mudata(self, mdata_with_varp, links) -> None:
        """Round trip test"""
        mdata = mdata_with_varp.copy()
        schema = MuLinkSchema(version="1-0-0", links=links)

        schema.to_mudata(mdata)

        new_schema = MuLinkSchema.from_mudata(mdata)

        assert schema == new_schema

    def test_mulinkschema_io(self, tmp_path, mdata_with_varp, links) -> None:
        """Test IO"""
        mdata = mdata_with_varp.copy()
        schema = MuLinkSchema(version="1-0-0", links=links)
        write_path = tmp_path / "test.h5mu"

        schema.to_mudata(mdata)

        mdata.write_h5mu(write_path)

        new_mdata = md.read_h5mu(write_path)
        new_schema = MuLinkSchema.from_mudata(new_mdata)
        assert schema == new_schema
