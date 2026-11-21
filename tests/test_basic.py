import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

import mulink


class TestAddLink:
    def test_add_link__stores_key(self) -> None:
        rna = ad.AnnData(np.array([[1, 2]]))
        rna.var_names = ["a", "b"]
        mdata = md.MuData({"rna": rna})

        adj = csr_array(np.array([[0, 1], [0, 0]]))
        mdata.link.add_link(adj)  # type: ignore

        # varp
        assert "feature_mapping" in mdata.varp
        assert isinstance(mdata.varp["feature_mapping"], csr_array)
        # uns
        assert "mulink" in mdata.uns
        assert "feature_mapping" in mdata.uns["mulink"]["links"]

    def test_add_link__custom_key(self) -> None:
        rna = ad.AnnData(np.array([[1, 2]]))
        rna.var_names = ["a", "b"]
        mdata = md.MuData({"rna": rna})

        adj = csr_array(np.array([[0, 1], [0, 0]]))
        mdata.link.add_link(adj, key="custom")  # type: ignore

        assert "custom" in mdata.varp
        assert "custom" in mdata.uns["mulink"]["links"]

    def test__add_link__multiple_keys(self, simple_mudata) -> None:
        mdata = simple_mudata.copy()
        assert "feature_mapping" in mdata.varp
        assert "feature_mapping" in mdata.uns["mulink"]["links"]

        adj = csr_array(np.zeros(shape=(mdata.n_var, mdata.n_var)))
        mdata.link.add_link(adj, key="custom")  # type: ignore

        # Old keys are retained
        assert "feature_mapping" in mdata.varp
        assert "feature_mapping" in mdata.uns["mulink"]["links"]
        # New keys added
        assert "custom" in mdata.varp
        assert "custom" in mdata.uns["mulink"]["links"]

    def test_add_link__wrong_axis_raises(self):
        mdata = md.MuData(axis=1)  # shared variables

        with pytest.raises(NotImplementedError, match="Shared axis"):
            mdata.link.add_link(link=csr_array([[]]))


class TestLink:
    def test_returns_dataframe(self, simple_mudata) -> None:
        result = simple_mudata.link.link()

        assert isinstance(result, pd.DataFrame)

    def test_index_and_columns_match_var_names(self, simple_mudata) -> None:
        result = simple_mudata.link.link()

        assert_array_equal(result.index, simple_mudata.var_names)
        assert_array_equal(result.columns, simple_mudata.var_names)

    def test_values_match_adjacency(self, simple_mudata) -> None:
        result = simple_mudata.link.link()
        expected = simple_mudata.varp["feature_mapping"].toarray()

        np.testing.assert_array_equal(result.values, expected)


class TestMeta:
    def test_returns_linkschema(self, simple_mudata) -> None:
        result = simple_mudata.link.meta()

        assert isinstance(result, mulink.schema.LinkSchema)

    def test_meta_schema_matches_expectations(self, simple_mudata) -> None:
        result = simple_mudata.link.meta().model_dump(mode="json")
        expected = mulink.schema.LinkSchema(semantics=None).model_dump(mode="json")

        assert result == expected
