import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

import mulink  # noqa: F401 — registers the namespace


@pytest.fixture
def simple_mudata() -> md.MuData:
    """2-modality MuData with 1:1 mapping: gene_A -> prot_C, gene_B -> prot_D"""
    rna = ad.AnnData(np.array([[1, 2], [3, 4]]))
    rna.var_names = ["gene_A", "gene_B"]
    rna.obs_names = ["cell_1", "cell_2"]

    prot = ad.AnnData(np.array([[5, 6], [7, 8]]))
    prot.var_names = ["prot_C", "prot_D"]
    prot.obs_names = ["cell_1", "cell_2"]

    mdata = md.MuData({"rna": rna, "prot": prot})
    # var_names: [gene_A, gene_B, prot_C, prot_D]
    adj = csr_matrix(
        np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )
    mdata.varp["feature_mapping"] = adj
    return mdata


@pytest.fixture
def n_to_m_mudata() -> md.MuData:
    """2-modality MuData with N:M mapping: gene_A -> prot_C, gene_A -> prot_D, gene_B -> prot_D"""
    gene = ad.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        var=pd.DataFrame(index=["gene_A", "gene_B"]),
        obs=pd.DataFrame(index=["cell_1", "cell_2"]),
    )

    prot = ad.AnnData(X=np.array([[5, 6], [7, 8]]))
    prot.var_names = ["prot_C", "prot_D"]
    prot.obs_names = ["cell_1", "cell_2"]

    mdata = md.MuData({"gene": gene, "prot": prot})
    # var_names: [gene_A, gene_B, prot_C, prot_D]
    adj = csr_matrix(
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
    )
    mdata.varp["feature_mapping"] = adj
    return mdata


class TestAddLink:
    def test_stores_in_varp(self) -> None:
        rna = ad.AnnData(np.array([[1, 2]]))
        rna.var_names = ["a", "b"]
        mdata = md.MuData({"rna": rna})

        adj = csr_matrix(np.array([[0, 1], [0, 0]]))
        mdata.link.add_link(adj)  # type: ignore

        assert "feature_mapping" in mdata.varp
        assert isinstance(mdata.varp["feature_mapping"], csr_matrix)

    def test_custom_key(self) -> None:
        rna = ad.AnnData(np.array([[1, 2]]))
        rna.var_names = ["a", "b"]
        mdata = md.MuData({"rna": rna})

        adj = csr_matrix(np.array([[0, 1], [0, 0]]))
        mdata.link.add_link(adj, key="custom")  # type: ignore

        assert "custom" in mdata.varp


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


class TestQueryDescendants:
    def test_single_feature_string(self, simple_mudata) -> None:
        result = simple_mudata.link.query_descendants("gene_A")

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_single_feature_list(self, simple_mudata) -> None:
        result = simple_mudata.link.query_descendants(["gene_A"])

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_multiple_features(self, simple_mudata) -> None:
        result = simple_mudata.link.query_descendants(["gene_A", "gene_B"])

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "gene_B", "prot_C", "prot_D"])

    def test_include_self_true(self, simple_mudata) -> None:
        result = simple_mudata.link.query_descendants("gene_A", include_self=True)

        assert "gene_A" in result.var_names

    def test_include_self_false(self, simple_mudata) -> None:
        result = simple_mudata.link.query_descendants("gene_A", include_self=False)

        assert "gene_A" not in result.var_names
        assert_array_equal(list(result.var_names), ["prot_C"])

    def test_n_to_m_mapping(self, n_to_m_mudata) -> None:
        result = n_to_m_mudata.link.query_descendants("gene_A", include_self=False)

        assert_array_equal(sorted(result.var_names), ["prot_C", "prot_D"])


class TestQueryAncestors:
    def test_single_feature(self, simple_mudata):
        result = simple_mudata.link.query_ancestors("prot_C")

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_include_self_false(self, simple_mudata):
        result = simple_mudata.link.query_ancestors("prot_C", include_self=False)

        assert result.n_obs == simple_mudata.n_obs

        assert "prot_C" not in result.var_names
        assert_array_equal(list(result.var_names), ["gene_A"])

    def test_all_observations_preserved(self, simple_mudata):
        result = simple_mudata.link.query_ancestors("prot_C")

        assert result.shape[0] == simple_mudata.shape[0]

    def test_n_to_m_shared_ancestors(self, n_to_m_mudata):
        result = n_to_m_mudata.link.query_ancestors("prot_D", include_self=False)

        assert_array_equal(sorted(result.var_names), ["gene_A", "gene_B"])
