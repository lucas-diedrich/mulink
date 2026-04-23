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
