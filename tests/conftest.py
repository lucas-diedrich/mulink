import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
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
