"""Test simulation framework"""

import anndata as ad
import mudata as md
import networkx as nx
import pytest

from mulink.simulate import _generate_anndata, _generate_dag, hierarchical_mudata

md.set_options(pull_on_update=False)


@pytest.mark.parametrize(("n_obs", "n_var"), [(0, 0), (1, 2)])
def test__generate_anndata(n_obs: int, n_var: int) -> None:
    adata = _generate_anndata(n_obs=n_obs, n_var=n_var)

    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (n_obs, n_var)


@pytest.mark.parametrize("transitive_closure", [True, False])
@pytest.mark.parametrize("extra_edge_probability", [None, 0, 0.1, 0.9])
@pytest.mark.parametrize("n_vertices", [1, 2])
@pytest.mark.parametrize("n_level", [1, 2, 3])
def test__generate_dag(
    n_vertices: int, n_level: int, extra_edge_probability: float | None, transitive_closure: bool
) -> None:
    dag, n_nodes_per_level = _generate_dag(
        n_level=n_level,
        n_vertices=n_vertices,
        extra_edge_probability=extra_edge_probability,
        transitive_closure=transitive_closure,
    )

    expected_nodes_per_level = {level: n_vertices ** (level + 1) for level in range(n_level)}

    assert isinstance(dag, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(dag)
    assert n_nodes_per_level == expected_nodes_per_level


@pytest.mark.parametrize("linkage_key", ["feature_mapping", "test"])
@pytest.mark.parametrize("n_obs", [5])
@pytest.mark.parametrize("n_vertices", [1, 3])
@pytest.mark.parametrize("n_mod", [1, 3])
def test_hierarchical_mudata(n_mod: int, n_vertices: int, n_obs: int, linkage_key: str) -> None:
    mdata = hierarchical_mudata(n_mod=n_mod, n_vertices=n_vertices, linkage_key=linkage_key)

    expected_modality_names = {f"mod{idx}" for idx in range(n_mod)}
    expected_n_vars = sum(n_vertices ** (mod + 1) for mod in range(n_mod))

    assert isinstance(mdata, md.MuData)
    assert set(mdata.mod.keys()) == expected_modality_names
    assert mdata.shape == (n_obs, expected_n_vars)
    assert nx.is_directed_acyclic_graph(nx.from_scipy_sparse_array(mdata.varp[linkage_key], create_using=nx.DiGraph))
