from contextlib import nullcontext

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from mulink.schema import GraphConstraints
from mulink.validate import (
    ConnectivityError,
    CycleError,
    HierarchyError,
    SelfReferenceError,
    check_acyclic,
    feature_modalities,
    validate_link,
)

# Example from lucas-diedrich/mulink#22: peptides -> proteins -> genes
FEATURES = {"peptides": ["p1", "p2", "p3"], "proteins": ["Pa", "Pb"], "genes": ["G1", "G2"]}
VALID_EDGES = [("p1", "Pa"), ("p1", "Pb"), ("p2", "Pa"), ("p3", "Pb"), ("Pa", "G1"), ("Pb", "G2")]
HIERARCHY = {"peptides": ["proteins"], "proteins": ["genes"]}


@pytest.fixture
def mdata() -> md.MuData:
    mods = {
        mod: ad.AnnData(np.zeros((1, len(var_names))), var=pd.DataFrame(index=var_names))
        for mod, var_names in FEATURES.items()
    }
    return md.MuData(mods)


def adjacency_from_edges(mdata: md.MuData, edges: list[tuple[str, str, float]] | list[tuple[str, str]]) -> csr_array:
    index = {name: i for i, name in enumerate(mdata.var_names)}
    adjacency = np.zeros((mdata.n_var, mdata.n_var))
    for source, target, *weight in edges:
        adjacency[index[source], index[target]] = weight[0] if weight else 1
    return csr_array(adjacency)


def test_feature_modalities(mdata) -> None:
    result = feature_modalities(mdata.varmap)

    expected = [mod for mod, var_names in FEATURES.items() for _ in var_names]
    assert list(result) == expected


@pytest.mark.parametrize(
    "constraints",
    [
        None,
        GraphConstraints(),
        GraphConstraints(
            max_outgoing_edges=2, max_incoming_edges=2, allow_isolated_features=False, allowed_targets=HIERARCHY
        ),
    ],
)
def test_validate_link__valid_graph_passes(mdata, constraints) -> None:
    adjacency = adjacency_from_edges(mdata, VALID_EDGES)

    validate_link(adjacency, mdata.varmap, constraints)


@pytest.mark.parametrize(
    ("allow_isolated_features", "expectation"),
    [(True, nullcontext()), (False, pytest.raises(ConnectivityError, match="1 features have no"))],
)
def test_validate_link__isolated_features(mdata, allow_isolated_features, expectation) -> None:
    # Dropping Pb -> G2 leaves G2 without any edge
    edges = [edge for edge in VALID_EDGES if edge != ("Pb", "G2")]
    adjacency = adjacency_from_edges(mdata, edges)
    constraints = GraphConstraints(allow_isolated_features=allow_isolated_features)

    with expectation:
        validate_link(adjacency, mdata.varmap, constraints)


def test_validate_link__empty_graph_passes(mdata) -> None:
    adjacency = csr_array((mdata.n_var, mdata.n_var))
    constraints = GraphConstraints(max_outgoing_edges=0, max_incoming_edges=0, allowed_targets={"peptides": []})

    validate_link(adjacency, mdata.varmap, constraints)


def test_validate_link__explicit_zeros_are_not_edges(mdata) -> None:
    # p1 -> p2 would be a self-reference if the stored zero counted as an edge
    p1, p2 = 0, 1
    adjacency = csr_array((np.array([0.0]), (np.array([p1]), np.array([p2]))), shape=(mdata.n_var, mdata.n_var))
    assert adjacency.nnz == 1

    validate_link(adjacency, mdata.varmap)


@pytest.mark.parametrize(
    ("extra_edges", "constraints", "error", "match"),
    [
        pytest.param([("p1", "p2")], None, SelfReferenceError, "peptides", id="self_reference"),
        pytest.param([("p1", "p2", -1)], None, SelfReferenceError, "peptides", id="self_reference_negative_weight"),
        pytest.param([("p1", "p1")], None, SelfReferenceError, "peptides", id="diagonal_self_loop"),
        pytest.param([("G1", "p2")], None, CycleError, "cycle", id="cycle_across_modalities"),
        pytest.param([], GraphConstraints(max_outgoing_edges=1), ConnectivityError, "outgoing", id="max_outgoing"),
        pytest.param([], GraphConstraints(max_incoming_edges=1), ConnectivityError, "incoming", id="max_incoming"),
        pytest.param(
            [("p2", "Pb", -1)],
            GraphConstraints(max_outgoing_edges=1),
            ConnectivityError,
            "outgoing",
            id="max_outgoing_cancelling_weights",
        ),
        pytest.param(
            [("p1", "G1")],
            GraphConstraints(allowed_targets=HIERARCHY),
            HierarchyError,
            "peptides -> genes",
            id="hierarchy",
        ),
    ],
)
def test_validate_link__invalid_graph_raises(mdata, extra_edges, constraints, error, match) -> None:
    adjacency = adjacency_from_edges(mdata, VALID_EDGES + extra_edges)

    with pytest.raises(error, match=match):
        validate_link(adjacency, mdata.varmap, constraints)


def test_check_acyclic__diagonal_self_loop_raises(mdata) -> None:
    adjacency = adjacency_from_edges(mdata, [("p1", "p1")])

    with pytest.raises(CycleError):
        check_acyclic(adjacency)


@pytest.mark.parametrize(
    "allowed_targets",
    [{"metabolites": ["genes"]}, {"peptides": ["metabolites"]}],
    ids=["unknown_source", "unknown_target"],
)
def test_validate_link__unknown_modality_in_hierarchy_raises(mdata, allowed_targets) -> None:
    adjacency = adjacency_from_edges(mdata, VALID_EDGES)

    with pytest.raises(ValueError, match="unknown modalities.*metabolites"):
        validate_link(adjacency, mdata.varmap, GraphConstraints(allowed_targets=allowed_targets))


class TestMuLinkValidation:
    def test_add_link__invalid_raises_without_modifying(self, mdata) -> None:
        adjacency = adjacency_from_edges(mdata, [("p1", "p2")])

        with pytest.raises(SelfReferenceError):
            mdata.link.add_link(adjacency, key="invalid")

        assert "invalid" not in mdata.varp
        assert "mulink" not in mdata.uns

    def test_add_link__validate_false_skips_validation(self, mdata) -> None:
        adjacency = adjacency_from_edges(mdata, [("p1", "p2")])

        mdata.link.add_link(adjacency, key="invalid", validate=False)

        assert "invalid" in mdata.varp

    def test_add_link__checks_constraints(self, mdata) -> None:
        adjacency = adjacency_from_edges(mdata, VALID_EDGES)

        with pytest.raises(ConnectivityError):
            mdata.link.add_link(adjacency, constraints={"max_outgoing_edges": 1})

    @pytest.mark.parametrize("keys", [None, "feature_mapping", ["feature_mapping"]])
    def test_validate__uses_stored_constraints(self, mdata, keys) -> None:
        mdata.link.add_link(adjacency_from_edges(mdata, VALID_EDGES), constraints={"allowed_targets": HIERARCHY})
        mdata.varp["feature_mapping"] = adjacency_from_edges(mdata, [*VALID_EDGES, ("p1", "G1")])

        with pytest.raises(HierarchyError):
            mdata.link.validate(keys=keys)

    def test_constraints_survive_io(self, mdata, tmp_path) -> None:
        constraints = {"max_outgoing_edges": 2, "allow_isolated_features": False, "allowed_targets": HIERARCHY}
        mdata.link.add_link(adjacency_from_edges(mdata, VALID_EDGES), constraints=constraints)
        path = tmp_path / "test.h5mu"

        mdata.write_h5mu(path)
        result = md.read_h5mu(path)

        assert result.link.meta().constraints == GraphConstraints(**constraints)
        result.link.validate()
