import mudata as md
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mulink  # noqa: F401 — registers the namespace
from mulink.query import get_ancestors, get_descendants


@pytest.fixture
def simple_adjacency_matrix() -> np.ndarray:
    """Simple adjacency matrix

    Encodes the graph:

    A -> B
    A -> C
    B -> D
    B -> E
    C -> E
    """
    return np.array(
        [
            # A, B, C, D, E
            [0, 1, 1, 0, 0],  # A
            [0, 0, 0, 1, 1],  # B
            [0, 0, 0, 0, 1],  # C
            [0, 0, 0, 0, 0],  # D
            [0, 0, 0, 0, 0],  # E
        ]
    )


class TestGetDescendants:
    @pytest.mark.parametrize(
        ("query_indices", "expected_result_indices"),
        [
            (0, np.array([0, 1, 2, 3, 4])),  # A->B, A->C, B->D, B->E, C->E; search includes self
            (4, np.array([4])),  # Does not map to any other node. Search includes self
            ([0, 1], np.array([0, 1, 2, 3, 4])),  # A includes all, B does not contribute
            ([1], np.array([1, 3, 4])),  # B->D, C->E; search includes self
            ([3, 4], np.array([3, 4])),  # Do not map to anything, only includes self
        ],
    )
    def test_get_descendants(self, simple_adjacency_matrix, query_indices, expected_result_indices) -> None:
        """Test that the function correctly queries the adjacency matrix"""
        result = get_descendants(vertices=query_indices, adjacency_matrix=simple_adjacency_matrix)

        assert_array_equal(result, expected_result_indices)


class TestGetAncestors:
    @pytest.mark.parametrize(
        ("query_indices", "expected_result_indices"),
        [
            (0, np.array([0])),  # No ancestors, get self
            (4, np.array([0, 1, 2, 4])),  # E <- C; E <- B; E <- B <- A; E <- C <- A
            ([4], np.array([0, 1, 2, 4])),
            ([1, 2], np.array([0, 1, 2])),
        ],
    )
    def test_get_ancestors(self, simple_adjacency_matrix, query_indices, expected_result_indices) -> None:
        """Test that the function correctly queries the adjacency matrix"""
        result = get_ancestors(vertices=query_indices, adjacency_matrix=simple_adjacency_matrix)

        assert_array_equal(result, expected_result_indices)


class TestQueryDescendants:
    def test_single_feature_string(self, simple_mudata) -> None:
        result = simple_mudata.link.query.descendants("gene_A")

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_single_feature_list(self, simple_mudata) -> None:
        result = simple_mudata.link.query.descendants(["gene_A"])

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_multiple_features(self, simple_mudata) -> None:
        result = simple_mudata.link.query.descendants(["gene_A", "gene_B"])

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "gene_B", "prot_C", "prot_D"])

    def test_include_self_true(self, simple_mudata) -> None:
        result = simple_mudata.link.query.descendants("gene_A", include_self=True)

        assert "gene_A" in result.var_names

    def test_include_self_false(self, simple_mudata) -> None:
        result = simple_mudata.link.query.descendants("gene_A", include_self=False)

        assert "gene_A" not in result.var_names
        assert_array_equal(list(result.var_names), ["prot_C"])

    def test_n_to_m_mapping(self, n_to_m_mudata) -> None:
        result = n_to_m_mudata.link.query.descendants("gene_A", include_self=False)

        assert_array_equal(sorted(result.var_names), ["prot_C", "prot_D"])


class TestQueryAncestors:
    def test_single_feature(self, simple_mudata):
        result = simple_mudata.link.query.ancestors("prot_C")

        assert isinstance(result, md.MuData)
        assert result.n_obs == simple_mudata.n_obs

        assert_array_equal(sorted(result.var_names), ["gene_A", "prot_C"])

    def test_include_self_false(self, simple_mudata):
        result = simple_mudata.link.query.ancestors("prot_C", include_self=False)

        assert result.n_obs == simple_mudata.n_obs

        assert "prot_C" not in result.var_names
        assert_array_equal(list(result.var_names), ["gene_A"])

    def test_all_observations_preserved(self, simple_mudata):
        result = simple_mudata.link.query.ancestors("prot_C")

        assert result.shape[0] == simple_mudata.shape[0]

    def test_n_to_m_shared_ancestors(self, n_to_m_mudata):
        result = n_to_m_mudata.link.query.ancestors("prot_D", include_self=False)

        assert_array_equal(sorted(result.var_names), ["gene_A", "gene_B"])
