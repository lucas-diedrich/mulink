import mudata as md
from numpy.testing import assert_array_equal

import mulink  # noqa: F401 — registers the namespace


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
