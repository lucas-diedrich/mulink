import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pytest
from numpy.testing import assert_array_equal

import mulink  # noqa: F401 — registers the namespace


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestAdjacencyMatrix:
    @pytest.mark.parametrize("mod_limits", [True, False])
    @pytest.mark.parametrize("label", [True, False])
    def test_returns_axes(self, simple_mudata_any_axis, mod_limits: bool, label: bool) -> None:
        ax = simple_mudata_any_axis.link.pl.adjacency_matrix(mod_limits=mod_limits, label=label)

        assert isinstance(ax, mpl.axes.Axes)

    @pytest.mark.parametrize("mod_limits", [True, False])
    @pytest.mark.parametrize("label", [True, False])
    def test_uses_passed_axes(self, simple_mudata_any_axis, mod_limits: bool, label: bool) -> None:
        _, ax = plt.subplots()

        result = simple_mudata_any_axis.link.pl.adjacency_matrix(ax=ax, mod_limits=mod_limits, label=label)

        assert result is ax

    def test_labels_true_sets_tick_labels(self, simple_mudata_any_axis) -> None:
        ax = simple_mudata_any_axis.link.pl.adjacency_matrix(label=True)

        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]

        names = simple_mudata_any_axis.link._axis.get_names(simple_mudata_any_axis)
        assert_array_equal(xticklabels, list(names))
        assert_array_equal(yticklabels, list(names))

    def test_labels_false_has_no_tick_labels(self, simple_mudata_any_axis) -> None:
        ax = simple_mudata_any_axis.link.pl.adjacency_matrix(label=False)

        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []


class TestGraph:
    def test_pl_graph(self, simple_mudata_any_axis) -> None:
        simple_mudata_any_axis.link.pl.graph()

    def test_pl_graph__custom_axis(self, simple_mudata_any_axis) -> None:
        _, ax = plt.subplots()

        simple_mudata_any_axis.link.pl.graph(ax=ax)

    def test_pl_graph__custom_layout(self, simple_mudata_any_axis) -> None:
        _, ax = plt.subplots()

        simple_mudata_any_axis.link.pl.graph(ax=ax, pos=nx.layout.circular_layout)
