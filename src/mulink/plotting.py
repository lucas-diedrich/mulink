from collections.abc import Callable
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import mudata as md
import networkx as nx
import numpy as np

DEFAULT_COLOR = "#59b9c0"
DEFAULT_CMAP = mpl.colors.ListedColormap(["#ffffff", DEFAULT_COLOR])


def _modality_limits(mdata: md.MuData, axis: Literal[0, 1] = 0) -> np.ndarray:
    if axis == 0:
        return np.array([np.argmax(mdata.varmap[key]) for key in mdata.mod.keys()])
    elif axis == 1:
        return np.array([np.argmax(mdata.obsmap[key]) for key in mdata.mod.keys()])
    else:
        raise ValueError(f"Only `axis=0` or `axis=1` supported, got `axis={axis}`")


class PlotAccessor:
    """Plotting accessor for mulink"""

    def __init__(self, link):
        self._link = link
        self._mdata = self._link._obj

    def adjacency_matrix(
        self,
        key: str = "feature_mapping",
        axis: Literal[0, 1] = 0,
        cmap=DEFAULT_CMAP,
        ax: mpl.axes.Axes = None,
        *,
        label: bool = False,
        mod_limits: bool = True,
    ) -> mpl.axes.Axes:
        """Plot adjacency matrix of object"""
        if ax is None:
            _, ax = plt.subplots(1, 1)

        adjaceny_matrix = self._link.link(key=key, axis=axis)
        ax.imshow(adjaceny_matrix, cmap=cmap)

        if mod_limits:
            modality_limits = _modality_limits(self._mdata, axis=axis)
            for limit in modality_limits:
                ax.axhline(limit + 0.5, color="#000000")
                ax.axvline(limit + 0.5, color="#000000")

        if label:
            tickpositions = range(adjaceny_matrix.shape[0])
            ax.set_xticks(tickpositions)
            ax.set_yticks(tickpositions)
            ax.set_xticklabels(adjaceny_matrix.index, ha="center", rotation=90)
            ax.set_yticklabels(adjaceny_matrix.columns, ha="right", rotation=0)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_ylabel("Source")
        ax.set_xlabel("Target")

        return ax

    def graph(
        self,
        key: str = "feature_mapping",
        axis: Literal[0, 1] = 0,
        pos: Callable[[nx.DiGraph], Any] | None = None,
        ax: mpl.axes.Axes = None,
        **kwargs,
    ) -> mpl.axes.Axes:
        """Plot graph of mulink mapping

        Notes
        -----
        This method is expensive and only recommended for small subsets of the data
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        dag = dag = nx.from_pandas_adjacency(
            self._link.link(key=key, axis=axis),
            create_using=nx.DiGraph,
        )

        if pos is None:
            pos = nx.layout.multipartite_layout(
                dag,
                subset_key={key: self._link._get_link_indices(axis=axis, mod=key) for key in self._mdata.mod.keys()},
                align="vertical",
            )
        else:
            pos = pos(dag)

        return nx.draw(dag, pos=pos, ax=ax, **kwargs)
