from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Literal

import mudata as md
import networkx as nx
import numpy as np
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from mudata._core.mudata import ModalityMapAxisArrays

    from mulink.basic import MuLink


class CycleError(ValueError):
    """Graph contains cycles"""


class ConnectivityError(ValueError):
    """Vertices in the graph violate connectivity rules"""


def validation_decorator(graph_property: str) -> Callable[..., None]:
    """Decorator that marks a function as validator for a graph and names the validation type"""

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., None]:
            return func(*args, **kwargs)

        return wrapper

    decorator.__name__ = graph_property
    return decorator


class GraphValidator:
    """Validate graph properties for mulink"""

    def __init__(self, link) -> None:
        self._link: MuLink = link
        self._mdata: md.MuData = self._link._obj

    def validate(self, keys: str | Iterable[str] | None = None):
        """Validate mulink object based on metadata

        Parameters
        ----------
        keys
            Which mulink keys to validate. Defaults to all registered keys.
        """
        if keys is None:
            keys = self._link.link_keys
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            self._validate(key=key)

    def _validate(self, key: str) -> None:
        """Validate a single key"""
        link = self._link.link(key=key)
        meta = self._link.meta(key=key)
        modality_mapping = self._mdata.varmap

        graph = nx.from_pandas_adjacency(link, create_using=nx.DiGraph)

        if constraints := meta.graph_constraints:
            for constraint in constraints:
                func = self.__getattribute__(constraint)
                print(func)

    @validation_decorator(graph_property="acyclic")
    def _validate_acyclicity(self, graph: nx.DiGraph):
        """Acyclicity of graph"""
        is_acyclic = nx.is_directed_acyclic_graph(graph)

        if not is_acyclic:
            raise CycleError("Graph contains cycles")

    @validation_decorator(graph_property="connectivity")
    def _validate_connectivity(self, graph):
        """Validate that graph is tranistively closed"""
        is_connected = nx.is_connected(graph)

        if not is_connected:
            raise CycleError("Graph contains cycles")

    def _connectivity(self, adjacency_matrix, axis: Literal[0, 1] = 0):
        non_zero = adjacency_matrix != 0

        return non_zero.sum(axis=axis)

    @validation_decorator(graph_property="max_outgoing_edges")
    def _validate_maximum_outgoing_edges(self, adjacency_matrix: csr_array, max_connectivity: int) -> None:

        # Convention: row maps to column
        if self._connectivity(adjacency_matrix, axis=0).max() > max_connectivity:
            raise ConnectivityError("Too many outgoing connections")

    @validation_decorator(graph_property="max_incoming_edges")
    def _validate_maximum_incoming_edges(self, adjacency_matrix: csr_array, max_connectivity: int) -> None:

        # Convention: row maps to column
        if self._connectivity(adjacency_matrix, axis=1).max() > max_connectivity:
            raise ConnectivityError("Too many incoming connections")

    @validation_decorator(graph_property="min_outgoing_edges")
    def _validate_minimum_outgoing_edges(self, adjacency_matrix: csr_array, min_connectivity: int) -> None:

        # Convention: row maps to column
        if self._connectivity(adjacency_matrix, axis=0).min() < min_connectivity:
            raise ConnectivityError("Too few outgoing connections")

    @validation_decorator(graph_property="maximimum_outgoing_edges")
    def _validate_minimum_incoming_edges(self, adjacency_matrix: csr_array, min_connectivity: int) -> None:

        # Convention: row maps to column
        if self._connectivity(adjacency_matrix, axis=1).min() < min_connectivity:
            raise ConnectivityError("Too few incoming connections")

    @validation_decorator(graph_property="self_references")
    def _validate_self_references(self, modality_mapping: "ModalityMapAxisArrays", adj: csr_array) -> None:

        modality_index_locations = {mod: np.nonzero(array)[0] for mod, array in modality_mapping.items()}

        failing_mods: set[str] = set()
        for mod, mod_indices in modality_index_locations.items():
            if (adj[np.ix_(mod_indices, mod_indices)].sum(axis=0) > 0).any():
                failing_mods.add(mod)

        if failing_mods:
            raise ConnectivityError(f"Self references in modalities {failing_mods}")
