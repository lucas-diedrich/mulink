"""Validation of graph properties of links

Convention: a non-zero entry at row `u` and column `v` of the adjacency matrix is an edge `u -> v`.
"""

from collections.abc import Mapping

import networkx as nx
import numpy as np
from scipy.sparse import csr_array

from mulink.schema import GraphConstraints


class LinkValidationError(ValueError):
    """Link graph violates a structural rule or a constraint"""


class CycleError(LinkValidationError):
    """Link graph contains cycles"""


class SelfReferenceError(LinkValidationError):
    """Link graph contains edges between features of the same modality"""


class ConnectivityError(LinkValidationError):
    """Features in the link graph have too many or too few edges"""


class HierarchyError(LinkValidationError):
    """Link graph contains edges between modalities that are not allowed to be linked"""


def feature_modalities(varmap: Mapping[str, np.ndarray]) -> np.ndarray:
    """Name of the modality of each global feature

    Parameters
    ----------
    varmap
        Mapping from modality name to its feature map, as in `mdata.varmap` (or `mdata.obsmap` for `mdata.axis == 1`).

    Returns
    -------
    Array of modality names with one entry per global feature.
    """
    n_var = len(next(iter(varmap.values()))) if varmap else 0
    modalities = np.empty(n_var, dtype=object)
    for mod, mapping in varmap.items():
        # mudata encodes membership as 1-based positions in the modality, 0 marks absence
        modalities[np.asarray(mapping) > 0] = mod
    return modalities


def _edges(adjacency: csr_array) -> tuple[np.ndarray, np.ndarray]:
    """Binarized representation of adjacency matrix"""
    return adjacency.nonzero()


def check_acyclic(adjacency: csr_array) -> None:
    """Check that the link graph is a directed acyclic graph

    Raises
    ------
    CycleError
        If the graph contains a cycle, including self-loops.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    graph.add_edges_from(zip(*_edges(adjacency), strict=True))

    try:
        cycle = nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return
    raise CycleError(f"Link graph contains a cycle along edges (by feature index) {cycle}")


def check_no_self_references(adjacency: csr_array, modalities: np.ndarray) -> None:
    """Check that no feature links to a feature of its own modality

    Parameters
    ----------
    adjacency
        Adjacency matrix of the link graph.
    modalities
        Modality name of each feature, see :func:`feature_modalities`.

    Raises
    ------
    SelfReferenceError
        If any edge connects two features of the same modality.
    """
    sources, targets = _edges(adjacency)
    is_self_reference = modalities[sources] == modalities[targets]

    if is_self_reference.any():
        failing_mods = sorted(set(modalities[sources[is_self_reference]]))
        raise SelfReferenceError(f"Link graph contains edges within modalities {failing_mods}")


def _check_max_degree(adjacency: csr_array, max_edges: int, *, axis: int, direction: str) -> None:
    degree = np.asarray((adjacency != 0).sum(axis=axis)).ravel()
    n_failing = int((degree > max_edges).sum())

    if n_failing > 0:
        raise ConnectivityError(
            f"{n_failing} features have more than {max_edges} {direction} edges (maximum: {degree.max()})"
        )


def check_max_outgoing_edges(adjacency: csr_array, max_edges: int) -> None:
    """Check that no feature has more than `max_edges` outgoing edges

    Raises
    ------
    ConnectivityError
        If any feature exceeds the maximum.
    """
    _check_max_degree(adjacency, max_edges, axis=1, direction="outgoing")


def check_max_incoming_edges(adjacency: csr_array, max_edges: int) -> None:
    """Check that no feature has more than `max_edges` incoming edges

    Raises
    ------
    ConnectivityError
        If any feature exceeds the maximum.
    """
    _check_max_degree(adjacency, max_edges, axis=0, direction="incoming")


def check_no_isolated_features(adjacency: csr_array) -> None:
    """Check that every feature has at least one incoming or outgoing edge

    Raises
    ------
    ConnectivityError
        If any feature has no edges.
    """
    is_edge = adjacency != 0
    degree = np.asarray(is_edge.sum(axis=0)).ravel() + np.asarray(is_edge.sum(axis=1)).ravel()
    n_isolated = int((degree == 0).sum())

    if n_isolated > 0:
        raise ConnectivityError(f"{n_isolated} features have no incoming or outgoing edges")


def check_allowed_targets(
    adjacency: csr_array, modalities: np.ndarray, allowed_targets: Mapping[str, list[str]]
) -> None:
    """Check that features only link to features of allowed modalities

    Parameters
    ----------
    adjacency
        Adjacency matrix of the link graph.
    modalities
        Modality name of each feature, see :func:`feature_modalities`.
    allowed_targets
        Mapping from a source modality to the modalities it may link to.
        Source modalities that are not listed are unrestricted.

    Raises
    ------
    ValueError
        If `allowed_targets` refers to a modality that does not exist.
    HierarchyError
        If any edge links a source modality to a modality that is not allowed.
    """
    known_mods = set(modalities)
    referenced_mods = set(allowed_targets) | {mod for targets in allowed_targets.values() for mod in targets}
    if unknown_mods := referenced_mods - known_mods:
        raise ValueError(f"`allowed_targets` refers to unknown modalities {sorted(unknown_mods)}")

    sources, targets = _edges(adjacency)
    violations = {
        (source_mod, target_mod)
        for source_mod, target_mod in zip(modalities[sources], modalities[targets], strict=True)
        if source_mod in allowed_targets and target_mod not in allowed_targets[source_mod]
    }

    if violations:
        formatted = ", ".join(f"{source} -> {target}" for source, target in sorted(violations))
        raise HierarchyError(f"Link graph contains edges between modalities that are not allowed: {formatted}")


def validate_link(
    adjacency: csr_array, varmap: Mapping[str, np.ndarray], constraints: GraphConstraints | None = None
) -> None:
    """Validate a link graph against the structural rules of the format and optional constraints

    Parameters
    ----------
    adjacency
        Adjacency matrix of the link graph.
    varmap
        Mapping from modality name to its feature map, as in `mdata.varmap` (or `mdata.obsmap` for `mdata.axis == 1`).
    constraints
        Optional constraints. Acyclicity and absence of self-references are always enforced.

    Raises
    ------
    LinkValidationError
        If the link graph violates a rule or a constraint. The subclass indicates the violated property.
    ValueError
        If the constraints refer to unknown modalities.
    """
    adjacency = csr_array(adjacency)
    modalities = feature_modalities(varmap)

    check_no_self_references(adjacency, modalities)
    check_acyclic(adjacency)

    if constraints is None:
        return
    if constraints.max_outgoing_edges is not None:
        check_max_outgoing_edges(adjacency, constraints.max_outgoing_edges)
    if constraints.max_incoming_edges is not None:
        check_max_incoming_edges(adjacency, constraints.max_incoming_edges)
    if not constraints.allow_isolated_features:
        check_no_isolated_features(adjacency)
    if constraints.allowed_targets is not None:
        check_allowed_targets(adjacency, modalities, constraints.allowed_targets)
