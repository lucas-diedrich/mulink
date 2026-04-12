"""Query an adjacency matrix"""

from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix


def get_descendants(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all direct descendants for a feature or a list of features

    A direct descendants represents a vertix that can be reached by a single hop along the
    edge directionality.

    Parameters
    ----------
    vertices
        List of vertices for which the descendants should be queried.
    adjacency_matrix
        Adjacency matrix which indicates that u -> v (u maps to v)
        if (u, v) is nonzero.

    Returns
    -------
    List of direct successors of the provided vertices
    """
    # For an adjacency matrix following networkx convention, finding
    # all successors of a vertix corresponds to finding the columns containing nonzero `columns`
    # in the `row` corresponding to the vertix
    _, cols = adjacency_matrix[vertices, :].nonzero()

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(cols)


def get_ancestors(vertices: int | Iterable[int], adjacency_matrix: csr_matrix) -> np.ndarray:
    """Get all direct ancestors for a feature or a list of features

    A direct ancestors represents a vertix that can be reached by a single hop against
    edge directionality.

    Returns
    -------
    List of direct ancestors of the provided vertices
    """
    # For an adjacency matrix following networkx convention, finding
    # all ancestors of a vertix corresponds to finding the columns containing nonzero `columns`
    # in the `row` corresponding to the vertix
    rows, _ = adjacency_matrix[:, vertices].nonzero()

    # N:M mapping might yield redundant features - only return unique features
    return np.unique(rows)
