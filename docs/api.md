# API

## Basic

```{eval-rst}
.. module:: mulink.basic
.. currentmodule:: mulink

.. autosummary::
    :toctree: generated

    basic.MuLink
```

## Querying

```{eval-rst}
.. module:: mulink.query
.. currentmodule:: mulink

.. autosummary::
    :toctree: generated

    query.QueryAccessor
```

## Validation

```{eval-rst}
.. module:: mulink.validate
.. currentmodule:: mulink

.. autosummary::
    :toctree: generated

    validate.validate_link
    validate.acyclic
    validate.check_no_self_references
    validate.check_max_outgoing_edges
    validate.check_max_incoming_edges
    validate.check_no_isolated_features
    validate.check_allowed_targets
    validate.LinkValidationError
    validate.CycleError
    validate.SelfReferenceError
    validate.ConnectivityError
    validate.HierarchyError
```

## Plotting

```{eval-rst}
.. module:: mulink.plotting
.. currentmodule:: mulink

.. autosummary::
    :toctree: generated

    plotting.PlotAccessor
```

## Simulate

```{eval-rst}
.. module:: mulink.simulate
.. currentmodule:: mulink

.. autosummary::
    :toctree: generated

    simulate.hierarchical_mudata
```
