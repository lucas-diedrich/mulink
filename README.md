# mulink

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/mulink/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mulink

Links between modalities in mudata

## Getting started


`mulink` models feature relationships between `mudata` modalities as directed acyclic graph. To work with this mapping, it extends the namespace of `mudata` with a custom `link` attribute. The individual functionalities are implemented as accessors to this namespace.

```python
import mudata as md
import mulink  # registers the `link` namespace on mudata
from scipy.sparse import csr_matrix

# minimal example
mdata = mulink.simulate.hierarchical_mudata(n_mod=3)
assert isinstance(mdata, md.MuData)
assert "feature_mapping" in mdata.varp.keys()
assert hasattr(mdata, "link")

# add linkage matrix
mdata.link.add_link(csr_matrix(...))

# querying
mdata.link.query.descendants("mod0-0")
mdata.link.query.ancestors("mod2-0")

# plotting
mdata.link.pl.graph()
```

For more information, please refer to the [documentation][],
in particular, the [API documentation][] and the [Design Document][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install mulink:

<!--
1) Install the latest release of `mulink` from [PyPI][]:

```bash
pip install mulink
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lucas-diedrich/mulink.git@main
```

## Release notes

See the [changelog][].

## Contact

<!-- For questions and help requests, you can reach out in the [scverse discourse][]. -->
If you found a bug, please use the [issue tracker][].

## Citation

This project was started at the [scverse proteomics hackathon in Berlin 2026](https://github.com/scverse/202603_hackathon_proteomics). Comparable features are implemented in the [QFeatures](https://rformassspectrometry.github.io/QFeatures/) package in R.

> scverse proteomics working group 2026. mulink.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/mulink/issues
[tests]: https://github.com/lucas-diedrich/mulink/actions/workflows/test.yaml
[documentation]: https://mulink.readthedocs.io
[changelog]: https://mulink.readthedocs.io/en/latest/changelog.html
[api documentation]: https://mulink.readthedocs.io/en/latest/api.html
[Design Document]: https://mulink.readthedocs.io/en/latest/rfc.html

[pypi]: https://pypi.org/project/mulink
