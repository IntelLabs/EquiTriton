# EquiTriton
[![CodeQL](https://github.com/ossf/scorecard-action/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/IntelLabs/EquiTriton/actions/workflows/codeql-analysis.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/EquiTriton/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/EquiTriton)

<div align="center">

[![pytorch](https://img.shields.io/badge/PyTorch-v2.1.0-red?logo=pytorch)](https://pytorch.org/get-started/locally/)
[![License: Apache2.0](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/apache-2-0)
![python-support](https://img.shields.io/badge/Python-3.10%7C3.11%7C3.12-3?logo=python)
![triton](https://img.shields.io/badge/Triton-2.10-2?link=https%3A%2F%2Fgithub.com%2Fintel%2Fintel-xpu-backend-for-triton%2Freleases%2Ftag%2Fv2.1.0)
[![paper](https://img.shields.io/badge/Paper-OpenReview-blue.svg)](https://openreview.net/forum?id=ftK00FO5wq)


</div>

_Performant kernels for equivariant neural networks in Triton-lang_

## Introduction

_EquiTriton_ is a project that seeks to implement high-performance kernels
for commonly used building blocks in equivariant neural networks, enabling
compute efficient training and inference. The advantage of Triton-lang is
portability across GPU architectures: kernels here have been tested against
GPUs from multiple vendors, including A100/H100 from Nvidia, and the Intel®️
Data Center GPU Max Series 1550.

Our current scope includes components such as spherical harmonics (including
derivatives, up to $l=4$), and we intend to expand this set quickly. If you
feel that a particular set of kernels would be valuable, please feel free
to submit an issue or pull request!


## Getting Started

For users, run `pip install git+https://github.com/IntelLabs/EquiTriton`. For those who
are using Intel XPUs, we recommend you reading the section on Intel XPU usage first,
and setting up an environment with PyTorch, IPEX, and Triton for XPU before installing
_EquiTriton_.

For developers/contributors, please clone this repository and install it in editable mode:

```console
git clone https://github.com/IntelLabs/EquiTriton
cd EquiTriton
pip install -e './[dev]'
```

...which will include development dependencies such as `pre-commit` (used for linting
and formatting), and `jupyter` used for symbolic differentiation for kernel development.

Finally, we provide `Dockerfile`s for users who prefer containers.

## Usage

As a drop-in replacement for `e3nn` spherical harmonics, simply include the
following in your code:

```python
from equitriton import patch
```

This will dynamically replace the `e3nn` spherical harmonics implementation
with the _EquiTriton_ kernels.

There are two important things to consider before replacing:

1. Numerically, there are small differences between implementations, primarily
in the backward pass. Because terms in the gradients are implemented as literals,
they can be more susceptible to rounding errors at lower precision. In most
(not all!) instances, they are numerically equivalent for `torch.float32`, and
basically _always_ different for `torch.float16`. At double precision (`torch.float64`)
this does not seem to be an issue, which makes it ideal for use in simulation loops but
please be aware that if it is used for training, the optimization trajectory may not
be exactly the same; we have not tested for divergence and encourage experimentation.
2. Triton kernels are compiled just-in-time and a cached every time it encounters
a new input tensor shape. In `equitriton.sph_harm.SphericalHarmonics`, the `pad_tensor`
argument (default is `True`) is used to try and maximize cache re-use by padding
nodes and masking in the forward pass. The script `scripts/dynamic_shapes.py` will
let you test the performance over a range of shapes; we encourage you to test it
before performing full-scale training/inference.

### Development and usage on Intel XPU

Development on Intel XPUs such as the Data Center GPU Max Series 1550 requires
a number of manual components for bare metal. The core dependency to consider
is the [Intel XPU backend for Triton][triton-git], which will dictate the version
of oneAPI, PyTorch, and Intel Extension for PyTorch to install. At the time
of release, _EquiTriton_ has been developed on the following:

- oneAPI 2024.0
- PyTorch 2.1.0
- IPEX 2.1.10+xpu
- Intel XPU backend for Triton [2.1.0](https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v2.1.0)

Due to the way that wheels are distributed, please install PyTorch
and IPEX per `intel-requirements.txt`. Alternatively, use the provided
Docker image for development.

```python
>>> import intel_extension_for_pytorch
>>> import torch
>>> torch.xpu.device_count()
# should be greater than zero
```
[triton-git]: https://github.com/intel/intel-xpu-backend-for-triton/releases/tag/v2.1.0

## Useful commands for Intel GPUs

- `xpu-smi` (might not be installed) as the name suggests is the equivalent to `nvidia-smi`,
but with a bit more functionality based on our architecture
- `sycl-ls` is provided by the `dpcpp` runtime, and lists out all devices that are OpenCL
and SYCL capable. Notably this can be used to quickly check how many GPUs are available.
- [pti-gpu](https://github.com/intel/pti-gpu) provides a set of tools that you can compile for profiling. Notably,
`unitrace` and `oneprof` allows you do to low-level profiling for the device.


Contributing
------------

We welcome contributions from the open-source community! If you have any
questions or suggestions, feel free to create an issue in our
repository. We will be happy to work with you to make this project even
better.

License
-------

The code and documentation in this repository are licensed under the Apache 2.0
license. By contributing to this project, you agree that your
contributions will be licensed under this license.

Citation
--------
If you find this repo useful, please consider citing the corresponding paper:

```bibtex
@inproceedings{lee2024scaling,
    title={Scaling Computational Performance of Spherical Harmonics Kernels with Triton},
    author={Kin Long Kelvin Lee and Mikhail Galkin and Santiago Miret},
    booktitle={AI for Accelerated Materials Design - Vienna 2024},
    year={2024},
    url={https://openreview.net/forum?id=ftK00FO5wq}
}
```