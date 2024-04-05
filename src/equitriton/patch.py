# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from logging import getLogger
import math

import torch
from e3nn.o3 import SphericalHarmonics
from equitriton.sph_harm.main import SphericalHarmonics as TritonHarmonics

"""
This module will monkey patch ``e3nn``: in other words, when loaded,
it will dynamically replace the ``forward`` call for the ``e3nn``
Spherical Harmonic class that is commonly used in equivariant models with
the _Equitriton_ version (i.e. replace ``torchscript`` kernels with the
``triton`` ones.

If this behavior is _not_ desired, do _not_ import this module at runtime.
"""

logger = getLogger("equitriton")


def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Patched version of the forward call, which instead relies on Triton
    kernels for each value of l_max.
    """
    if self.normalize:
        x = torch.nn.functional.normalize(
            x, dim=-1
        )  # forward 0's instead of nan for zero-radius

    # initialize the spherical harmonics wrapper
    if not hasattr(self, "_triton"):
        self._triton = TritonHarmonics(self._lmax)
    # do the spherical harmonic evaluation with triton kernels instead
    sh = self._triton(x)

    if not self._is_range_lmax:
        sh = torch.cat(
            [sh[..., l * l : (l + 1) * (l + 1)] for l in self._ls_list],  # noqa: E741
            dim=-1,
        )

    if self.normalization == "integral":
        sh.div_(math.sqrt(4 * math.pi))
    elif self.normalization == "norm":
        sh.div_(
            torch.cat(
                [
                    math.sqrt(2 * l + 1)
                    * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
                    for l in self._ls_list  # noqa: E741
                ]
            )
        )

    return sh


# apply the monkey patch
logger.info("Patching e3nn `SphericalHarmonics.forward` with Triton kernels.")
SphericalHarmonics.forward = forward
