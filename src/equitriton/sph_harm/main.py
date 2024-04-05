# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from equitriton.sph_harm.bindings import *
from equitriton.utils import pad_tensor_to_power


class SphericalHarmonics(nn.Module):
    # None is prepended to keep the indexing consistent with l_max
    __fwd_kernel_mapping__ = [
        None,
        FirstOrderSphericalHarmonics,
        SecondOrderSphericalHarmonics,
        ThirdOrderSphericalHarmonics,
        FourthOrderSphericalHarmonics,
    ]

    def __init__(self, lmax: int, pad_tensors: bool = True) -> None:
        """
        Initialize a ``SphericalHarmonics`` object that computes
        up to some maximum value of ``l``.

        Optionally, to minimize kernel JIT overhead, the option to
        pad tensors under the hood is provided: by rounding the
        number of nodes to the nearest power of two, we are able
        to improve re-use of kernels compiled for specific shapes.

        Parameters
        ----------
        lmax : int
            Maximum value of ``l`` to use for embedding.
        pad_tensors : bool, default True
            If set to True, this will pad the number of nodes
            up to the nearest power of two. This results in
            higher memory usage during the forward pass, but
            the tradeoff is minimizing overhead from needing
            to recompile kernels for every single batch shape.

            In cases where shapes are expected to be static
            (e.g. in MD simulations), this can be safely disabled.
        """
        super().__init__()
        self.lmax = lmax
        self.pad_tensors = pad_tensors

    def _preprocess_tensors(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # last dimension should be xyz
        assert (
            input_tensor.size(-1) == 3
        ), f"Expected last input dimension to be 3 (x,y,z). Got {input_tensor.size(-1)}"
        # pad tensor if requested
        if self.pad_tensors:
            input_tensor, mask = pad_tensor_to_power(input_tensor)
            self.mask = mask
        else:
            self.mask = None
        # make tensors contiguous for better memory access
        x, y, z = (
            input_tensor[..., 0].contiguous(),
            input_tensor[..., 1].contiguous(),
            input_tensor[..., 2].contiguous(),
        )
        return (x, y, z)

    def _determine_kernel(self) -> Callable:
        try:
            kernel = self.__fwd_kernel_mapping__[self.lmax]
        except IndexError as e:
            raise NotImplementedError(
                f"Kernels only implemented up to lmax = {len(self.__fwd_kernel_mapping__)}."
            ) from e
        if kernel is None:
            raise NotImplementedError(
                "Zeroth order kernel is not implemented; it's too trivial 😏"
            )
        return kernel

    def _forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        kernel = self._determine_kernel()
        return kernel.apply(x, y, z, self.mask)

    def forward(self, input_tensor: torch.Tensor):
        x, y, z = self._preprocess_tensors(input_tensor)
        return self._forward(x, y, z)
