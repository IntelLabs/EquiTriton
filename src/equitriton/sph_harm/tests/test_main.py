# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import pytest
import torch

from equitriton.sph_harm import SphericalHarmonics
from equitriton import __HAS_XPU__, __HAS_CUDA__


@pytest.mark.parametrize(
    "l_max",
    [
        pytest.param(
            0, marks=pytest.mark.xfail(reason="Zeroth order not implemented.")
        ),
        1,
        2,
        3,
        4,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape",
    [
        (64, 3),
        (256, 64, 3),
        (512, 128, 8, 3),
        pytest.param(
            (10, 8, 40, 1), marks=pytest.mark.xfail(reason="Bad last dimension.")
        ),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not __HAS_CUDA__, reason="No CUDA device available."
            ),
        ),
        pytest.param(
            "xpu:0",
            marks=pytest.mark.skipif(
                not __HAS_XPU__, reason="No XPU device available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("node_padding", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_main_interface(l_max, tensor_shape, device, node_padding, dtype):
    joint_tensor = torch.rand(tensor_shape, device=device, dtype=dtype)
    sph_harm = SphericalHarmonics(l_max, pad_tensors=node_padding)
    output = sph_harm(joint_tensor)
    assert torch.isfinite(output).all()
