# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations
import pytest

import torch

from equitriton import __HAS_XPU__, __HAS_CUDA__
from equitriton.sph_harm import bindings
from e3nn.o3._spherical_harmonics import _spherical_harmonics

# make sure values are the same every time
torch.manual_seed(3125161)

"""
This test suite parametrizes over l, device, and tensor shapes to
test for functionality and correctness.

TODO: expand to parametrize data types as well
"""

RTOL = 1e-4
ATOL = 1e-6


@pytest.mark.parametrize("l_func_name", bindings.__all__)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(not __HAS_XPU__, reason="No XPUs available."),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not __HAS_CUDA__, reason="No CUDA GPUs available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
def test_bound_kernel(l_func_name, device, tensor_shape):
    """
    Iterate through exported autograd bindings, and make sure that
    the forward application passes.
    """
    l_func = getattr(bindings, l_func_name)
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    x, y, z = joint_tensor[..., 0], joint_tensor[..., 1], joint_tensor[..., 2]
    outputs = l_func.apply(x, y, z)
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("l", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(not __HAS_XPU__, reason="No XPUs available."),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not __HAS_CUDA__, reason="No CUDA GPUs available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_correctness_fwd_bwd(l, device, tensor_shape, dtype):
    """Compare e3nn and triton results for the forward and backward passes."""
    joint = torch.rand(tensor_shape, device=device, dtype=dtype, requires_grad=True)
    x, y, z = joint[..., 0], joint[..., 1], joint[..., 2]
    # run the test with e3nn forward then backward
    e3nn_result = _spherical_harmonics(l, x, y, z)
    e3nn_result.backward(gradient=torch.ones_like(e3nn_result))
    e3nn_grad = joint.grad.clone().detach()
    # reset grads for the next round
    joint.grad = None
    # index the exported bindings, which starts from 1
    l_func_name = bindings.__all__[l - 1]
    l_func = getattr(bindings, l_func_name)
    triton_result = l_func.apply(x, y, z)
    assert triton_result.shape == e3nn_result.shape
    # loop over spherical harmonics terms so we can get informative results
    dim_mismatchs = [
        torch.allclose(triton_result[..., i], e3nn_result[..., i], atol=ATOL, rtol=RTOL)
        for i in range(e3nn_result.size(-1))
    ]
    if not all(dim_mismatchs):
        bad_dims = [i for i, test in enumerate(dim_mismatchs) if not test]
        raise AssertionError(
            f"Forward call mismatch on l={l} for dimensions {bad_dims}"
        )
    triton_result.backward(gradient=torch.ones_like(triton_result))
    triton_grad = joint.grad.clone().detach()
    joint.grad = None
    # check the tensor outputs
    assert triton_grad.shape == e3nn_grad.shape
    dim_mismatchs = [
        torch.allclose(triton_grad[..., i], e3nn_grad[..., i], atol=ATOL, rtol=RTOL)
        for i in range(e3nn_grad.size(-1))
    ]
    if not all(dim_mismatchs):
        bad_dims = [i for i, test in enumerate(dim_mismatchs) if not test]
        raise AssertionError(
            f"Backward call mismatch on l={l} for dimensions {bad_dims}"
        )
