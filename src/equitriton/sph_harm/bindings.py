# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import torch
import triton
import numpy as np

from equitriton.sph_harm import triton_kernels as tk


__all__ = [
    "FirstOrderSphericalHarmonics",
    "SecondOrderSphericalHarmonics",
    "ThirdOrderSphericalHarmonics",
    "FourthOrderSphericalHarmonics",
]


def _num_projections(l: int) -> int:  # noqa: E741
    """Calculate the number of projections of m based on l"""
    return 2 * l + 1


def total_projections(l_max: int) -> int:
    """Calculate the total number of projects for a given l_max"""
    return sum([_num_projections(m) for m in range(l_max + 1)])


def make_output_tensor(x: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """Create a list of tensors with the correct size and mapping to be concatenated afterwards"""
    total_num_projections = total_projections(l_max)
    last_dim = x.size(-1)
    remainder = x.shape[:-1]
    # add an extra 1 dimension to the end to facilitate concatenation
    output = [
        torch.empty((*remainder, last_dim, 1), dtype=x.dtype, device=x.device)
        for _ in range(total_num_projections)
    ]
    return output


def split_tensor_by_l(
    joint_tensor: torch.Tensor, l_max: int, dim: int = -1
) -> list[torch.Tensor]:
    """The reverse operation of the concatenate step"""
    num_projections = [total_projections(l_value) for l_value in range(l_max + 1)]
    proj_indices = list(np.cumsum(num_projections) - 1)
    # the first output is empty, so we exclude it
    return torch.tensor_split(joint_tensor, proj_indices, dim=dim)[1:]


def slice_and_dice_tensor(joint_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Completely slices up a tensor along the last dimension, returning N views of an N length dimension."""
    num_slices = joint_tensor.size(-1)
    slice_indices = np.arange(num_slices).tolist()
    # the first output is empty, so we exclude it
    result = torch.tensor_split(joint_tensor, slice_indices, dim=-1)[1:]
    return result


class FirstOrderSphericalHarmonics(torch.autograd.Function):
    """
    First order spherical harmonics. This doesn't actually
    even use Triton, but is implement for consistency in the
    interface.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # for the current parallel model to work, the pointers must be contiguous!
        # otherwise the result will be completely scrambled, as the output tensor
        # indexing will be mismatched from xyz
        # TODO: move this to the high level wrapper
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 1)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_first_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)
        # the expected shape is [..., num_projections]
        output = torch.cat(output_tensors, dim=-1)
        # remove contributions from padded nodes
        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # derivative of projections of each spherical harmonic order
        # zeroth order is constant and doesn't contribute derivatives
        d_sph_0, d_sph_1_x, d_sph_1_y, d_sph_1_z = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        # factor of sqrt3 for all values
        sqrt3 = 3**0.5
        # we expect three tensors back for xyz
        x_grad = d_sph_1_x * sqrt3
        y_grad = d_sph_1_y * sqrt3
        z_grad = d_sph_1_z * sqrt3
        # intended gradients should be shape [num_nodes] per coordinate
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class SecondOrderSphericalHarmonics(torch.autograd.Function):
    """
    Second order spherical harmonics. A little more involved than
    the first order case, and actually gives something interesting
    to look at.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 2)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_second_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],  # unpack pointers without verbosity
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z)
        output = torch.cat(output_tensors, dim=-1)
        # remove contributions from padded nodes
        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # derivative of projections of each spherical harmonic order
        # zeroth order is constant and doesn't contribute derivatives
        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_second_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class ThirdOrderSphericalHarmonics(torch.autograd.Function):
    """
    Third order spherical harmonics. Starting to get more cookiecutter.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 3)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_third_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],  # unpack pointers without verbosity
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)
        output = torch.cat(output_tensors, dim=-1)
        # remove contributions from padded nodes
        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # derivative of projections of each spherical harmonic order
        # zeroth order is constant and doesn't contribute derivatives
        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_third_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask


class FourthOrderSphericalHarmonics(torch.autograd.Function):
    """
    Fourth order spherical harmonics. Starting to get tediuous...
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()
        output_tensors = make_output_tensor(x, 4)
        output_tensors[0][:] = 1.0
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_fourth_order_fwd[num_blocks,](
            x,
            y,
            z,
            *output_tensors[1:],  # unpack pointers without verbosity
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        ctx.save_for_backward(x, y, z, mask)
        output = torch.cat(output_tensors, dim=-1)
        # remove contributions from padded nodes
        if isinstance(mask, torch.Tensor):
            output = output[mask]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # derivative of projections of each spherical harmonic order
        # zeroth order is constant and doesn't contribute derivatives
        gradient_collection = slice_and_dice_tensor(grad_output)
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 3:
            x, y, z = saved_tensors
            mask = None
        else:
            x, y, z, mask = saved_tensors
        x_grad = torch.zeros_like(x)
        y_grad = torch.zeros_like(y)
        z_grad = torch.zeros_like(z)
        block_size = 256
        vector_length = x.numel()
        # ceiling divide makes sure it works for block sizes larger than
        # the total number of samples
        num_blocks = triton.next_power_of_2(triton.cdiv(vector_length, block_size))
        tk._triton_fourth_order_bwd[num_blocks,](
            x,
            y,
            z,
            x_grad,
            y_grad,
            z_grad,
            *gradient_collection[1:],
            BLOCK_SIZE=block_size,
            vector_length=vector_length,
        )
        return x_grad.squeeze(), y_grad.squeeze(), z_grad.squeeze(), mask
