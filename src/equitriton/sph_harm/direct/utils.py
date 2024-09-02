from __future__ import annotations

from importlib import import_module
from typing import Callable

import torch
import numpy as np

from equitriton.utils import num_irreps_projections, calculate_lastdim_num_blocks

__all__ = ["torch_spherical_harmonic", "triton_spherical_harmonic"]

BLOCK_SIZE = 64


def _get_fwd_kernel(l: int) -> Callable:
    """
    Reach into the module of a specified l value and grab
    the corresponding forward Triton kernel function.

    Parameters
    ----------
    l : int
        Spherical harmonic l value to search for.

    Returns
    -------
    Callable
        Triton forward kernel

    Raises
    ------
    ModuleNotFoundError:
        If the l value is not implemented, the module will
        not exist and raises a ``ModuleNotFoundError``.
    RuntimeError:
        If the module exists but we aren't able to find
        a forward kernel defined, it's broken.
    """
    try:
        target_module = import_module(f"equitriton.sph_harm.direct.y_{l}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Spherical harmonic order l={l} requested, but not found!"
        ) from e
    defined_objs = dir(target_module)
    for key in defined_objs:
        if "order_fwd" in key:
            sph_harm_func = getattr(target_module, key)
            return sph_harm_func
    raise RuntimeError(f"Namespace for module l={l} is broken!")


def _get_bwd_kernel(l: int) -> Callable:
    """
    Reach into the module of a specified l value and grab
    the corresponding backward Triton kernel function.

    Parameters
    ----------
    l : int
        Spherical harmonic l value to search for.

    Returns
    -------
    Callable
        Triton backward kernel

    Raises
    ------
    ModuleNotFoundError:
        If the l value is not implemented, the module will
        not exist and raises a ``ModuleNotFoundError``.
    RuntimeError:
        If the module exists but we aren't able to find
        a backward kernel defined, it's broken.
    """
    try:
        target_module = import_module(f"equitriton.sph_harm.direct.y_{l}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Spherical harmonic order l={l} requested, but not found!"
        ) from e
    defined_objs = dir(target_module)
    for key in defined_objs:
        if "order_bwd" in key:
            sph_harm_func = getattr(target_module, key)
            return sph_harm_func
    raise RuntimeError(f"Namespace for module l={l} is broken!")


def torch_spherical_harmonic(l: int, coords: torch.Tensor) -> torch.Tensor:
    """
    Utility function that will call the PyTorch implementation
    of a spherical harmonic order.

    This is not intended for production use, but mainly for
    sanity checking and convenience.

    Parameters
    ----------
    l : int
        Order of spherical harmonic requested.
    coords : torch.Tensor
        N-d tensor, where the last dimension should correspond
        to xyz vectors.

    Returns
    -------
    torch.Tensor
        N-d tensor of the same dimensionality as the input coordinates,
        but the size of the last dimension equal to [2 * l + 1].

    Raises
    ------
    ModuleNotFoundError
        If order of spherical harmonic requested is not found, it is
        likely not yet implemented.
    RuntimeError
        If the PyTorch implementation of the spherical harmonic is
        not found within the module.
    RuntimeError
        If the shape of the last dimension of the ``coords`` tensor
        is not equal to three.
    """
    try:
        target_module = import_module(f"equitriton.sph_harm.direct.y_{l}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Spherical harmonic order l={l} requested, but not found!"
        ) from e
    torch_func = getattr(target_module, "_torch_fwd", None)
    if not torch_func:
        raise RuntimeError(f"PyTorch implementation of l={l} not found.")
    if coords.size(-1) != 3:
        raise RuntimeError("Expects last dimension of coordinate tensor to be 3!")
    return torch_func(coords)


def triton_spherical_harmonic(
    l_values: int | list[int], coords: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Utility function that will call the Triton implementation
    of a spherical harmonic order.

    This is not intended for production use, but mainly for
    sanity checking and convenience.

    Parameters
    ----------
    l : int
        Order of spherical harmonic requested.
    coords : torch.Tensor
        N-d tensor, where the last dimension should correspond
        to xyz vectors.

    Returns
    -------
    torch.Tensor
        N-d tensor of the same dimensionality as the input coordinates,
        but the size of the last dimension equal to [2 * l + 1].

    Raises
    ------
    ModuleNotFoundError
        If order of spherical harmonic requested is not found, it is
        likely not yet implemented.
    RuntimeError
        If the Triton implementation of the spherical harmonic is
        not found within the module.
    RuntimeError
        If the shape of the last dimension of the ``coords`` tensor
        is not equal to three.
    """
    if coords.size(-1) != 3:
        raise RuntimeError("Expects last dimension of coordinate tensor to be 3!")
    if isinstance(l_values, int):
        l_values = [
            l_values,
        ]
    # ensure we are in ascending order
    l_values = list(sorted(l_values))
    dims = [num_irreps_projections(l) for l in l_values]
    offsets = np.zeros_like(dims)
    # prepend zero, since we start with zero offset
    offsets[1:] = np.cumsum(dims[:-1])

    # convert into a list, since np.int64 is not desired
    offsets = offsets.tolist()
    # preallocate a tensor that holds all of the spherical harmonic terms
    output_tensor = torch.empty(
        (*coords.shape[:-1], sum(dims)),
        device=coords.device,
        dtype=coords.dtype,
        requires_grad=True,
    )
    for l, offset in zip(l_values, offsets):
        sph_harm_func = _get_fwd_kernel(l)
        sph_harm_func.apply(coords, output_tensor, mask, BLOCK_SIZE, offset)
    return output_tensor


class TritonSphericalHarmonic(torch.autograd.Function):
    __l_values__: list
    __offsets__: list

    @staticmethod
    def forward(
        ctx,
        l_values: int | list[int],
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if coords.size(-1) != 3:
            raise RuntimeError("Expects last dimension of coordinate tensor to be 3!")
        if isinstance(l_values, int):
            l_values = [
                l_values,
            ]
        # ensure we are in ascending order
        l_values = list(sorted(l_values))
        dims = [num_irreps_projections(l) for l in l_values]
        offsets = np.zeros_like(dims)
        # prepend zero, since we start with zero offset
        offsets[1:] = np.cumsum(dims[:-1])
        # convert into a list, since np.int64 is not desired
        offsets = offsets.tolist()
        # preallocate a tensor that holds all of the spherical harmonic terms
        output_tensor = torch.empty(
            (*coords.shape[:-1], sum(dims)),
            device=coords.device,
            dtype=coords.dtype,
            requires_grad=True,
        )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        # this corresponds to the number of projections
        output_stride = output_tensor.stride(-2)
        num_blocks = calculate_lastdim_num_blocks(coords, BLOCK_SIZE)
        for l, offset in zip(l_values, offsets):
            sph_harm_func = _get_fwd_kernel(l)
            sph_harm_func[num_blocks,](
                coords,
                output_tensor,
                BLOCK_SIZE,
                coord_numel,
                output_numel,
                offset,
                output_stride,
            )
        ctx.save_for_backward(coords)
        # stash values as class attributes, as they are the same
        # and ctx can only hold tensors
        TritonSphericalHarmonic.__l_values__ = l_values
        TritonSphericalHarmonic.__offsets__ = offsets
        return output_tensor

    @staticmethod
    def backward(ctx, sph_harm_grads: torch.Tensor):
        (coords,) = ctx.saved_tensors
        # grab from private class variables
        l_values = TritonSphericalHarmonic.__l_values__
        offsets = TritonSphericalHarmonic.__offsets__
        coord_grad_output = torch.zeros_like(coords)
        # combine start and end together to slice the gradient tensor
        coord_numel = coords.numel()
        grads_numel = sph_harm_grads.numel()
        # this corresponds to the number of projections
        output_stride = sph_harm_grads.stride(-2)
        num_blocks = calculate_lastdim_num_blocks(coords, BLOCK_SIZE)
        for l, offset in zip(l_values, offsets):
            sph_harm_bwd = _get_bwd_kernel(l)
            sph_harm_bwd[num_blocks,](
                coords,
                coord_grad_output,
                sph_harm_grads,
                BLOCK_SIZE,
                coord_numel,
                grads_numel,
                offset,
                output_stride,
            )
        # first element ise None becausey are l_values which
        # can't have gradients
        return None, coord_grad_output
