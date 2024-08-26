from __future__ import annotations

import math

import torch
import triton
from e3nn import o3

__all__ = ["pad_tensor_to_power", "calculate_lastdim_num_blocks"]


def pad_tensor_to_power(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a tensor to the nearest power of two.

    The goal of this is to minimize the number of compiled
    kernels due to large variations in tensor shapes. By
    padding to the nearest power of two, we hopefully only
    encounter typical tensor shapes, with the cost of a bit
    of memory overhead.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor to be padded.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A 2-tuple of tensors: the first is the padded tensor,
        and the second is a 1D mask to be applied along the
        node dimension of a tensor.
    """
    num_nodes = input_tensor.size(0)
    pad_size = triton.next_power_of_2(num_nodes)
    num_pad = pad_size - num_nodes
    # allocate tensor of zeros to pad with
    zero_pad = torch.zeros(
        (num_pad, *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    joint_tensor = torch.cat([input_tensor, zero_pad], dim=0)
    mask = torch.ones(pad_size, device=joint_tensor.device, dtype=torch.bool)
    mask[num_nodes:] = False
    return (joint_tensor, mask)


def calculate_lastdim_num_blocks(input_tensor: torch.Tensor, block_size: int) -> int:
    """
    Calculate the number of blocks for a tensor, assuming we
    stride along the last dimension, and a given block size.

    The corresponding pointer arithmetic looks like this:

    ```python
    block_id = tl.program_id(0)
    striding = tl.arange(0, block_size) * stride
    offset = (striding + (block_size * stride * block_id))
    ```

    This function is used to work out the amount of parallel
    work that needs to be done, given as the total number of
    elements divided by the last dimension stride, and a specified
    block size that will then divvy up the work.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Torch N-d tensor to operate over.

    Returns
    -------
    int
        Number of blocks of work, given a block size.
    """
    # get the stride of the last dimension
    stride = input_tensor.stride(-2)
    numel = input_tensor.numel()
    total_blocks = math.ceil(numel / stride)
    return total_blocks


def unravel_index(tensor: torch.Tensor, index: int) -> tuple[int, ...]:
    """
    For a given N-d tensor and a 1D index, work out the corresponding
    index tuple for the N-d tensor.

    This is equivalent to the `torch.unravel_index` function, but
    makes it a bit more friendlier in terms of Python types.

    Parameters
    ----------
    tensor : torch.Tensor
        Torch N-D tensor to index.
    index : int
        1D index value to map onto an N-tuple, where N
        is the dimensionality of the tensor. Must be
        greater or equal to zero, and smaller than the
        total number of elements.

    Returns
    -------
    tuple[int, ...]
        An N-tuple of integers corresponding to the
        N-d index of the provided index.
    """
    # make sure that the index is within bounds
    assert 0 <= index < tensor.numel()
    indices = []
    for size in reversed(tensor.shape):
        indices.append(index % size)
        index //= size
    return tuple(reversed(indices))


def spherical_harmonics_irreps(l_values: list[int], num_feat: int = 1) -> o3.Irreps:
    """
    Generate the set of irreducible representations given a list of
    arbitrary l values; i.e. they need not be contiguous.

    While ``l_values`` does not need to be contiguous, this function
    will sort in ascending order of ``l``, such that the returned
    representations are in order. This makes it a lot more straightforward
    for building off of.

    Parameters
    ----------
    l_values : list[int]
        List of l values to generate representations for.
    num_feat : int
        Number of features for the associated representations.
        Defaults to 1, which can be used for specifying a spherical
        harmonic basis, but values greater than one can be used to
        specify weights.

    Returns
    -------
    o3.Irreps
        Irreducible representations for the set of spherical harmonics.
    """
    assert num_feat > 1, "Number of features must be positive!"
    joint = []
    for l in sorted(l_values):
        parity = "e" if (-1) ** l > 0 else "o"
        joint.append(f"{num_feat}x{l}{parity}")
    return o3.Irreps("+".join(joint))
