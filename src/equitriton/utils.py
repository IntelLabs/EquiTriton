from __future__ import annotations

import torch
import triton

__all__ = ["pad_tensor_to_power"]


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
