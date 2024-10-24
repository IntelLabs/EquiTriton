import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["FusedSecondOrderSphericalHarmonic"]


class FusedSecondOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
    ):
        output_tensor = torch.empty(
            (*coords.shape[:-1], 9), dtype=coords.dtype, device=coords.device
        )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        joint_second_order_fwd[num_blocks,](
            coords, output_tensor, block_size, coord_numel, output_numel
        )
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(
        ctx, sph_grad_tensor: torch.Tensor, block_size: int = 64
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # call backward kernel
        joint_second_order_bwd[num_blocks,](
            coords,
            coord_grad_output,
            sph_grad_tensor,
            block_size,
            coords.numel(),
            sph_grad_tensor.numel(),
        )
        return coord_grad_output


def _torch_fwd(coords: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of the kernel. This is designed
    purely for unit testing to ensure that the Triton implementation
    is behaving as intended.

    This function is generically named to make it easier for
    it to be called programmatically: it is _not_ intended
    to be used manually.

    Parameters
    ----------
    coords : torch.Tensor
        N-d tensor, where the last dimension corresponds to
        xyz values.

    Returns
    -------
    torch.Tensor
        N-d tensor, where the last dimension corresponds to
        each projection of the second order spherical harmonic.
    """
    x = coords[..., 0].contiguous().unsqueeze(-1)
    y = coords[..., 1].contiguous().unsqueeze(-1)
    z = coords[..., 2].contiguous().unsqueeze(-1)
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = -1.11803398874989
    CONST_03 = 1.93649167310371
    CONST_04 = 3**0.5
    Y00 = torch.ones_like(x)
    Y10 = x * CONST_04
    Y11 = y * CONST_04
    Y12 = z * CONST_04
    Y20 = CONST_00 * x * z
    Y21 = CONST_00 * x * y
    Y23 = CONST_00 * y * z  # looks jarring but just helping the compiler ;)
    Y22 = CONST_02 * x * x + CONST_01 * y * y + CONST_02 * z * z
    Y24 = -CONST_03 * x * x + CONST_03 * z * z
    return torch.cat([Y00, Y10, Y11, Y12, Y20, Y21, Y22, Y23, Y24], dim=-1)


@triton.jit
def joint_second_order_fwd(
    coord_ptr: tl.tensor,
    output_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
):
    """
    This Triton implementation includes l=0, 1, 2 within the
    same kernel, as it would be a common operation.
    """
    # these are hardcoded because they are predetermined;
    coord_stride = 3
    # work out the row offsets
    block_id = tl.program_id(0)
    coord_striding = tl.arange(0, block_size) * coord_stride
    # as the name suggests, this is effectively every node/atom
    coord_row_offset = coord_striding + (block_size * coord_stride * block_id)
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(
        coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    z = tl.load(
        coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = -1.11803398874989
    CONST_03 = 1.93649167310371
    CONST_04 = tl.sqrt(3.0)
    Y10 = CONST_04 * x
    Y11 = CONST_04 * y
    Y12 = CONST_04 * z
    Y20 = CONST_00 * x * z
    Y21 = CONST_00 * x * y
    Y23 = CONST_00 * y * z  # looks jarring but just helping the compiler ;)
    Y22 = CONST_02 * x * x + CONST_01 * y * y + CONST_02 * z * z
    Y24 = -CONST_03 * x * x + CONST_03 * z * z
    output_stride = 9  # sum of [2l + 1] over l=0, 1, 2
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id)
    # first column are all zeros, per zeroth order
    tl.store(output_ptr + output_row_offset, 1.0, mask=output_row_offset < output_numel)
    tl.store(
        output_ptr + output_row_offset + 1,
        Y10,
        mask=output_row_offset + 1 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 2,
        Y11,
        mask=output_row_offset + 2 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 3,
        Y12,
        mask=output_row_offset + 3 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 4,
        Y20,
        mask=output_row_offset + 4 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 5,
        Y21,
        mask=output_row_offset + 5 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 6,
        Y22,
        mask=output_row_offset + 6 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 7,
        Y23,
        mask=output_row_offset + 6 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 8,
        Y24,
        mask=output_row_offset + 7 < output_numel,
    )


@triton.jit
def joint_second_order_bwd(
    coord_ptr: tl.tensor,
    coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
):
    # work out the row offsets
    block_id = tl.program_id(0)
    # these are hardcoded because they are predetermined;
    coord_stride = 3
    coord_striding = tl.arange(0, block_size) * coord_stride
    # as the name suggests, this is effectively every node/atom
    coord_row_offset = coord_striding + (block_size * coord_stride * block_id)
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset < coord_numel)
    y = tl.load(
        coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    z = tl.load(
        coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    output_stride = 9  # [2l + 1]
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id)
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = 4.47213595499958
    CONST_03 = tl.sqrt(3.0)
    # load in gradients w.r.t. spherical harmonic projections.
    # gradient of l = 0 goes to zero
    g_Y10 = tl.load(
        sph_grad_ptr + output_row_offset + 1, mask=output_row_offset + 1 < output_numel
    )
    g_Y11 = tl.load(
        sph_grad_ptr + output_row_offset + 2, mask=output_row_offset + 2 < output_numel
    )
    g_Y12 = tl.load(
        sph_grad_ptr + output_row_offset + 3, mask=output_row_offset + 3 < output_numel
    )
    g_Y20 = tl.load(
        sph_grad_ptr + output_row_offset + 4, mask=output_row_offset + 4 < output_numel
    )
    g_Y21 = tl.load(
        sph_grad_ptr + output_row_offset + 5, mask=output_row_offset + 5 < output_numel
    )
    g_Y22 = tl.load(
        sph_grad_ptr + output_row_offset + 6, mask=output_row_offset + 6 < output_numel
    )
    g_Y23 = tl.load(
        sph_grad_ptr + output_row_offset + 7, mask=output_row_offset + 7 < output_numel
    )
    g_Y24 = tl.load(
        sph_grad_ptr + output_row_offset + 8, mask=output_row_offset + 8 < output_numel
    )
    g_x = (
        CONST_00 * g_Y20 * z
        + CONST_00 * g_Y21 * y
        - CONST_01 * g_Y22 * x
        - CONST_00 * g_Y24 * x
        + CONST_03 * g_Y10
    )
    g_y = (
        CONST_00 * g_Y21 * x
        + CONST_02 * g_Y22 * y
        + CONST_00 * g_Y23 * z
        + CONST_03 * g_Y11
    )
    g_z = (
        CONST_00 * g_Y20 * x
        - CONST_01 * g_Y22 * z
        + CONST_00 * g_Y23 * y
        + CONST_00 * g_Y24 * z
        + CONST_03 * g_Y12
    )
    # write out gradients
    tl.store(
        coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset < coord_numel
    )
    tl.store(
        coord_grad_ptr + coord_row_offset + 1,
        g_y,
        mask=coord_row_offset + 1 < coord_numel,
    )
    tl.store(
        coord_grad_ptr + coord_row_offset + 2,
        g_z,
        mask=coord_row_offset + 2 < coord_numel,
    )
