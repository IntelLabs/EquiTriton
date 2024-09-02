import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["ThirdOrderSphericalHarmonic"]


class ThirdOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        output_tensor: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ):
        # allocate a tensor if one isn't given
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.empty(
                (*coords.shape[:-1], 7), dtype=coords.dtype, device=coords.device
            )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        third_order_fwd[num_blocks,](
            coords,
            output_tensor,
            block_size,
            coord_numel,
            output_numel,
            col_offset,
            output_tensor.stride(-2),
        )
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(
        ctx,
        sph_grad_tensor: torch.Tensor,
        coord_grad_output: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        if not isinstance(coord_grad_output, torch.Tensor):
            coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # call backward kernel
        third_order_bwd[num_blocks,](
            coords,
            coord_grad_output,
            sph_grad_tensor,
            block_size,
            coords.numel(),
            sph_grad_tensor.numel(),
            col_offset,
            sph_grad_tensor.stride(-2),
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
    # -------------------- variable and constant definitions
    CONST000 = 2.64575131106459
    CONST002 = 5.12347538297980
    CONST004 = 6.48074069840786
    CONST005 = 10.2469507659596
    CONST006 = -2.09165006633519
    CONST007 = -1
    CONST008 = -6.27495019900557
    CONST009 = -3.96862696659689
    CONST010 = -1.62018517460197
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    # -------------------- kernel implementations
    Y00 = CONST006 * VAR07 - CONST008 * VAR26 * x
    Y01 = CONST005 * x * y * z
    Y02 = CONST010 * VAR07 + x * (CONST004 * VAR17 + CONST010 * VAR26)
    Y03 = CONST000 * VAR16 + CONST009 * VAR08 * y + CONST009 * VAR26 * y
    Y04 = CONST010 * VAR25 + z * (CONST004 * VAR17 + CONST010 * VAR08)
    Y05 = CONST002 * y * (CONST007 * VAR08 + VAR26)
    Y06 = -CONST006 * VAR25 + CONST008 * VAR08 * z
    tensors = [Y00, Y01, Y02, Y03, Y04, Y05, Y06]
    return torch.cat(tensors, dim=-1)


@triton.jit
def third_order_fwd(
    coord_ptr: tl.tensor,
    output_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
    col_offset: tl.constexpr,
    output_stride: tl.constexpr,
):
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
    # -------------------- variable and constant definitions
    CONST000 = 2.64575131106459
    CONST002 = 5.12347538297980
    CONST004 = 6.48074069840786
    CONST005 = 10.2469507659596
    CONST006 = -2.09165006633519
    CONST007 = -1
    CONST008 = -6.27495019900557
    CONST009 = -3.96862696659689
    CONST010 = -1.62018517460197
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    # -------------------- kernel implementations
    Y00 = CONST006 * VAR07 - CONST008 * VAR26 * x
    Y01 = CONST005 * x * y * z
    Y02 = CONST010 * VAR07 + x * (CONST004 * VAR17 + CONST010 * VAR26)
    Y03 = CONST000 * VAR16 + CONST009 * VAR08 * y + CONST009 * VAR26 * y
    Y04 = CONST010 * VAR25 + z * (CONST004 * VAR17 + CONST010 * VAR08)
    Y05 = CONST002 * y * (CONST007 * VAR08 + VAR26)
    Y06 = -CONST006 * VAR25 + CONST008 * VAR08 * z
    output_striding = tl.arange(0, block_size) * output_stride
    # zero on the row offset is the first spherical harmonic term of this order
    output_row_offset = (
        output_striding + (block_size * output_stride * block_id) + col_offset
    )
    tl.store(output_ptr + output_row_offset, Y00, mask=output_row_offset < output_numel)
    tl.store(
        output_ptr + output_row_offset + 1,
        Y01,
        mask=output_row_offset + 1 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 2,
        Y02,
        mask=output_row_offset + 2 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 3,
        Y03,
        mask=output_row_offset + 3 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 4,
        Y04,
        mask=output_row_offset + 4 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 5,
        Y05,
        mask=output_row_offset + 5 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 6,
        Y06,
        mask=output_row_offset + 6 < output_numel,
    )


@triton.jit
def third_order_bwd(
    coord_ptr: tl.tensor,
    coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
    col_offset: tl.constexpr,
    output_stride: tl.constexpr,
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
    output_striding = tl.arange(0, block_size) * output_stride
    # zero on the row offset is the first spherical harmonic term of this order
    output_row_offset = (
        output_striding + (block_size * output_stride * block_id) + col_offset
    )
    # load in gradients w.r.t. spherical harmonic projections
    g_0 = tl.load(
        sph_grad_ptr + output_row_offset, mask=output_row_offset < output_numel
    )
    g_1 = tl.load(
        sph_grad_ptr + output_row_offset + 1, mask=output_row_offset + 1 < output_numel
    )
    g_2 = tl.load(
        sph_grad_ptr + output_row_offset + 2, mask=output_row_offset + 2 < output_numel
    )
    g_3 = tl.load(
        sph_grad_ptr + output_row_offset + 3, mask=output_row_offset + 3 < output_numel
    )
    g_4 = tl.load(
        sph_grad_ptr + output_row_offset + 4, mask=output_row_offset + 4 < output_numel
    )
    g_5 = tl.load(
        sph_grad_ptr + output_row_offset + 5, mask=output_row_offset + 5 < output_numel
    )
    g_6 = tl.load(
        sph_grad_ptr + output_row_offset + 6, mask=output_row_offset + 6 < output_numel
    )
    # -------------------- variable and constant definitions
    CONST002 = 6.48074069840786
    CONST005 = 12.9614813968157
    CONST007 = -3.96862696659689
    CONST008 = -12.5499003980111
    CONST009 = -10.2469507659596
    CONST010 = -7.93725393319377
    CONST011 = -6.27495019900557
    CONST012 = -5.12347538297980
    CONST013 = -4.86055552380590
    CONST014 = -3.24037034920393
    CONST015 = -1.62018517460197
    VAR08 = x * x
    VAR17 = y * y
    VAR26 = z * z
    # -------------------- kernel implementations
    g_x = (
        CONST008 * g_6 * x * z
        - CONST009 * g_1 * y * z
        + CONST009 * g_5 * x * y
        + CONST010 * g_3 * x * y
        + CONST014 * g_4 * x * z
        + g_0 * (CONST011 * VAR08 - CONST011 * VAR26)
        + g_2 * (CONST002 * VAR17 + CONST013 * VAR08 + CONST015 * VAR26)
    )
    g_y = (
        CONST005 * g_2 * x * y
        + CONST005 * g_4 * y * z
        - CONST009 * g_1 * x * z
        + g_3 * (CONST007 * VAR08 + CONST007 * VAR26 - CONST010 * VAR17)
        + g_5 * (CONST012 * VAR08 - CONST012 * VAR26)
    )
    g_z = (
        -CONST008 * g_0 * x * z
        - CONST009 * g_1 * x * y
        - CONST009 * g_5 * y * z
        + CONST010 * g_3 * y * z
        + CONST014 * g_2 * x * z
        + g_4 * (CONST002 * VAR17 + CONST013 * VAR26 + CONST015 * VAR08)
        + g_6 * (CONST011 * VAR08 - CONST011 * VAR26)
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
