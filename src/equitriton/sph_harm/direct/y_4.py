import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["FourthOrderSphericalHarmonic"]


class FourthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        output_tensor: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
        col_offset: int = 0,
    ):
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.empty(
                (*coords.shape[:-1], 9), dtype=coords.dtype, device=coords.device
            )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        fourth_order_fwd[num_blocks,](
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
        block_size: int = 64,
        col_offset: int = 0,
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # call backward kernel
        fourth_order_bwd[num_blocks,](
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
    CONST000 = 1.12500000000000
    CONST001 = 2.25000000000000
    CONST002 = 3.00000000000000
    CONST005 = 2.21852991866236
    CONST007 = 9.48683298050514
    CONST010 = 20.1246117974981
    CONST011 = -18.8248505970167
    CONST012 = -13.3111795119741
    CONST013 = -10.0623058987491
    CONST014 = -9.00000000000000
    CONST015 = -8.87411967464942
    CONST016 = -7.11512473537885
    CONST017 = -6.27495019900557
    CONST018 = -3.35410196624968
    CONST019 = -1.67705098312484
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    # -------------------- kernel implementations
    Y00 = CONST015 * VAR07 * z - CONST015 * VAR25 * x
    Y01 = y * (-CONST011 * VAR26 * x + CONST017 * VAR07)
    Y02 = CONST018 * VAR07 * z + x * (CONST010 * VAR17 * z + CONST018 * VAR25)
    Y03 = CONST016 * VAR07 * y + x * (CONST007 * VAR16 + CONST016 * VAR26 * y)
    Y04 = (
        CONST000 * VAR06
        + CONST000 * VAR24
        + CONST002 * VAR15
        + CONST014 * VAR17 * VAR26
        + VAR08 * (CONST001 * VAR26 + CONST014 * VAR17)
    )
    Y05 = CONST016 * VAR25 * y + z * (CONST007 * VAR16 + CONST016 * VAR08 * y)
    Y06 = (
        -CONST019 * VAR06
        + CONST019 * VAR24
        + VAR17 * (CONST013 * VAR08 - CONST013 * VAR26)
    )
    Y07 = y * (CONST011 * VAR08 * z - CONST017 * VAR25)
    Y08 = CONST005 * VAR06 + CONST005 * VAR24 + CONST012 * VAR08 * VAR26
    tensors = [Y00, Y01, Y02, Y03, Y04, Y05, Y06, Y07, Y08]
    return torch.cat(tensors, dim=-1)


@triton.jit
def fourth_order_fwd(
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
    CONST000 = 1.12500000000000
    CONST001 = 2.25000000000000
    CONST002 = 3.00000000000000
    CONST005 = 2.21852991866236
    CONST007 = 9.48683298050514
    CONST010 = 20.1246117974981
    CONST011 = -18.8248505970167
    CONST012 = -13.3111795119741
    CONST013 = -10.0623058987491
    CONST014 = -9.00000000000000
    CONST015 = -8.87411967464942
    CONST016 = -7.11512473537885
    CONST017 = -6.27495019900557
    CONST018 = -3.35410196624968
    CONST019 = -1.67705098312484
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    # -------------------- kernel implementations
    Y00 = CONST015 * VAR07 * z - CONST015 * VAR25 * x
    Y01 = y * (-CONST011 * VAR26 * x + CONST017 * VAR07)
    Y02 = CONST018 * VAR07 * z + x * (CONST010 * VAR17 * z + CONST018 * VAR25)
    Y03 = CONST016 * VAR07 * y + x * (CONST007 * VAR16 + CONST016 * VAR26 * y)
    Y04 = (
        CONST000 * VAR06
        + CONST000 * VAR24
        + CONST002 * VAR15
        + CONST014 * VAR17 * VAR26
        + VAR08 * (CONST001 * VAR26 + CONST014 * VAR17)
    )
    Y05 = CONST016 * VAR25 * y + z * (CONST007 * VAR16 + CONST016 * VAR08 * y)
    Y06 = (
        -CONST019 * VAR06
        + CONST019 * VAR24
        + VAR17 * (CONST013 * VAR08 - CONST013 * VAR26)
    )
    Y07 = y * (CONST011 * VAR08 * z - CONST017 * VAR25)
    Y08 = CONST005 * VAR06 + CONST005 * VAR24 + CONST012 * VAR08 * VAR26
    output_striding = tl.arange(0, block_size) * output_stride
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
    tl.store(
        output_ptr + output_row_offset + 7,
        Y07,
        mask=output_row_offset + 7 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 8,
        Y08,
        mask=output_row_offset + 8 < output_numel,
    )


@triton.jit
def fourth_order_bwd(
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
    g_7 = tl.load(
        sph_grad_ptr + output_row_offset + 7, mask=output_row_offset + 7 < output_numel
    )
    g_8 = tl.load(
        sph_grad_ptr + output_row_offset + 8, mask=output_row_offset + 8 < output_numel
    )
    # -------------------- variable and constant definitions
    CONST000 = 2.00000000000000
    CONST001 = 4.50000000000000
    CONST002 = 2.25000000000000
    CONST006 = 9.48683298050514
    CONST008 = 12.0000000000000
    CONST012 = 28.4604989415154
    CONST014 = 40.2492235949962
    CONST015 = -37.6497011940334
    CONST016 = -6.70820393249937
    CONST017 = -26.6223590239483
    CONST018 = -21.3453742061366
    CONST019 = -20.1246117974981
    CONST020 = -18.8248505970167
    CONST021 = -18.0000000000000
    CONST022 = -14.2302494707577
    CONST023 = -10.0623058987491
    CONST024 = -9.00000000000000
    CONST025 = -8.87411967464942
    CONST026 = -7.11512473537885
    CONST027 = -6.27495019900557
    CONST028 = -3.35410196624968
    VAR07 = x * x * x
    VAR08 = x * x
    VAR16 = y * y * y
    VAR17 = y * y
    VAR25 = z * z * z
    VAR26 = z * z
    # -------------------- kernel implementations
    g_x = tl.load(
        coord_grad_ptr + coord_row_offset, mask=coord_row_offset < coord_numel
    )
    g_y = tl.load(
        coord_grad_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 < coord_numel
    )
    g_z = tl.load(
        coord_grad_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 < coord_numel
    )
    g_x += (
        CONST015 * g_7 * x * y * z
        + CONST022 * g_5 * x * y * z
        + g_0 * (CONST017 * VAR08 * z - CONST025 * VAR25)
        + g_1 * y * (CONST020 * VAR08 - CONST020 * VAR26)
        + g_2 * (-CONST019 * VAR17 * z + CONST023 * VAR08 * z + CONST028 * VAR25)
        + g_3 * (CONST006 * VAR16 + CONST018 * VAR08 * y + CONST026 * VAR26 * y)
        + g_4
        * (CONST000 * x * (CONST002 * VAR26 + CONST024 * VAR17) + CONST001 * VAR07)
        + g_6 * (-CONST016 * VAR07 + CONST019 * VAR17 * x)
        + g_8 * (CONST017 * VAR26 * x - CONST025 * VAR07)
    )
    g_y += (
        CONST000 * g_6 * y * (CONST023 * VAR08 - CONST023 * VAR26)
        + CONST014 * g_2 * x * y * z
        + g_1 * (-CONST020 * VAR26 * x + CONST027 * VAR07)
        + g_3 * (CONST026 * VAR07 + x * (CONST012 * VAR17 + CONST026 * VAR26))
        + g_4 * (CONST008 * VAR16 + CONST021 * VAR08 * y + CONST021 * VAR26 * y)
        + g_5 * (CONST026 * VAR25 + z * (CONST012 * VAR17 + CONST026 * VAR08))
        + g_7 * (CONST020 * VAR08 * z - CONST027 * VAR25)
    )
    g_z += (
        -CONST015 * g_1 * x * y * z
        + CONST022 * g_3 * x * y * z
        + g_0 * (-CONST017 * VAR26 * x + CONST025 * VAR07)
        + g_2 * (CONST028 * VAR07 + x * (-CONST019 * VAR17 + CONST023 * VAR26))
        + g_4 * (CONST001 * VAR08 * z + CONST001 * VAR25 + CONST021 * VAR17 * z)
        + g_5 * (CONST006 * VAR16 + CONST018 * VAR26 * y + CONST026 * VAR08 * y)
        + g_6 * (CONST016 * VAR25 - CONST019 * VAR17 * z)
        + g_7 * y * (CONST020 * VAR08 - CONST020 * VAR26)
        + g_8 * (CONST017 * VAR08 * z - CONST025 * VAR25)
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
