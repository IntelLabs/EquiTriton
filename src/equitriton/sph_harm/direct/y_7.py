import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["SeventhOrderSphericalHarmonic"]


class SeventhOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
    ):
        output_tensor = torch.empty(
            (*coords.shape[:-1], 15), dtype=coords.dtype, device=coords.device
        )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        seventh_order_fwd[num_blocks,](
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
        seventh_order_bwd[num_blocks,](
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
    CONST002 = 3.87298334620742
    CONST008 = 11.7655316231354
    CONST010 = 16.5555704843566
    CONST012 = 20.4939015319192
    CONST013 = 20.4939015319192
    CONST014 = 22.0740939791422
    CONST015 = 23.5310632462709
    CONST017 = 36.7901566319036
    CONST019 = 38.4260653723485
    CONST020 = 38.4260653723485
    CONST021 = 38.4260653723485
    CONST023 = -4.99169231699030
    CONST025 = 47.0621264925418
    CONST026 = 50.8329064189723
    CONST028 = 55.1852349478554
    CONST029 = 56.2781179722634
    CONST030 = 56.2781179722634
    CONST032 = 66.5558975598707
    CONST033 = 75.2994023880668
    CONST037 = 101.665812837945
    CONST038 = 110.370469895711
    CONST041 = 147.160626527614
    CONST042 = -1.66389743899677
    CONST043 = -9.37968632871057
    CONST044 = -1.66389743899677
    CONST045 = -220.740939791422
    CONST046 = -220.740939791422
    CONST047 = -1.60108605718119
    CONST048 = -187.593726574211
    CONST049 = -9.19753915797590
    CONST050 = -1.83950783159518
    CONST051 = -1.83950783159518
    CONST052 = -4.80325817154356
    CONST053 = -147.160626527614
    CONST054 = -140.695294930659
    CONST055 = -133.111795119741
    CONST056 = -125.499003980111
    CONST057 = -125.499003980111
    CONST058 = -99.8338463398060
    CONST059 = -87.7389315936062
    CONST060 = -76.8521307446970
    CONST061 = -66.5558975598707
    CONST062 = -62.7495019900557
    CONST063 = -52.6433589561637
    CONST064 = -44.1481879582843
    CONST065 = -44.3705983732471
    CONST066 = -40.6663251351779
    CONST067 = -40.6663251351779
    CONST068 = -8.31948719498384
    CONST069 = -37.6497011940334
    CONST070 = -33.2779487799353
    CONST071 = -25.4164532094862
    CONST072 = -25.4164532094862
    CONST073 = -17.5477863187212
    CONST074 = -11.7655316231354
    CONST075 = -11.0370469895711
    CONST076 = -9.19753915797590
    CONST077 = -8.47215106982872
    CONST078 = -4.80325817154356
    CONST079 = -2.50682661696018
    CONST080 = -1.60108605718119
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    # -------------------- kernel implementations
    Y00 = (
        CONST059 * VAR07 * VAR24
        - CONST063 * VAR05 * VAR26
        - CONST073 * VAR22 * x
        + CONST079 * VAR03
    )
    Y01 = y * (CONST029 * VAR23 * x + CONST030 * VAR05 * z + CONST048 * VAR07 * VAR25)
    Y02 = (
        CONST050 * VAR03
        + VAR05 * (CONST010 * VAR26 + CONST014 * VAR17)
        + VAR07 * (CONST045 * VAR17 * VAR26 - CONST076 * VAR24)
        + x * (CONST038 * VAR17 * VAR24 + CONST076 * VAR22)
    )
    Y03 = VAR16 * (CONST041 * VAR25 * x + CONST053 * VAR07 * z) + y * (
        -CONST064 * VAR05 * z + CONST064 * VAR23 * x
    )
    Y04 = (
        CONST042 * VAR03
        + VAR05 * (-CONST042 * VAR26 - CONST070 * VAR17)
        + VAR07 * (CONST061 * VAR17 * VAR26 + CONST065 * VAR15 - CONST068 * VAR24)
        + x * (-CONST023 * VAR22 - CONST055 * VAR15 * VAR26 + CONST058 * VAR17 * VAR24)
    )
    Y05 = (
        CONST015 * VAR05 * y * z
        + VAR07 * (CONST025 * VAR25 * y + CONST057 * VAR16 * z)
        + x * (CONST015 * VAR23 * y + CONST033 * VAR14 * z + CONST056 * VAR16 * VAR25)
    )
    Y06 = (
        CONST047 * VAR03
        + VAR05 * (CONST020 * VAR17 + CONST078 * VAR26)
        + VAR07 * (CONST052 * VAR24 + CONST060 * VAR15 - CONST060 * VAR17 * VAR26)
        + x
        * (
            CONST012 * VAR13
            + CONST019 * VAR17 * VAR24
            + CONST060 * VAR15 * VAR26
            + CONST080 * VAR22
        )
    )
    Y07 = (
        CONST002 * VAR12
        + VAR14 * (CONST066 * VAR08 + CONST067 * VAR26)
        + VAR16 * (CONST026 * VAR06 + CONST026 * VAR24 + CONST037 * VAR08 * VAR26)
        + y
        * (
            CONST071 * VAR06 * VAR26
            + CONST072 * VAR08 * VAR24
            + CONST077 * VAR04
            + CONST077 * VAR22
        )
    )
    Y08 = (
        CONST047 * VAR21
        + VAR23 * (CONST020 * VAR17 + CONST052 * VAR08)
        + VAR25 * (CONST052 * VAR06 - CONST060 * VAR08 * VAR17 + CONST060 * VAR15)
        + z
        * (
            CONST013 * VAR13
            + CONST021 * VAR06 * VAR17
            + CONST047 * VAR04
            + CONST060 * VAR08 * VAR15
        )
    )
    Y09 = (
        VAR14 * (CONST069 * VAR08 - CONST069 * VAR26)
        + VAR16 * (-CONST062 * VAR06 + CONST062 * VAR24)
        + y
        * (
            CONST008 * VAR08 * VAR24
            + CONST074 * VAR04
            + CONST074 * VAR06 * VAR26
            - CONST074 * VAR22
        )
    )
    Y10 = (
        -CONST042 * VAR21
        + VAR23 * (CONST044 * VAR08 + CONST070 * VAR17)
        + VAR25 * (CONST032 * VAR08 * VAR17 - CONST065 * VAR15 + CONST068 * VAR06)
        + z * (CONST023 * VAR04 + CONST055 * VAR08 * VAR15 - CONST058 * VAR06 * VAR17)
    )
    Y11 = VAR16 * (
        CONST017 * VAR06 + CONST017 * VAR24 + CONST046 * VAR08 * VAR26
    ) + y * (
        CONST028 * VAR06 * VAR26
        + CONST028 * VAR08 * VAR24
        + CONST075 * VAR04
        + CONST075 * VAR22
    )
    Y12 = (
        CONST051 * VAR21
        + VAR23 * (CONST010 * VAR08 + CONST014 * VAR17)
        + VAR25 * (CONST045 * VAR08 * VAR17 - CONST049 * VAR06)
        + z * (CONST038 * VAR06 * VAR17 + CONST049 * VAR04)
    )
    Y13 = y * (
        CONST043 * VAR04
        - CONST043 * VAR22
        - CONST054 * VAR06 * VAR26
        + CONST054 * VAR08 * VAR24
    )
    Y14 = (
        -CONST059 * VAR06 * VAR25
        + CONST063 * VAR08 * VAR23
        + CONST073 * VAR04 * z
        - CONST079 * VAR21
    )
    # not the prettiest way to concatenate, but better than
    # messing with the linter
    tensors = [
        Y00,
        Y01,
        Y02,
        Y03,
        Y04,
        Y05,
        Y06,
        Y07,
        Y08,
        Y09,
        Y10,
        Y11,
        Y12,
        Y13,
        Y14,
    ]
    return torch.cat(tensors, dim=-1)


@triton.jit
def seventh_order_fwd(
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
    CONST002 = 3.87298334620742
    CONST008 = 11.7655316231354
    CONST010 = 16.5555704843566
    CONST012 = 20.4939015319192
    CONST013 = 20.4939015319192
    CONST014 = 22.0740939791422
    CONST015 = 23.5310632462709
    CONST017 = 36.7901566319036
    CONST019 = 38.4260653723485
    CONST020 = 38.4260653723485
    CONST021 = 38.4260653723485
    CONST023 = -4.99169231699030
    CONST025 = 47.0621264925418
    CONST026 = 50.8329064189723
    CONST028 = 55.1852349478554
    CONST029 = 56.2781179722634
    CONST030 = 56.2781179722634
    CONST032 = 66.5558975598707
    CONST033 = 75.2994023880668
    CONST037 = 101.665812837945
    CONST038 = 110.370469895711
    CONST041 = 147.160626527614
    CONST042 = -1.66389743899677
    CONST043 = -9.37968632871057
    CONST044 = -1.66389743899677
    CONST045 = -220.740939791422
    CONST046 = -220.740939791422
    CONST047 = -1.60108605718119
    CONST048 = -187.593726574211
    CONST049 = -9.19753915797590
    CONST050 = -1.83950783159518
    CONST051 = -1.83950783159518
    CONST052 = -4.80325817154356
    CONST053 = -147.160626527614
    CONST054 = -140.695294930659
    CONST055 = -133.111795119741
    CONST056 = -125.499003980111
    CONST057 = -125.499003980111
    CONST058 = -99.8338463398060
    CONST059 = -87.7389315936062
    CONST060 = -76.8521307446970
    CONST061 = -66.5558975598707
    CONST062 = -62.7495019900557
    CONST063 = -52.6433589561637
    CONST064 = -44.1481879582843
    CONST065 = -44.3705983732471
    CONST066 = -40.6663251351779
    CONST067 = -40.6663251351779
    CONST068 = -8.31948719498384
    CONST069 = -37.6497011940334
    CONST070 = -33.2779487799353
    CONST071 = -25.4164532094862
    CONST072 = -25.4164532094862
    CONST073 = -17.5477863187212
    CONST074 = -11.7655316231354
    CONST075 = -11.0370469895711
    CONST076 = -9.19753915797590
    CONST077 = -8.47215106982872
    CONST078 = -4.80325817154356
    CONST079 = -2.50682661696018
    CONST080 = -1.60108605718119
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    # -------------------- kernel implementations
    Y00 = (
        CONST059 * VAR07 * VAR24
        - CONST063 * VAR05 * VAR26
        - CONST073 * VAR22 * x
        + CONST079 * VAR03
    )
    Y01 = y * (CONST029 * VAR23 * x + CONST030 * VAR05 * z + CONST048 * VAR07 * VAR25)
    Y02 = (
        CONST050 * VAR03
        + VAR05 * (CONST010 * VAR26 + CONST014 * VAR17)
        + VAR07 * (CONST045 * VAR17 * VAR26 - CONST076 * VAR24)
        + x * (CONST038 * VAR17 * VAR24 + CONST076 * VAR22)
    )
    Y03 = VAR16 * (CONST041 * VAR25 * x + CONST053 * VAR07 * z) + y * (
        -CONST064 * VAR05 * z + CONST064 * VAR23 * x
    )
    Y04 = (
        CONST042 * VAR03
        + VAR05 * (-CONST042 * VAR26 - CONST070 * VAR17)
        + VAR07 * (CONST061 * VAR17 * VAR26 + CONST065 * VAR15 - CONST068 * VAR24)
        + x * (-CONST023 * VAR22 - CONST055 * VAR15 * VAR26 + CONST058 * VAR17 * VAR24)
    )
    Y05 = (
        CONST015 * VAR05 * y * z
        + VAR07 * (CONST025 * VAR25 * y + CONST057 * VAR16 * z)
        + x * (CONST015 * VAR23 * y + CONST033 * VAR14 * z + CONST056 * VAR16 * VAR25)
    )
    Y06 = (
        CONST047 * VAR03
        + VAR05 * (CONST020 * VAR17 + CONST078 * VAR26)
        + VAR07 * (CONST052 * VAR24 + CONST060 * VAR15 - CONST060 * VAR17 * VAR26)
        + x
        * (
            CONST012 * VAR13
            + CONST019 * VAR17 * VAR24
            + CONST060 * VAR15 * VAR26
            + CONST080 * VAR22
        )
    )
    Y07 = (
        CONST002 * VAR12
        + VAR14 * (CONST066 * VAR08 + CONST067 * VAR26)
        + VAR16 * (CONST026 * VAR06 + CONST026 * VAR24 + CONST037 * VAR08 * VAR26)
        + y
        * (
            CONST071 * VAR06 * VAR26
            + CONST072 * VAR08 * VAR24
            + CONST077 * VAR04
            + CONST077 * VAR22
        )
    )
    Y08 = (
        CONST047 * VAR21
        + VAR23 * (CONST020 * VAR17 + CONST052 * VAR08)
        + VAR25 * (CONST052 * VAR06 - CONST060 * VAR08 * VAR17 + CONST060 * VAR15)
        + z
        * (
            CONST013 * VAR13
            + CONST021 * VAR06 * VAR17
            + CONST047 * VAR04
            + CONST060 * VAR08 * VAR15
        )
    )
    Y09 = (
        VAR14 * (CONST069 * VAR08 - CONST069 * VAR26)
        + VAR16 * (-CONST062 * VAR06 + CONST062 * VAR24)
        + y
        * (
            CONST008 * VAR08 * VAR24
            + CONST074 * VAR04
            + CONST074 * VAR06 * VAR26
            - CONST074 * VAR22
        )
    )
    Y10 = (
        -CONST042 * VAR21
        + VAR23 * (CONST044 * VAR08 + CONST070 * VAR17)
        + VAR25 * (CONST032 * VAR08 * VAR17 - CONST065 * VAR15 + CONST068 * VAR06)
        + z * (CONST023 * VAR04 + CONST055 * VAR08 * VAR15 - CONST058 * VAR06 * VAR17)
    )
    Y11 = VAR16 * (
        CONST017 * VAR06 + CONST017 * VAR24 + CONST046 * VAR08 * VAR26
    ) + y * (
        CONST028 * VAR06 * VAR26
        + CONST028 * VAR08 * VAR24
        + CONST075 * VAR04
        + CONST075 * VAR22
    )
    Y12 = (
        CONST051 * VAR21
        + VAR23 * (CONST010 * VAR08 + CONST014 * VAR17)
        + VAR25 * (CONST045 * VAR08 * VAR17 - CONST049 * VAR06)
        + z * (CONST038 * VAR06 * VAR17 + CONST049 * VAR04)
    )
    Y13 = y * (
        CONST043 * VAR04
        - CONST043 * VAR22
        - CONST054 * VAR06 * VAR26
        + CONST054 * VAR08 * VAR24
    )
    Y14 = (
        -CONST059 * VAR06 * VAR25
        + CONST063 * VAR08 * VAR23
        + CONST073 * VAR04 * z
        - CONST079 * VAR21
    )
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
    tl.store(
        output_ptr + output_row_offset + 9,
        Y09,
        mask=output_row_offset + 9 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 10,
        Y10,
        mask=output_row_offset + 10 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 11,
        Y11,
        mask=output_row_offset + 11 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 12,
        Y12,
        mask=output_row_offset + 12 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 13,
        Y13,
        mask=output_row_offset + 13 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 14,
        Y14,
        mask=output_row_offset + 14 < output_numel,
    )


@triton.jit
def seventh_order_bwd(
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
    g_9 = tl.load(
        sph_grad_ptr + output_row_offset + 9, mask=output_row_offset + 9 < output_numel
    )
    g_10 = tl.load(
        sph_grad_ptr + output_row_offset + 10,
        mask=output_row_offset + 10 < output_numel,
    )
    g_11 = tl.load(
        sph_grad_ptr + output_row_offset + 11,
        mask=output_row_offset + 11 < output_numel,
    )
    g_12 = tl.load(
        sph_grad_ptr + output_row_offset + 12,
        mask=output_row_offset + 12 < output_numel,
    )
    g_13 = tl.load(
        sph_grad_ptr + output_row_offset + 13,
        mask=output_row_offset + 13 < output_numel,
    )
    g_14 = tl.load(
        sph_grad_ptr + output_row_offset + 14,
        mask=output_row_offset + 14 < output_numel,
    )
    # -------------------- variable and constant definitions
    CONST000 = 1.66389743899677
    CONST001 = 3.00000000000000
    CONST003 = 5.00000000000000
    CONST004 = 3.32779487799353
    CONST009 = 11.7655316231354
    CONST012 = 16.5555704843566
    CONST014 = 20.4939015319192
    CONST016 = 22.0740939791422
    CONST018 = 23.5310632462709
    CONST019 = 20.4939015319192
    CONST020 = 27.1108834234519
    CONST022 = 33.1111409687132
    CONST024 = 36.7901566319036
    CONST025 = 36.7901566319036
    CONST026 = 38.4260653723485
    CONST027 = 38.4260653723485
    CONST029 = 38.4260653723485
    CONST030 = 44.1481879582843
    CONST032 = -4.99169231699030
    CONST037 = 47.0621264925417
    CONST039 = 56.2781179722634
    CONST044 = -441.481879582843
    CONST045 = -441.481879582843
    CONST048 = 76.8521307446970
    CONST049 = 76.8521307446970
    CONST050 = -8.47215106982872
    CONST054 = 110.370469895711
    CONST055 = 110.370469895711
    CONST056 = -399.335385359224
    CONST057 = 117.655316231354
    CONST058 = 122.963409191515
    CONST059 = 122.963409191515
    CONST061 = -376.497011940334
    CONST062 = -376.497011940334
    CONST064 = 141.186379477625
    CONST066 = 147.160626527614
    CONST067 = 153.704261489394
    CONST069 = -350.955726374425
    CONST072 = 203.331625675889
    CONST073 = 203.331625675889
    CONST074 = -307.408522978788
    CONST075 = -9.60651634308713
    CONST076 = -9.37968632871057
    CONST079 = -281.390589861317
    CONST080 = -1.66389743899677
    CONST081 = -266.223590239483
    CONST082 = -263.216794780819
    CONST084 = -263.216794780818
    CONST085 = -250.998007960223
    CONST089 = 281.390589861317
    CONST091 = -220.740939791422
    CONST092 = -220.740939791422
    CONST093 = -199.667692679612
    CONST094 = -1.60108605718119
    CONST095 = -187.593726574211
    CONST096 = -177.482393492989
    CONST097 = -9.60651634308712
    CONST098 = -9.19753915797590
    CONST100 = -153.704261489394
    CONST101 = -147.160626527614
    CONST102 = -140.695294930659
    CONST104 = -133.111795119741
    CONST105 = -133.111795119741
    CONST106 = -125.499003980111
    CONST107 = -125.499003980111
    CONST109 = -105.286717912327
    CONST110 = -101.665812837945
    CONST111 = -99.8338463398060
    CONST112 = -101.665812837945
    CONST113 = -4.80325817154356
    CONST114 = -81.3326502703558
    CONST115 = -81.3326502703557
    CONST116 = -76.8521307446970
    CONST117 = -75.2994023880668
    CONST119 = -70.5931897388126
    CONST121 = -66.2222819374265
    CONST122 = -66.5558975598707
    CONST123 = -66.5558975598707
    CONST124 = -62.7495019900557
    CONST125 = -56.2781179722634
    CONST126 = -55.1852349478554
    CONST127 = -55.1852349478554
    CONST128 = -50.8329064189723
    CONST129 = -50.8329064189723
    CONST130 = -562.781179722634
    CONST131 = -47.0621264925418
    CONST132 = -50.8329064189724
    CONST133 = -44.1481879582843
    CONST134 = -44.3705983732471
    CONST135 = -40.6663251351779
    CONST136 = -40.6663251351779
    CONST137 = -8.31948719498384
    CONST138 = -37.6497011940334
    CONST139 = -33.2779487799353
    CONST140 = -29.9501539019418
    CONST141 = -25.4164532094862
    CONST142 = -25.4164532094862
    CONST143 = -23.5310632462709
    CONST144 = -532.447180478965
    CONST145 = -19.2130326861743
    CONST146 = -17.5477863187212
    CONST147 = -12.8765548211663
    CONST148 = -11.6472820729774
    CONST149 = -11.2076024002683
    CONST150 = -9.19753915797590
    CONST151 = -11.0370469895711
    CONST152 = -11.7655316231354
    CONST153 = -12.8765548211663
    CONST154 = -4.80325817154356
    CONST155 = -3.32779487799353
    CONST156 = -1.60108605718119
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR16 = y * y * y
    VAR17 = y * y
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR15 = VAR17 * VAR17
    VAR25 = z * z * z
    VAR26 = z * z
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    VAR24 = VAR26 * VAR26
    # -------------------- kernel implementations
    g_x = (
        g_0
        * (
            CONST082 * VAR08 * VAR24
            - CONST084 * VAR06 * VAR26
            + CONST146 * VAR04
            - CONST146 * VAR22
        )
        + g_1 * y * (CONST039 * VAR23 + CONST089 * VAR06 * z + CONST130 * VAR08 * VAR25)
        + g_10
        * (
            CONST155 * VAR23 * x
            + VAR25 * (-CONST105 * VAR17 * x + CONST139 * VAR07)
            + z * (-CONST056 * VAR07 * VAR17 + CONST081 * VAR15 * x + CONST140 * VAR05)
        )
        + g_11
        * (
            VAR16 * (CONST044 * VAR26 * x - CONST101 * VAR07)
            + y * (CONST054 * VAR24 * x - CONST091 * VAR07 * VAR26 + CONST121 * VAR05)
        )
        + g_12
        * (
            CONST022 * VAR23 * x
            + VAR25 * (CONST024 * VAR07 + CONST045 * VAR17 * x)
            + z * (-CONST044 * VAR07 * VAR17 + CONST126 * VAR05)
        )
        + g_13
        * y
        * (CONST079 * VAR24 * x + CONST125 * VAR05 - CONST130 * VAR07 * VAR26)
        + g_14
        * (-CONST069 * VAR07 * VAR25 + CONST109 * VAR05 * z + CONST109 * VAR23 * x)
        + g_2
        * (
            CONST001 * VAR08 * (CONST091 * VAR17 * VAR26 - CONST150 * VAR24)
            + CONST003 * VAR06 * (CONST012 * VAR26 + CONST016 * VAR17)
            + CONST055 * VAR17 * VAR24
            + CONST147 * VAR04
            + CONST150 * VAR22
        )
        + g_3
        * (
            VAR16 * (CONST044 * VAR08 * z + CONST066 * VAR25)
            + y * (-CONST091 * VAR06 * z + CONST133 * VAR23)
        )
        + g_4
        * (
            CONST001
            * VAR08
            * (CONST122 * VAR17 * VAR26 + CONST134 * VAR15 - CONST137 * VAR24)
            + CONST003 * VAR06 * (CONST000 * VAR26 - CONST139 * VAR17)
            - CONST032 * VAR22
            - CONST105 * VAR15 * VAR26
            + CONST111 * VAR17 * VAR24
            + CONST148 * VAR04
        )
        + g_5
        * (
            CONST001 * VAR08 * (CONST106 * VAR16 * z - CONST131 * VAR25 * y)
            + CONST057 * VAR06 * y * z
            + CONST107 * VAR16 * VAR25
            - CONST117 * VAR14 * z
            - CONST143 * VAR23 * y
        )
        + g_6
        * (
            CONST001
            * VAR08
            * (CONST116 * VAR15 - CONST116 * VAR17 * VAR26 + CONST154 * VAR24)
            + CONST003 * VAR06 * (CONST026 * VAR17 + CONST113 * VAR26)
            + CONST014 * VAR13
            + CONST027 * VAR17 * VAR24
            + CONST116 * VAR15 * VAR26
            + CONST149 * VAR04
            + CONST156 * VAR22
        )
        + g_7
        * (
            CONST114 * VAR14 * x
            + VAR16 * (CONST072 * VAR07 + CONST073 * VAR26 * x)
            + y * (CONST110 * VAR07 * VAR26 + CONST128 * VAR05 + CONST129 * VAR24 * x)
        )
        + g_8
        * (
            CONST075 * VAR23 * x
            + VAR25 * (-CONST100 * VAR17 * x + CONST145 * VAR07)
            + z * (CONST067 * VAR07 * VAR17 + CONST097 * VAR05 + CONST100 * VAR15 * x)
        )
        + g_9
        * (
            -CONST085 * VAR07 * VAR16
            + CONST117 * VAR14 * x
            + y * (CONST018 * VAR24 * x + CONST119 * VAR05 + CONST131 * VAR07 * VAR26)
        )
    )
    g_y = (
        g_1 * (CONST039 * VAR23 * x + CONST095 * VAR07 * VAR25 - CONST125 * VAR05 * z)
        + g_10
        * (
            CONST123 * VAR23 * y
            + VAR25 * (-CONST096 * VAR16 - CONST105 * VAR08 * y)
            + z * (-CONST093 * VAR06 * y + CONST144 * VAR08 * VAR16)
        )
        + g_11
        * (
            CONST001
            * VAR17
            * (CONST025 * VAR06 + CONST025 * VAR24 + CONST092 * VAR08 * VAR26)
            - CONST126 * VAR06 * VAR26
            - CONST126 * VAR08 * VAR24
            + CONST151 * VAR04
            + CONST151 * VAR22
        )
        + g_12
        * (
            CONST030 * VAR23 * y
            + CONST045 * VAR08 * VAR25 * y
            - CONST092 * VAR06 * y * z
        )
        + g_13
        * (
            CONST076 * VAR04
            - CONST076 * VAR22
            - CONST102 * VAR06 * VAR26
            + CONST102 * VAR08 * VAR24
        )
        + g_2
        * (
            CONST030 * VAR05 * y
            + CONST045 * VAR07 * VAR26 * y
            - CONST092 * VAR24 * x * y
        )
        + g_3
        * (
            CONST001 * VAR17 * (CONST066 * VAR25 * x + CONST101 * VAR07 * z)
            - CONST133 * VAR05 * z
            + CONST133 * VAR23 * x
        )
        + g_4
        * (
            -CONST123 * VAR05 * y
            + VAR07 * (CONST096 * VAR16 + CONST104 * VAR26 * y)
            + x * (CONST093 * VAR24 * y - CONST144 * VAR16 * VAR26)
        )
        + g_5
        * (
            -CONST143 * VAR05 * z
            + VAR07 * (CONST062 * VAR17 * z - CONST131 * VAR25)
            + x * (CONST061 * VAR17 * VAR25 - CONST062 * VAR15 * z - CONST143 * VAR23)
        )
        + g_6
        * (
            CONST048 * VAR05 * y
            + VAR07 * (CONST074 * VAR16 - CONST100 * VAR26 * y)
            + x * (CONST058 * VAR14 + CONST074 * VAR16 * VAR26 - CONST116 * VAR24 * y)
        )
        + g_7
        * (
            CONST001
            * VAR17
            * (-CONST112 * VAR08 * VAR26 - CONST128 * VAR06 - CONST128 * VAR24)
            + CONST003 * VAR15 * (CONST135 * VAR08 + CONST136 * VAR26)
            + CONST020 * VAR13
            + CONST050 * VAR04
            + CONST050 * VAR22
            + CONST141 * VAR06 * VAR26
            + CONST142 * VAR08 * VAR24
        )
        + g_8
        * (
            CONST048 * VAR23 * y
            + VAR25 * (CONST074 * VAR16 - CONST100 * VAR08 * y)
            + z * (CONST049 * VAR06 * y + CONST059 * VAR14 + CONST074 * VAR08 * VAR16)
        )
        + g_9
        * (
            CONST001 * VAR17 * (-CONST124 * VAR06 + CONST124 * VAR24)
            + CONST003 * VAR15 * (CONST138 * VAR08 - CONST138 * VAR26)
            + CONST009 * VAR08 * VAR24
            + CONST152 * VAR04
            + CONST152 * VAR06 * VAR26
            - CONST152 * VAR22
        )
    )
    g_z = (
        g_0 * (CONST069 * VAR07 * VAR25 - CONST109 * VAR05 * z - CONST109 * VAR23 * x)
        + g_1
        * y
        * (-CONST079 * VAR24 * x - CONST125 * VAR05 + CONST130 * VAR07 * VAR26)
        + g_10
        * (
            CONST001
            * VAR26
            * (-CONST123 * VAR08 * VAR17 - CONST134 * VAR15 + CONST137 * VAR06)
            + CONST003 * VAR24 * (CONST080 * VAR08 + CONST139 * VAR17)
            + CONST032 * VAR04
            + CONST105 * VAR08 * VAR15
            - CONST111 * VAR06 * VAR17
            - CONST148 * VAR22
        )
        + g_11
        * (
            VAR16 * (CONST044 * VAR08 * z - CONST101 * VAR25)
            + y * (CONST054 * VAR06 * z - CONST091 * VAR08 * VAR25 + CONST121 * VAR23)
        )
        + g_12
        * (
            CONST001 * VAR26 * (CONST091 * VAR08 * VAR17 - CONST098 * VAR06)
            + CONST003 * VAR24 * (CONST012 * VAR08 + CONST016 * VAR17)
            + CONST055 * VAR06 * VAR17
            + CONST098 * VAR04
            + CONST153 * VAR22
        )
        + g_13
        * y
        * (-CONST079 * VAR06 * z - CONST125 * VAR23 + CONST130 * VAR08 * VAR25)
        + g_14
        * (
            -CONST082 * VAR06 * VAR26
            + CONST084 * VAR08 * VAR24
            + CONST146 * VAR04
            - CONST146 * VAR22
        )
        + g_2
        * (
            CONST022 * VAR05 * z
            + VAR07 * (CONST025 * VAR25 + CONST045 * VAR17 * z)
            + x * (-CONST044 * VAR17 * VAR25 + CONST127 * VAR23)
        )
        + g_3
        * (
            VAR16 * (-CONST045 * VAR26 * x + CONST101 * VAR07)
            + y * (CONST091 * VAR24 * x - CONST133 * VAR05)
        )
        + g_4
        * (
            CONST004 * VAR05 * z
            + VAR07 * (CONST104 * VAR17 * z - CONST139 * VAR25)
            + x * (CONST056 * VAR17 * VAR25 - CONST081 * VAR15 * z - CONST140 * VAR23)
        )
        + g_5
        * (
            -CONST143 * VAR05 * y
            + VAR07 * (CONST064 * VAR26 * y + CONST106 * VAR16)
            + x * (CONST057 * VAR24 * y + CONST061 * VAR16 * VAR26 - CONST117 * VAR14)
        )
        + g_6
        * (
            CONST097 * VAR05 * z
            + VAR07 * (-CONST100 * VAR17 * z + CONST145 * VAR25)
            + x * (CONST075 * VAR23 + CONST100 * VAR15 * z - CONST100 * VAR17 * VAR25)
        )
        + g_7
        * (
            CONST115 * VAR14 * z
            + VAR16 * (CONST072 * VAR25 + CONST073 * VAR08 * z)
            + y * (CONST112 * VAR08 * VAR25 + CONST128 * VAR23 + CONST132 * VAR06 * z)
        )
        + g_8
        * (
            CONST001
            * VAR26
            * (-CONST116 * VAR08 * VAR17 + CONST116 * VAR15 + CONST154 * VAR06)
            + CONST003 * VAR24 * (CONST026 * VAR17 + CONST154 * VAR08)
            + CONST019 * VAR13
            + CONST029 * VAR06 * VAR17
            + CONST094 * VAR04
            + CONST116 * VAR08 * VAR15
            + CONST149 * VAR22
        )
        + g_9
        * (
            CONST085 * VAR16 * VAR25
            - CONST117 * VAR14 * z
            + y * (CONST037 * VAR08 * VAR25 - CONST119 * VAR23 + CONST143 * VAR06 * z)
        )
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
