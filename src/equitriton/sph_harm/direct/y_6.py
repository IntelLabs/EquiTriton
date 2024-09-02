import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["SixthOrderSphericalHarmonic"]


class SixthOrderSphericalHarmonic(torch.autograd.Function):
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
                (*coords.shape[:-1], 13), dtype=coords.dtype, device=coords.device
            )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        sixth_order_fwd[num_blocks,](
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
        ctx, sph_grad_tensor: torch.Tensor, block_size: int = 64, col_offset: int = 0
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # call backward kernel
        sixth_order_bwd[num_blocks,](
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
    CONST002 = 3.26558761940328
    CONST003 = 3.26558761940328
    CONST004 = 6.53117523880657
    CONST006 = 8.38944649544891
    CONST007 = 9.79676285820985
    CONST008 = 10.3266947761614
    CONST009 = 3.60555127546399
    CONST010 = -1.78863600265677
    CONST011 = 14.5309475774982
    CONST012 = 8.94318001328386
    CONST013 = 16.5227116418583
    CONST014 = 16.5227116418583
    CONST015 = 17.8863600265677
    CONST017 = 20.6533895523229
    CONST018 = 20.2812259244849
    CONST019 = -107.318160159406
    CONST020 = 17.8863600265677
    CONST022 = 29.3902885746295
    CONST024 = 40.5624518489699
    CONST025 = 41.9472324772445
    CONST026 = -1.63279380970164
    CONST027 = -83.8944649544891
    CONST028 = -78.3741028656788
    CONST030 = -71.5454401062709
    CONST032 = -52.2494019104525
    CONST033 = -52.2494019104525
    CONST035 = -48.4364919249939
    CONST036 = -41.3067791046458
    CONST037 = -36.3273689437454
    CONST038 = -29.3902885746295
    CONST039 = -27.0416345659799
    CONST040 = -26.1247009552263
    CONST041 = -26.1247009552263
    CONST042 = -19.5935257164197
    CONST043 = -2.42182459624970
    CONST044 = -9.79676285820985
    CONST045 = -7.15454401062709
    CONST046 = -3.38020432074749
    CONST047 = -1.12673477358250
    VAR07 = x * x * x
    VAR08 = x * x
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR06 = VAR08 * VAR08
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
    Y00 = CONST011 * VAR05 * z + CONST011 * VAR23 * x + CONST035 * VAR07 * VAR25
    Y01 = y * (CONST006 * VAR05 + CONST025 * VAR24 * x + CONST027 * VAR07 * VAR26)
    Y02 = (
        -CONST045 * VAR05 * z
        + CONST045 * VAR23 * x
        + VAR17 * (CONST030 * VAR07 * z - CONST030 * VAR25 * x)
    )
    Y03 = VAR16 * (-CONST028 * VAR26 * x + CONST040 * VAR07) + y * (
        CONST007 * VAR05 + CONST038 * VAR24 * x + CONST042 * VAR07 * VAR26
    )
    Y04 = (
        CONST003 * VAR05 * z
        + VAR07 * (CONST004 * VAR25 + CONST033 * VAR17 * z)
        + x * (CONST002 * VAR23 - CONST032 * VAR15 * z + CONST032 * VAR17 * VAR25)
    )
    Y05 = (
        CONST008 * VAR05 * y
        + VAR07 * (CONST017 * VAR26 * y + CONST036 * VAR16)
        + x * (CONST008 * VAR24 * y + CONST013 * VAR14 + CONST036 * VAR16 * VAR26)
    )
    Y06 = (
        CONST009 * VAR13
        + CONST018 * VAR17 * VAR24
        + CONST039 * VAR15 * VAR26
        + CONST047 * VAR04
        + CONST047 * VAR22
        + VAR06 * (CONST018 * VAR17 + CONST046 * VAR26)
        + VAR08 * (CONST024 * VAR17 * VAR26 + CONST039 * VAR15 + CONST046 * VAR24)
    )
    Y07 = (
        CONST008 * VAR23 * y
        + VAR25 * (CONST017 * VAR08 * y + CONST036 * VAR16)
        + z * (CONST008 * VAR06 * y + CONST014 * VAR14 + CONST036 * VAR08 * VAR16)
    )
    Y08 = (
        CONST026 * VAR04
        - CONST026 * VAR22
        + CONST040 * VAR17 * VAR24
        - CONST041 * VAR15 * VAR26
        + VAR06 * (CONST026 * VAR26 - CONST041 * VAR17)
        + VAR08 * (-CONST026 * VAR24 + CONST041 * VAR15)
    )
    Y09 = VAR16 * (CONST028 * VAR08 * z - CONST041 * VAR25) + y * (
        CONST022 * VAR06 * z - CONST042 * VAR08 * VAR25 + CONST044 * VAR23
    )
    Y10 = (
        CONST010 * VAR04
        + CONST010 * VAR22
        + CONST020 * VAR17 * VAR24
        + VAR06 * (CONST012 * VAR26 + CONST015 * VAR17)
        + VAR08 * (CONST012 * VAR24 + CONST019 * VAR17 * VAR26)
    )
    Y11 = y * (CONST006 * VAR23 + CONST025 * VAR06 * z + CONST027 * VAR08 * VAR25)
    Y12 = (
        -CONST037 * VAR06 * VAR26
        + CONST037 * VAR08 * VAR24
        + CONST043 * VAR04
        - CONST043 * VAR22
    )
    # not the prettiest way to concatenate, but better than
    # messing with the linter
    tensors = [Y00, Y01, Y02, Y03, Y04, Y05, Y06, Y07, Y08, Y09, Y10, Y11, Y12]
    return torch.cat(tensors, dim=-1)


@triton.jit
def sixth_order_fwd(
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
    CONST002 = 3.26558761940328
    CONST003 = 3.26558761940328
    CONST004 = 6.53117523880657
    CONST006 = 8.38944649544891
    CONST007 = 9.79676285820985
    CONST008 = 10.3266947761614
    CONST009 = 3.60555127546399
    CONST010 = -1.78863600265677
    CONST011 = 14.5309475774982
    CONST012 = 8.94318001328386
    CONST013 = 16.5227116418583
    CONST014 = 16.5227116418583
    CONST015 = 17.8863600265677
    CONST017 = 20.6533895523229
    CONST018 = 20.2812259244849
    CONST019 = -107.318160159406
    CONST020 = 17.8863600265677
    CONST022 = 29.3902885746295
    CONST024 = 40.5624518489699
    CONST025 = 41.9472324772445
    CONST026 = -1.63279380970164
    CONST027 = -83.8944649544891
    CONST028 = -78.3741028656788
    CONST030 = -71.5454401062709
    CONST032 = -52.2494019104525
    CONST033 = -52.2494019104525
    CONST035 = -48.4364919249939
    CONST036 = -41.3067791046458
    CONST037 = -36.3273689437454
    CONST038 = -29.3902885746295
    CONST039 = -27.0416345659799
    CONST040 = -26.1247009552263
    CONST041 = -26.1247009552263
    CONST042 = -19.5935257164197
    CONST043 = -2.42182459624970
    CONST044 = -9.79676285820985
    CONST045 = -7.15454401062709
    CONST046 = -3.38020432074749
    CONST047 = -1.12673477358250
    VAR07 = x * x * x
    VAR08 = x * x
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR06 = VAR08 * VAR08
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
    Y00 = CONST011 * VAR05 * z + CONST011 * VAR23 * x + CONST035 * VAR07 * VAR25
    Y01 = y * (CONST006 * VAR05 + CONST025 * VAR24 * x + CONST027 * VAR07 * VAR26)
    Y02 = (
        -CONST045 * VAR05 * z
        + CONST045 * VAR23 * x
        + VAR17 * (CONST030 * VAR07 * z - CONST030 * VAR25 * x)
    )
    Y03 = VAR16 * (-CONST028 * VAR26 * x + CONST040 * VAR07) + y * (
        CONST007 * VAR05 + CONST038 * VAR24 * x + CONST042 * VAR07 * VAR26
    )
    Y04 = (
        CONST003 * VAR05 * z
        + VAR07 * (CONST004 * VAR25 + CONST033 * VAR17 * z)
        + x * (CONST002 * VAR23 - CONST032 * VAR15 * z + CONST032 * VAR17 * VAR25)
    )
    Y05 = (
        CONST008 * VAR05 * y
        + VAR07 * (CONST017 * VAR26 * y + CONST036 * VAR16)
        + x * (CONST008 * VAR24 * y + CONST013 * VAR14 + CONST036 * VAR16 * VAR26)
    )
    Y06 = (
        CONST009 * VAR13
        + CONST018 * VAR17 * VAR24
        + CONST039 * VAR15 * VAR26
        + CONST047 * VAR04
        + CONST047 * VAR22
        + VAR06 * (CONST018 * VAR17 + CONST046 * VAR26)
        + VAR08 * (CONST024 * VAR17 * VAR26 + CONST039 * VAR15 + CONST046 * VAR24)
    )
    Y07 = (
        CONST008 * VAR23 * y
        + VAR25 * (CONST017 * VAR08 * y + CONST036 * VAR16)
        + z * (CONST008 * VAR06 * y + CONST014 * VAR14 + CONST036 * VAR08 * VAR16)
    )
    Y08 = (
        CONST026 * VAR04
        - CONST026 * VAR22
        + CONST040 * VAR17 * VAR24
        - CONST041 * VAR15 * VAR26
        + VAR06 * (CONST026 * VAR26 - CONST041 * VAR17)
        + VAR08 * (-CONST026 * VAR24 + CONST041 * VAR15)
    )
    Y09 = VAR16 * (CONST028 * VAR08 * z - CONST041 * VAR25) + y * (
        CONST022 * VAR06 * z - CONST042 * VAR08 * VAR25 + CONST044 * VAR23
    )
    Y10 = (
        CONST010 * VAR04
        + CONST010 * VAR22
        + CONST020 * VAR17 * VAR24
        + VAR06 * (CONST012 * VAR26 + CONST015 * VAR17)
        + VAR08 * (CONST012 * VAR24 + CONST019 * VAR17 * VAR26)
    )
    Y11 = y * (CONST006 * VAR23 + CONST025 * VAR06 * z + CONST027 * VAR08 * VAR25)
    Y12 = (
        -CONST037 * VAR06 * VAR26
        + CONST037 * VAR08 * VAR24
        + CONST043 * VAR04
        - CONST043 * VAR22
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


@triton.jit
def sixth_order_bwd(
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
    # -------------------- variable and constant definitions
    CONST000 = 2.00000000000000
    CONST002 = 4.00000000000000
    CONST003 = 3.00000000000000
    CONST004 = 6.53117523880657
    CONST006 = 8.94318001328386
    CONST007 = 8.38944649544891
    CONST008 = 10.3266947761614
    CONST009 = 9.79676285820985
    CONST013 = 16.3279380970164
    CONST014 = 17.8863600265677
    CONST015 = 16.5227116418583
    CONST016 = 20.6533895523229
    CONST017 = 20.2812259244849
    CONST018 = 21.6333076527839
    CONST020 = 17.8863600265677
    CONST022 = 29.3902885746295
    CONST024 = 35.7727200531355
    CONST026 = 40.5624518489699
    CONST028 = 41.9472324772445
    CONST029 = 48.9838142910493
    CONST030 = 51.6334738808072
    CONST035 = 71.5454401062709
    CONST037 = 81.1249036979398
    CONST039 = 82.6135582092915
    CONST040 = -3.26558761940328
    CONST042 = 117.561154298518
    CONST046 = 208.997607641810
    CONST048 = -251.683394863467
    CONST049 = -214.636320318813
    CONST050 = -214.636320318813
    CONST051 = 16.5227116418583
    CONST052 = -167.788929908978
    CONST053 = -156.748205731358
    CONST054 = -145.309475774982
    CONST055 = -123.920337313937
    CONST056 = -117.561154298518
    CONST057 = 3.26558761940328
    CONST058 = -108.166538263920
    CONST059 = -107.318160159406
    CONST060 = -104.498803820905
    CONST061 = -104.498803820905
    CONST062 = -83.8944649544891
    CONST063 = -82.6135582092915
    CONST064 = -78.3741028656788
    CONST065 = -72.6547378874909
    CONST066 = -71.5454401062709
    CONST067 = -58.7805771492591
    CONST068 = -54.0832691319598
    CONST069 = -52.2494019104525
    CONST070 = -52.2494019104525
    CONST071 = -48.9838142910492
    CONST072 = -41.3067791046458
    CONST073 = -39.1870514328394
    CONST074 = -35.7727200531355
    CONST075 = -29.3902885746295
    CONST076 = -27.0416345659799
    CONST077 = -26.1247009552263
    CONST078 = -26.1247009552263
    CONST079 = -19.5935257164197
    CONST080 = -14.5309475774982
    CONST081 = -13.5208172829900
    CONST082 = -10.7318160159406
    CONST083 = -9.79676285820985
    CONST084 = -7.15454401062709
    CONST085 = -6.76040864149498
    CONST086 = -3.38020432074749
    CONST087 = -1.63279380970164
    VAR07 = x * x * x
    VAR08 = x * x
    VAR05 = VAR07 * VAR08
    VAR06 = VAR08 * VAR08
    VAR16 = y * y * y
    VAR17 = y * y
    VAR14 = VAR16 * VAR17
    VAR15 = VAR17 * VAR17
    VAR25 = z * z * z
    VAR26 = z * z
    VAR23 = VAR25 * VAR26
    VAR24 = VAR26 * VAR26
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
        g_0 * (CONST054 * VAR08 * VAR25 - CONST065 * VAR06 * z - CONST080 * VAR23)
        + g_1 * y * (CONST028 * VAR06 + CONST028 * VAR24 + CONST048 * VAR08 * VAR26)
        + g_10
        * (
            CONST000 * x * (CONST006 * VAR24 + CONST059 * VAR17 * VAR26)
            + CONST002 * VAR07 * (CONST006 * VAR26 + CONST014 * VAR17)
            + CONST082 * VAR05
        )
        + g_11 * y * (-CONST052 * VAR07 * z + CONST052 * VAR25 * x)
        + g_12 * (-CONST054 * VAR07 * VAR26 + CONST065 * VAR24 * x + CONST080 * VAR05)
        + g_2
        * (
            -CONST074 * VAR06 * z
            + CONST084 * VAR23
            + VAR17 * (CONST049 * VAR08 * z - CONST066 * VAR25)
        )
        + g_3
        * (
            VAR16 * (CONST064 * VAR08 - CONST064 * VAR26)
            + y * (CONST029 * VAR06 + CONST067 * VAR08 * VAR26 + CONST075 * VAR24)
        )
        + g_4
        * (
            CONST003 * VAR08 * (CONST004 * VAR25 + CONST069 * VAR17 * z)
            + CONST013 * VAR06 * z
            - CONST040 * VAR23
            - CONST070 * VAR15 * z
            + CONST070 * VAR17 * VAR25
        )
        + g_5
        * (
            CONST003 * VAR08 * (CONST016 * VAR26 * y + CONST072 * VAR16)
            + CONST008 * VAR24 * y
            + CONST015 * VAR14
            + CONST030 * VAR06 * y
            + CONST072 * VAR16 * VAR26
        )
        + g_6
        * (
            CONST000
            * x
            * (CONST026 * VAR17 * VAR26 + CONST076 * VAR15 + CONST086 * VAR24)
            + CONST002 * VAR07 * (CONST017 * VAR17 + CONST086 * VAR26)
            + CONST085 * VAR05
        )
        + g_7
        * (
            -CONST072 * VAR25 * x * y
            + z * (CONST063 * VAR16 * x - CONST072 * VAR07 * y)
        )
        + g_8
        * (
            CONST000 * x * (CONST077 * VAR15 - CONST087 * VAR24)
            + CONST002 * VAR07 * (-CONST077 * VAR17 + CONST087 * VAR26)
            + CONST083 * VAR05
        )
        + g_9
        * (CONST053 * VAR16 * x * z + y * (CONST042 * VAR07 * z - CONST073 * VAR25 * x))
    )
    g_y += (
        CONST000 * g_2 * y * (CONST066 * VAR07 * z - CONST066 * VAR25 * x)
        + g_1 * (CONST007 * VAR05 + CONST028 * VAR24 * x + CONST062 * VAR07 * VAR26)
        + g_10
        * (CONST024 * VAR06 * y + CONST050 * VAR08 * VAR26 * y - CONST074 * VAR24 * y)
        + g_11 * (CONST007 * VAR23 + CONST028 * VAR06 * z + CONST062 * VAR08 * VAR25)
        + g_3
        * (
            CONST003 * VAR17 * (-CONST064 * VAR26 * x + CONST078 * VAR07)
            + CONST009 * VAR05
            + CONST075 * VAR24 * x
            + CONST079 * VAR07 * VAR26
        )
        + g_4
        * (CONST061 * VAR07 * y * z + x * (CONST046 * VAR16 * z + CONST060 * VAR25 * y))
        + g_5
        * (
            CONST008 * VAR05
            + VAR07 * (CONST016 * VAR26 + CONST055 * VAR17)
            + x * (CONST008 * VAR24 + CONST055 * VAR17 * VAR26 - CONST063 * VAR15)
        )
        + g_6
        * (
            CONST018 * VAR14
            + CONST026 * VAR06 * y
            + CONST026 * VAR24 * y
            + CONST058 * VAR16 * VAR26
            + VAR08 * (CONST037 * VAR26 * y + CONST058 * VAR16)
        )
        + g_7
        * (
            CONST008 * VAR23
            + VAR25 * (CONST016 * VAR08 + CONST055 * VAR17)
            + z * (CONST008 * VAR06 + CONST039 * VAR15 + CONST055 * VAR08 * VAR17)
        )
        + g_8
        * (
            CONST060 * VAR08 * VAR16
            - CONST060 * VAR16 * VAR26
            + CONST069 * VAR24 * y
            - CONST070 * VAR06 * y
        )
        + g_9
        * (
            CONST003 * VAR17 * (CONST064 * VAR08 * z - CONST077 * VAR25)
            + CONST022 * VAR06 * z
            - CONST079 * VAR08 * VAR25
            + CONST083 * VAR23
        )
    )
    g_z += (
        g_0 * (CONST054 * VAR07 * VAR26 - CONST065 * VAR24 * x - CONST080 * VAR05)
        + g_1 * y * (CONST052 * VAR07 * z - CONST052 * VAR25 * x)
        + g_10
        * (
            CONST020 * VAR06 * z
            + CONST035 * VAR17 * VAR25
            + CONST082 * VAR23
            + VAR08 * (CONST050 * VAR17 * z - CONST074 * VAR25)
        )
        + g_11 * y * (CONST028 * VAR06 + CONST028 * VAR24 + CONST048 * VAR08 * VAR26)
        + g_12 * (CONST054 * VAR08 * VAR25 - CONST065 * VAR06 * z - CONST080 * VAR23)
        + g_2
        * (
            CONST074 * VAR24 * x
            - CONST084 * VAR05
            + VAR17 * (-CONST049 * VAR26 * x + CONST066 * VAR07)
        )
        + g_3
        * (
            -CONST053 * VAR16 * x * z
            + y * (CONST056 * VAR25 * x + CONST073 * VAR07 * z)
        )
        + g_4
        * (
            CONST057 * VAR05
            + VAR07 * (CONST069 * VAR17 - CONST079 * VAR26)
            + x * (CONST013 * VAR24 + CONST053 * VAR17 * VAR26 - CONST070 * VAR15)
        )
        + g_5
        * (
            -CONST072 * VAR07 * y * z
            + x * (CONST063 * VAR16 * z - CONST072 * VAR25 * y)
        )
        + g_6
        * (
            CONST037 * VAR17 * VAR25
            + CONST068 * VAR15 * z
            + CONST085 * VAR06 * z
            + CONST085 * VAR23
            + VAR08 * (CONST037 * VAR17 * z + CONST081 * VAR25)
        )
        + g_7
        * (
            CONST003 * VAR26 * (CONST016 * VAR08 * y + CONST072 * VAR16)
            + CONST008 * VAR06 * y
            + CONST030 * VAR24 * y
            + CONST051 * VAR14
            + CONST072 * VAR08 * VAR16
        )
        + g_8
        * (
            CONST004 * VAR08 * VAR25
            + CONST040 * VAR06 * z
            + CONST061 * VAR17 * VAR25
            - CONST070 * VAR15 * z
            - CONST083 * VAR23
        )
        + g_9
        * (
            VAR16 * (CONST064 * VAR08 - CONST064 * VAR26)
            + y * (CONST022 * VAR06 - CONST067 * VAR08 * VAR26 + CONST071 * VAR24)
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
