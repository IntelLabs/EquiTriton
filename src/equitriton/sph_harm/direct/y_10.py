import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["TenthOrderSphericalHarmonic"]


class TenthOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
    ):
        output_tensor = torch.empty(
            (*coords.shape[:-1], 21), dtype=coords.dtype, device=coords.device
        )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        tenth_order_fwd[num_blocks,](
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
        tenth_order_bwd[num_blocks,](
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
    CONST001 = 1.75869118663323
    CONST002 = -1021.92317475320
    CONST004 = 4.58257569495584
    CONST005 = 6.63243980843400
    CONST006 = 4.82870805793735
    CONST007 = 4.97432985632550
    CONST008 = 1545.18657853995
    CONST009 = 10.5521471197994
    CONST010 = 12.1657520803952
    CONST011 = 13.2648796168680
    CONST013 = 15.7883647328499
    CONST014 = 15.7302121789667
    CONST015 = 16.4144510752435
    CONST016 = 12.8765548211663
    CONST017 = 19.3148322317494
    CONST018 = 16.7271353825295
    CONST019 = 22.8629854262320
    CONST020 = 535.268332240943
    CONST021 = 23.2135393295190
    CONST022 = 24.6216766128653
    CONST023 = 27.2034486491732
    CONST024 = 541.428124558099
    CONST025 = -994.666978169547
    CONST026 = 33.9852909359329
    CONST027 = 33.9852909359329
    CONST028 = 35.5238206489124
    CONST029 = -984.867064514610
    CONST030 = -4.82870805793735
    CONST031 = 1070.53666448189
    CONST032 = -463.555973561985
    CONST034 = 53.2857309733686
    CONST035 = 53.2857309733686
    CONST036 = 56.3871618715269
    CONST037 = 56.3871618715269
    CONST039 = -1989.33395633909
    CONST041 = -450.224943778107
    CONST042 = 66.9085415301178
    CONST043 = 69.6406179885570
    CONST044 = 69.6406179885570
    CONST045 = -437.967074894228
    CONST046 = 77.2593289269976
    CONST047 = 78.6510608948335
    CONST049 = -1969.73412902922
    CONST050 = 77.3468749368712
    CONST051 = 1624.28437367430
    CONST054 = 94.7301883970997
    CONST056 = 100.362812295177
    CONST057 = -412.049754277320
    CONST058 = 101.517773354644
    CONST059 = -5.63871618715269
    CONST060 = -406.071093418574
    CONST061 = 109.491768723557
    CONST062 = -393.946825805844
    CONST063 = -902.194589944431
    CONST065 = -386.296644634988
    CONST066 = -386.296644634988
    CONST070 = 4.97432985632550
    CONST071 = 150.074981259369
    CONST074 = 685.526905959165
    CONST075 = -337.668707833581
    CONST076 = -337.668707833581
    CONST077 = 176.178376404427
    CONST078 = 176.592751833137
    CONST079 = 185.708314636152
    CONST080 = -326.441383790078
    CONST081 = -1.60956935264578
    CONST082 = -1.97354559160624
    CONST083 = 196.973412902922
    CONST085 = -824.099508554641
    CONST087 = -1.97354559160624
    CONST088 = -305.867618423396
    CONST089 = -305.867618423396
    CONST090 = 721.755671955545
    CONST091 = -305.867618423396
    CONST092 = -300.731529981477
    CONST093 = -300.731529981477
    CONST094 = -1.75869118663323
    CONST095 = -290.050781013267
    CONST097 = 225.548647486108
    CONST098 = 225.548647486108
    CONST099 = -284.190565191299
    CONST101 = -278.562471954228
    CONST102 = -278.562471954228
    CONST103 = -787.893651611688
    CONST104 = -787.893651611688
    CONST105 = 772.593289269975
    CONST106 = 787.893651611688
    CONST107 = 787.893651611688
    CONST108 = 278.562471954228
    CONST109 = -742.833258544608
    CONST110 = -1.65810995210850
    CONST112 = -1761.78376404427
    CONST113 = -223.028471767059
    CONST114 = -734.076568351780
    CONST116 = -220.222970505534
    CONST117 = 1321.33782303320
    CONST118 = 1321.33782303320
    CONST119 = -203.035546709287
    CONST120 = -1.65810995210850
    CONST121 = -196.973412902922
    CONST122 = -196.973412902922
    CONST123 = -696.406179885570
    CONST125 = 338.322971229162
    CONST126 = -1181.84047741753
    CONST127 = -669.085415301178
    CONST128 = -669.085415301178
    CONST129 = -154.518657853995
    CONST130 = -154.518657853995
    CONST131 = 360.877835977772
    CONST132 = -150.074981259369
    CONST133 = -2707.14062279049
    CONST134 = -146.815313670356
    CONST135 = 880.891882022136
    CONST136 = 1392.81235977114
    CONST137 = 1392.81235977114
    CONST138 = -131.315608601948
    CONST139 = -131.315608601948
    CONST141 = -125.841697431734
    CONST142 = -125.841697431734
    CONST143 = -122.415518921279
    CONST145 = 406.071093418574
    CONST146 = -103.107953136506
    CONST147 = -103.107953136506
    CONST148 = -101.517773354644
    CONST149 = -98.4867064514610
    CONST150 = 412.049754277320
    CONST151 = -94.7301883970997
    CONST152 = -1114.24988781691
    CONST153 = -88.2963759165686
    CONST154 = -1624.28437367430
    CONST155 = -82.8889148474622
    CONST156 = -82.8889148474622
    CONST158 = -590.920238708766
    CONST159 = -77.3468749368713
    CONST160 = -77.2593289269975
    CONST161 = 2486.66744542387
    CONST162 = -2626.31217203896
    CONST165 = -571.272421632637
    CONST166 = -56.2781179722634
    CONST167 = -49.2433532257305
    CONST168 = -49.2433532257305
    CONST169 = 984.867064514610
    CONST170 = -541.428124558099
    CONST171 = -24.6216766128653
    CONST172 = -22.8629854262320
    CONST173 = -16.4144510752435
    CONST174 = -15.7883647328499
    CONST175 = -14.0695294930659
    CONST176 = -13.2648796168680
    CONST177 = -11.2774323743054
    CONST178 = -14.5025390506634
    CONST179 = -6.63243980843400
    CONST180 = -5.63871618715269
    CONST181 = 1532.88476212980
    CONST182 = -3.21913870529156
    CONST183 = -2.72034486491732
    CONST184 = -1.12774323743054
    # ordering is really messy because I've refactored
    # the higher powers in terms of the lower ones
    VAR05 = x * x * x * x * x
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR00 = VAR05 * VAR05
    VAR01 = VAR05 * VAR06
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR14 = y * y * y * y * y
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR09 = VAR14 * VAR14
    VAR10 = VAR14 * VAR15
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR23 = z * z * z * z * z
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR18 = VAR23 * VAR23
    VAR19 = VAR23 * VAR24
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    # -------------------- kernel implementations
    Y00 = (
        CONST023 * VAR01 * z
        + CONST023 * VAR19 * x
        + CONST074 * VAR05 * VAR23
        + CONST080 * VAR03 * VAR25
        + CONST080 * VAR07 * VAR21
    )
    Y01 = y * (
        CONST002 * VAR07 * VAR22
        + CONST010 * VAR01
        + CONST045 * VAR03 * VAR26
        + CONST061 * VAR20 * x
        + CONST181 * VAR05 * VAR24
    )
    Y02 = (
        CONST013 * VAR01 * z
        + CONST054 * VAR07 * VAR21
        + CONST151 * VAR03 * VAR25
        + CONST174 * VAR19 * x
        + VAR17
        * (
            -CONST039 * VAR05 * VAR25
            + CONST039 * VAR07 * VAR23
            + CONST099 * VAR03 * z
            - CONST099 * VAR21 * x
        )
    )
    Y03 = VAR16 * (
        CONST024 * VAR22 * x
        + CONST051 * VAR05 * VAR26
        + CONST133 * VAR07 * VAR24
        + CONST159 * VAR03
    ) + y * (
        CONST095 * VAR03 * VAR26
        - CONST119 * VAR05 * VAR24
        + CONST145 * VAR07 * VAR22
        + CONST148 * VAR20 * x
        - CONST178 * VAR01
    )
    Y04 = (
        CONST009 * VAR01 * z
        + VAR03 * (CONST076 * VAR17 * z + CONST175 * VAR25)
        + VAR05 * (CONST106 * VAR15 * z + CONST107 * VAR17 * VAR25 + CONST167 * VAR23)
        + VAR07
        * (CONST106 * VAR17 * VAR23 + CONST162 * VAR15 * VAR25 + CONST175 * VAR21)
        + x * (CONST009 * VAR19 + CONST075 * VAR17 * VAR21 + CONST106 * VAR15 * VAR23)
    )
    Y05 = (
        VAR14 * (CONST077 * VAR05 + CONST112 * VAR07 * VAR26 + CONST135 * VAR24 * x)
        + VAR16
        * (
            -CONST114 * VAR07 * VAR24
            + CONST114 * VAR22 * x
            + CONST117 * VAR05 * VAR26
            + CONST134 * VAR03
        )
        + y
        * (
            CONST014 * VAR01
            + CONST047 * VAR20 * x
            + CONST116 * VAR05 * VAR24
            + CONST141 * VAR03 * VAR26
        )
    )
    Y06 = (
        CONST005 * VAR01 * z
        + VAR03 * (CONST011 * VAR25 + CONST102 * VAR17 * z)
        + VAR05 * (CONST101 * VAR17 * VAR25 - CONST152 * VAR15 * z)
        + VAR07 * (CONST108 * VAR17 * VAR23 + CONST109 * VAR13 * z + CONST176 * VAR21)
        + x
        * (
            CONST108 * VAR17 * VAR21
            - CONST109 * VAR13 * VAR25
            + CONST152 * VAR15 * VAR23
            + CONST179 * VAR19
        )
    )
    Y07 = (
        VAR12 * (-CONST041 * VAR26 * x + CONST132 * VAR07)
        + VAR14 * (-CONST062 * VAR05 + CONST103 * VAR07 * VAR26 + CONST126 * VAR24 * x)
        + VAR16
        * (
            CONST083 * VAR05 * VAR26
            + CONST121 * VAR03
            - CONST158 * VAR22 * x
            + CONST169 * VAR07 * VAR24
        )
        + y
        * (
            CONST015 * VAR01
            + CONST138 * VAR07 * VAR22
            + CONST149 * VAR05 * VAR24
            + CONST168 * VAR20 * x
        )
    )
    Y08 = (
        -CONST182 * VAR01 * z
        + VAR03 * (CONST016 * VAR25 + CONST129 * VAR17 * z)
        + VAR05 * (CONST017 * VAR23 + CONST032 * VAR17 * VAR25 + CONST105 * VAR15 * z)
        + VAR07
        * (
            CONST008 * VAR15 * VAR25
            + CONST016 * VAR21
            + CONST032 * VAR17 * VAR23
            + CONST085 * VAR13 * z
        )
        + x
        * (
            CONST078 * VAR11 * z
            + CONST085 * VAR13 * VAR25
            + CONST105 * VAR15 * VAR23
            + CONST129 * VAR17 * VAR21
            - CONST182 * VAR19
        )
    )
    Y09 = (
        CONST018 * VAR01 * y
        + VAR03 * (CONST042 * VAR26 * y + CONST113 * VAR16)
        + VAR05 * (CONST020 * VAR14 + CONST056 * VAR24 * y + CONST128 * VAR16 * VAR26)
        + VAR07
        * (
            CONST031 * VAR14 * VAR26
            + CONST042 * VAR22 * y
            + CONST088 * VAR12
            + CONST127 * VAR16 * VAR24
        )
        + x
        * (
            CONST018 * VAR20 * y
            + CONST020 * VAR14 * VAR24
            + CONST026 * VAR10
            + CONST088 * VAR12 * VAR26
            + CONST113 * VAR16 * VAR22
        )
    )
    Y10 = (
        CONST004 * VAR09
        + CONST037 * VAR17 * VAR20
        + CONST093 * VAR15 * VAR22
        + CONST131 * VAR13 * VAR24
        + CONST147 * VAR11 * VAR26
        + CONST184 * VAR00
        + CONST184 * VAR18
        + VAR02 * (CONST036 * VAR17 + CONST059 * VAR26)
        + VAR04 * (CONST092 * VAR15 + CONST098 * VAR17 * VAR26 + CONST177 * VAR24)
        + VAR06
        * (
            CONST063 * VAR15 * VAR26
            + CONST125 * VAR17 * VAR24
            + CONST131 * VAR13
            + CONST177 * VAR22
        )
        + VAR08
        * (
            CONST063 * VAR15 * VAR24
            + CONST090 * VAR13 * VAR26
            + CONST097 * VAR17 * VAR22
            + CONST146 * VAR11
            + CONST180 * VAR20
        )
    )
    Y11 = (
        CONST018 * VAR19 * y
        + VAR21 * (CONST042 * VAR08 * y + CONST113 * VAR16)
        + VAR23 * (CONST020 * VAR14 + CONST056 * VAR06 * y + CONST128 * VAR08 * VAR16)
        + VAR25
        * (
            CONST031 * VAR08 * VAR14
            + CONST042 * VAR04 * y
            + CONST091 * VAR12
            + CONST127 * VAR06 * VAR16
        )
        + z
        * (
            CONST018 * VAR02 * y
            + CONST020 * VAR06 * VAR14
            + CONST027 * VAR10
            + CONST089 * VAR08 * VAR12
            + CONST113 * VAR04 * VAR16
        )
    )
    Y12 = (
        CONST057 * VAR13 * VAR24
        - CONST066 * VAR15 * VAR22
        + CONST081 * VAR00
        - CONST081 * VAR18
        - CONST153 * VAR11 * VAR26
        + CONST160 * VAR17 * VAR20
        + VAR02 * (CONST030 * VAR26 + CONST046 * VAR17)
        + VAR04 * (CONST066 * VAR15 - CONST129 * VAR17 * VAR26 + CONST182 * VAR24)
        + VAR06 * (CONST065 * VAR15 * VAR26 + CONST150 * VAR13 - CONST182 * VAR22)
        + VAR08
        * (
            CONST006 * VAR20
            - CONST066 * VAR15 * VAR24
            + CONST130 * VAR17 * VAR22
            + CONST153 * VAR11
        )
    )
    Y13 = (
        VAR12 * (CONST041 * VAR08 * z + CONST071 * VAR25)
        + VAR14 * (CONST062 * VAR23 + CONST107 * VAR08 * VAR25 - CONST126 * VAR06 * z)
        + VAR16
        * (
            CONST029 * VAR06 * VAR25
            - CONST121 * VAR21
            + CONST122 * VAR08 * VAR23
            + CONST158 * VAR04 * z
        )
        + y
        * (
            -CONST138 * VAR04 * VAR25
            - CONST149 * VAR06 * VAR23
            - CONST168 * VAR02 * z
            + CONST173 * VAR19
        )
    )
    Y14 = (
        CONST044 * VAR17 * VAR20
        + CONST079 * VAR13 * VAR24
        + CONST101 * VAR15 * VAR22
        + CONST110 * VAR00
        + CONST120 * VAR18
        + VAR02 * (CONST043 * VAR17 + CONST070 * VAR26)
        + VAR04 * (CONST021 * VAR24 + CONST101 * VAR15 + CONST101 * VAR17 * VAR26)
        + VAR06
        * (
            CONST021 * VAR22
            + CONST079 * VAR13
            + CONST123 * VAR17 * VAR24
            + CONST137 * VAR15 * VAR26
        )
        + VAR08
        * (
            CONST007 * VAR20
            + CONST101 * VAR17 * VAR22
            + CONST136 * VAR15 * VAR24
            + CONST152 * VAR13 * VAR26
        )
    )
    Y15 = (
        VAR14 * (CONST077 * VAR23 + CONST112 * VAR08 * VAR25 + CONST135 * VAR06 * z)
        + VAR16
        * (
            CONST114 * VAR04 * z
            - CONST114 * VAR06 * VAR25
            + CONST118 * VAR08 * VAR23
            + CONST134 * VAR21
        )
        + y
        * (
            CONST014 * VAR19
            + CONST047 * VAR02 * z
            + CONST116 * VAR06 * VAR23
            + CONST142 * VAR08 * VAR21
        )
    )
    Y16 = (
        CONST001 * VAR18
        + CONST094 * VAR00
        - CONST139 * VAR15 * VAR22
        + CONST166 * VAR17 * VAR20
        + VAR02 * (CONST019 * VAR26 - CONST166 * VAR17)
        + VAR04 * (CONST022 * VAR24 + CONST104 * VAR17 * VAR26 + CONST139 * VAR15)
        + VAR06 * (-CONST049 * VAR15 * VAR26 + CONST171 * VAR22)
        + VAR08
        * (CONST049 * VAR15 * VAR24 + CONST106 * VAR17 * VAR22 + CONST172 * VAR20)
    )
    Y17 = VAR16 * (
        CONST050 * VAR21
        - CONST133 * VAR06 * VAR25
        + CONST154 * VAR08 * VAR23
        + CONST170 * VAR04 * z
    ) + y * (
        CONST058 * VAR02 * z
        + CONST060 * VAR04 * VAR25
        - CONST095 * VAR08 * VAR21
        + CONST119 * VAR06 * VAR23
        + CONST178 * VAR19
    )
    Y18 = (
        CONST034 * VAR02 * VAR26
        + CONST035 * VAR08 * VAR20
        + CONST082 * VAR00
        + CONST087 * VAR18
        + CONST155 * VAR04 * VAR24
        + CONST156 * VAR06 * VAR22
        + VAR17
        * (
            CONST025 * VAR04 * VAR26
            + CONST025 * VAR08 * VAR22
            + CONST028 * VAR02
            + CONST028 * VAR20
            + CONST161 * VAR06 * VAR24
        )
    )
    Y19 = y * (
        CONST002 * VAR04 * VAR25
        + CONST010 * VAR19
        + CONST045 * VAR08 * VAR21
        + CONST061 * VAR02 * z
        + CONST181 * VAR06 * VAR23
    )
    Y20 = (
        -CONST143 * VAR02 * VAR26
        + CONST143 * VAR08 * VAR20
        + CONST165 * VAR04 * VAR24
        - CONST165 * VAR06 * VAR22
        + CONST183 * VAR00
        - CONST183 * VAR18
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
        Y15,
        Y16,
        Y17,
        Y18,
        Y19,
        Y20,
    ]
    return torch.cat(tensors, dim=-1)


@triton.jit
def tenth_order_fwd(
    coord_ptr: tl.tensor,
    output_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
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
    CONST001 = 1.75869118663323
    CONST002 = -1021.92317475320
    CONST004 = 4.58257569495584
    CONST005 = 6.63243980843400
    CONST006 = 4.82870805793735
    CONST007 = 4.97432985632550
    CONST008 = 1545.18657853995
    CONST009 = 10.5521471197994
    CONST010 = 12.1657520803952
    CONST011 = 13.2648796168680
    CONST013 = 15.7883647328499
    CONST014 = 15.7302121789667
    CONST015 = 16.4144510752435
    CONST016 = 12.8765548211663
    CONST017 = 19.3148322317494
    CONST018 = 16.7271353825295
    CONST019 = 22.8629854262320
    CONST020 = 535.268332240943
    CONST021 = 23.2135393295190
    CONST022 = 24.6216766128653
    CONST023 = 27.2034486491732
    CONST024 = 541.428124558099
    CONST025 = -994.666978169547
    CONST026 = 33.9852909359329
    CONST027 = 33.9852909359329
    CONST028 = 35.5238206489124
    CONST029 = -984.867064514610
    CONST030 = -4.82870805793735
    CONST031 = 1070.53666448189
    CONST032 = -463.555973561985
    CONST034 = 53.2857309733686
    CONST035 = 53.2857309733686
    CONST036 = 56.3871618715269
    CONST037 = 56.3871618715269
    CONST039 = -1989.33395633909
    CONST041 = -450.224943778107
    CONST042 = 66.9085415301178
    CONST043 = 69.6406179885570
    CONST044 = 69.6406179885570
    CONST045 = -437.967074894228
    CONST046 = 77.2593289269976
    CONST047 = 78.6510608948335
    CONST049 = -1969.73412902922
    CONST050 = 77.3468749368712
    CONST051 = 1624.28437367430
    CONST054 = 94.7301883970997
    CONST056 = 100.362812295177
    CONST057 = -412.049754277320
    CONST058 = 101.517773354644
    CONST059 = -5.63871618715269
    CONST060 = -406.071093418574
    CONST061 = 109.491768723557
    CONST062 = -393.946825805844
    CONST063 = -902.194589944431
    CONST065 = -386.296644634988
    CONST066 = -386.296644634988
    CONST070 = 4.97432985632550
    CONST071 = 150.074981259369
    CONST074 = 685.526905959165
    CONST075 = -337.668707833581
    CONST076 = -337.668707833581
    CONST077 = 176.178376404427
    CONST078 = 176.592751833137
    CONST079 = 185.708314636152
    CONST080 = -326.441383790078
    CONST081 = -1.60956935264578
    CONST082 = -1.97354559160624
    CONST083 = 196.973412902922
    CONST085 = -824.099508554641
    CONST087 = -1.97354559160624
    CONST088 = -305.867618423396
    CONST089 = -305.867618423396
    CONST090 = 721.755671955545
    CONST091 = -305.867618423396
    CONST092 = -300.731529981477
    CONST093 = -300.731529981477
    CONST094 = -1.75869118663323
    CONST095 = -290.050781013267
    CONST097 = 225.548647486108
    CONST098 = 225.548647486108
    CONST099 = -284.190565191299
    CONST101 = -278.562471954228
    CONST102 = -278.562471954228
    CONST103 = -787.893651611688
    CONST104 = -787.893651611688
    CONST105 = 772.593289269975
    CONST106 = 787.893651611688
    CONST107 = 787.893651611688
    CONST108 = 278.562471954228
    CONST109 = -742.833258544608
    CONST110 = -1.65810995210850
    CONST112 = -1761.78376404427
    CONST113 = -223.028471767059
    CONST114 = -734.076568351780
    CONST116 = -220.222970505534
    CONST117 = 1321.33782303320
    CONST118 = 1321.33782303320
    CONST119 = -203.035546709287
    CONST120 = -1.65810995210850
    CONST121 = -196.973412902922
    CONST122 = -196.973412902922
    CONST123 = -696.406179885570
    CONST125 = 338.322971229162
    CONST126 = -1181.84047741753
    CONST127 = -669.085415301178
    CONST128 = -669.085415301178
    CONST129 = -154.518657853995
    CONST130 = -154.518657853995
    CONST131 = 360.877835977772
    CONST132 = -150.074981259369
    CONST133 = -2707.14062279049
    CONST134 = -146.815313670356
    CONST135 = 880.891882022136
    CONST136 = 1392.81235977114
    CONST137 = 1392.81235977114
    CONST138 = -131.315608601948
    CONST139 = -131.315608601948
    CONST141 = -125.841697431734
    CONST142 = -125.841697431734
    CONST143 = -122.415518921279
    CONST145 = 406.071093418574
    CONST146 = -103.107953136506
    CONST147 = -103.107953136506
    CONST148 = -101.517773354644
    CONST149 = -98.4867064514610
    CONST150 = 412.049754277320
    CONST151 = -94.7301883970997
    CONST152 = -1114.24988781691
    CONST153 = -88.2963759165686
    CONST154 = -1624.28437367430
    CONST155 = -82.8889148474622
    CONST156 = -82.8889148474622
    CONST158 = -590.920238708766
    CONST159 = -77.3468749368713
    CONST160 = -77.2593289269975
    CONST161 = 2486.66744542387
    CONST162 = -2626.31217203896
    CONST165 = -571.272421632637
    CONST166 = -56.2781179722634
    CONST167 = -49.2433532257305
    CONST168 = -49.2433532257305
    CONST169 = 984.867064514610
    CONST170 = -541.428124558099
    CONST171 = -24.6216766128653
    CONST172 = -22.8629854262320
    CONST173 = -16.4144510752435
    CONST174 = -15.7883647328499
    CONST175 = -14.0695294930659
    CONST176 = -13.2648796168680
    CONST177 = -11.2774323743054
    CONST178 = -14.5025390506634
    CONST179 = -6.63243980843400
    CONST180 = -5.63871618715269
    CONST181 = 1532.88476212980
    CONST182 = -3.21913870529156
    CONST183 = -2.72034486491732
    CONST184 = -1.12774323743054
    # ordering is really messy because I've refactored
    # the higher powers in terms of the lower ones
    VAR05 = x * x * x * x * x
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR00 = VAR05 * VAR05
    VAR01 = VAR05 * VAR06
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR14 = y * y * y * y * y
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR09 = VAR14 * VAR14
    VAR10 = VAR14 * VAR15
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR23 = z * z * z * z * z
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR18 = VAR23 * VAR23
    VAR19 = VAR23 * VAR24
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    # -------------------- kernel implementations
    Y00 = (
        CONST023 * VAR01 * z
        + CONST023 * VAR19 * x
        + CONST074 * VAR05 * VAR23
        + CONST080 * VAR03 * VAR25
        + CONST080 * VAR07 * VAR21
    )
    Y01 = y * (
        CONST002 * VAR07 * VAR22
        + CONST010 * VAR01
        + CONST045 * VAR03 * VAR26
        + CONST061 * VAR20 * x
        + CONST181 * VAR05 * VAR24
    )
    Y02 = (
        CONST013 * VAR01 * z
        + CONST054 * VAR07 * VAR21
        + CONST151 * VAR03 * VAR25
        + CONST174 * VAR19 * x
        + VAR17
        * (
            -CONST039 * VAR05 * VAR25
            + CONST039 * VAR07 * VAR23
            + CONST099 * VAR03 * z
            - CONST099 * VAR21 * x
        )
    )
    Y03 = VAR16 * (
        CONST024 * VAR22 * x
        + CONST051 * VAR05 * VAR26
        + CONST133 * VAR07 * VAR24
        + CONST159 * VAR03
    ) + y * (
        CONST095 * VAR03 * VAR26
        - CONST119 * VAR05 * VAR24
        + CONST145 * VAR07 * VAR22
        + CONST148 * VAR20 * x
        - CONST178 * VAR01
    )
    Y04 = (
        CONST009 * VAR01 * z
        + VAR03 * (CONST076 * VAR17 * z + CONST175 * VAR25)
        + VAR05 * (CONST106 * VAR15 * z + CONST107 * VAR17 * VAR25 + CONST167 * VAR23)
        + VAR07
        * (CONST106 * VAR17 * VAR23 + CONST162 * VAR15 * VAR25 + CONST175 * VAR21)
        + x * (CONST009 * VAR19 + CONST075 * VAR17 * VAR21 + CONST106 * VAR15 * VAR23)
    )
    Y05 = (
        VAR14 * (CONST077 * VAR05 + CONST112 * VAR07 * VAR26 + CONST135 * VAR24 * x)
        + VAR16
        * (
            -CONST114 * VAR07 * VAR24
            + CONST114 * VAR22 * x
            + CONST117 * VAR05 * VAR26
            + CONST134 * VAR03
        )
        + y
        * (
            CONST014 * VAR01
            + CONST047 * VAR20 * x
            + CONST116 * VAR05 * VAR24
            + CONST141 * VAR03 * VAR26
        )
    )
    Y06 = (
        CONST005 * VAR01 * z
        + VAR03 * (CONST011 * VAR25 + CONST102 * VAR17 * z)
        + VAR05 * (CONST101 * VAR17 * VAR25 - CONST152 * VAR15 * z)
        + VAR07 * (CONST108 * VAR17 * VAR23 + CONST109 * VAR13 * z + CONST176 * VAR21)
        + x
        * (
            CONST108 * VAR17 * VAR21
            - CONST109 * VAR13 * VAR25
            + CONST152 * VAR15 * VAR23
            + CONST179 * VAR19
        )
    )
    Y07 = (
        VAR12 * (-CONST041 * VAR26 * x + CONST132 * VAR07)
        + VAR14 * (-CONST062 * VAR05 + CONST103 * VAR07 * VAR26 + CONST126 * VAR24 * x)
        + VAR16
        * (
            CONST083 * VAR05 * VAR26
            + CONST121 * VAR03
            - CONST158 * VAR22 * x
            + CONST169 * VAR07 * VAR24
        )
        + y
        * (
            CONST015 * VAR01
            + CONST138 * VAR07 * VAR22
            + CONST149 * VAR05 * VAR24
            + CONST168 * VAR20 * x
        )
    )
    Y08 = (
        -CONST182 * VAR01 * z
        + VAR03 * (CONST016 * VAR25 + CONST129 * VAR17 * z)
        + VAR05 * (CONST017 * VAR23 + CONST032 * VAR17 * VAR25 + CONST105 * VAR15 * z)
        + VAR07
        * (
            CONST008 * VAR15 * VAR25
            + CONST016 * VAR21
            + CONST032 * VAR17 * VAR23
            + CONST085 * VAR13 * z
        )
        + x
        * (
            CONST078 * VAR11 * z
            + CONST085 * VAR13 * VAR25
            + CONST105 * VAR15 * VAR23
            + CONST129 * VAR17 * VAR21
            - CONST182 * VAR19
        )
    )
    Y09 = (
        CONST018 * VAR01 * y
        + VAR03 * (CONST042 * VAR26 * y + CONST113 * VAR16)
        + VAR05 * (CONST020 * VAR14 + CONST056 * VAR24 * y + CONST128 * VAR16 * VAR26)
        + VAR07
        * (
            CONST031 * VAR14 * VAR26
            + CONST042 * VAR22 * y
            + CONST088 * VAR12
            + CONST127 * VAR16 * VAR24
        )
        + x
        * (
            CONST018 * VAR20 * y
            + CONST020 * VAR14 * VAR24
            + CONST026 * VAR10
            + CONST088 * VAR12 * VAR26
            + CONST113 * VAR16 * VAR22
        )
    )
    Y10 = (
        CONST004 * VAR09
        + CONST037 * VAR17 * VAR20
        + CONST093 * VAR15 * VAR22
        + CONST131 * VAR13 * VAR24
        + CONST147 * VAR11 * VAR26
        + CONST184 * VAR00
        + CONST184 * VAR18
        + VAR02 * (CONST036 * VAR17 + CONST059 * VAR26)
        + VAR04 * (CONST092 * VAR15 + CONST098 * VAR17 * VAR26 + CONST177 * VAR24)
        + VAR06
        * (
            CONST063 * VAR15 * VAR26
            + CONST125 * VAR17 * VAR24
            + CONST131 * VAR13
            + CONST177 * VAR22
        )
        + VAR08
        * (
            CONST063 * VAR15 * VAR24
            + CONST090 * VAR13 * VAR26
            + CONST097 * VAR17 * VAR22
            + CONST146 * VAR11
            + CONST180 * VAR20
        )
    )
    Y11 = (
        CONST018 * VAR19 * y
        + VAR21 * (CONST042 * VAR08 * y + CONST113 * VAR16)
        + VAR23 * (CONST020 * VAR14 + CONST056 * VAR06 * y + CONST128 * VAR08 * VAR16)
        + VAR25
        * (
            CONST031 * VAR08 * VAR14
            + CONST042 * VAR04 * y
            + CONST091 * VAR12
            + CONST127 * VAR06 * VAR16
        )
        + z
        * (
            CONST018 * VAR02 * y
            + CONST020 * VAR06 * VAR14
            + CONST027 * VAR10
            + CONST089 * VAR08 * VAR12
            + CONST113 * VAR04 * VAR16
        )
    )
    Y12 = (
        CONST057 * VAR13 * VAR24
        - CONST066 * VAR15 * VAR22
        + CONST081 * VAR00
        - CONST081 * VAR18
        - CONST153 * VAR11 * VAR26
        + CONST160 * VAR17 * VAR20
        + VAR02 * (CONST030 * VAR26 + CONST046 * VAR17)
        + VAR04 * (CONST066 * VAR15 - CONST129 * VAR17 * VAR26 + CONST182 * VAR24)
        + VAR06 * (CONST065 * VAR15 * VAR26 + CONST150 * VAR13 - CONST182 * VAR22)
        + VAR08
        * (
            CONST006 * VAR20
            - CONST066 * VAR15 * VAR24
            + CONST130 * VAR17 * VAR22
            + CONST153 * VAR11
        )
    )
    Y13 = (
        VAR12 * (CONST041 * VAR08 * z + CONST071 * VAR25)
        + VAR14 * (CONST062 * VAR23 + CONST107 * VAR08 * VAR25 - CONST126 * VAR06 * z)
        + VAR16
        * (
            CONST029 * VAR06 * VAR25
            - CONST121 * VAR21
            + CONST122 * VAR08 * VAR23
            + CONST158 * VAR04 * z
        )
        + y
        * (
            -CONST138 * VAR04 * VAR25
            - CONST149 * VAR06 * VAR23
            - CONST168 * VAR02 * z
            + CONST173 * VAR19
        )
    )
    Y14 = (
        CONST044 * VAR17 * VAR20
        + CONST079 * VAR13 * VAR24
        + CONST101 * VAR15 * VAR22
        + CONST110 * VAR00
        + CONST120 * VAR18
        + VAR02 * (CONST043 * VAR17 + CONST070 * VAR26)
        + VAR04 * (CONST021 * VAR24 + CONST101 * VAR15 + CONST101 * VAR17 * VAR26)
        + VAR06
        * (
            CONST021 * VAR22
            + CONST079 * VAR13
            + CONST123 * VAR17 * VAR24
            + CONST137 * VAR15 * VAR26
        )
        + VAR08
        * (
            CONST007 * VAR20
            + CONST101 * VAR17 * VAR22
            + CONST136 * VAR15 * VAR24
            + CONST152 * VAR13 * VAR26
        )
    )
    Y15 = (
        VAR14 * (CONST077 * VAR23 + CONST112 * VAR08 * VAR25 + CONST135 * VAR06 * z)
        + VAR16
        * (
            CONST114 * VAR04 * z
            - CONST114 * VAR06 * VAR25
            + CONST118 * VAR08 * VAR23
            + CONST134 * VAR21
        )
        + y
        * (
            CONST014 * VAR19
            + CONST047 * VAR02 * z
            + CONST116 * VAR06 * VAR23
            + CONST142 * VAR08 * VAR21
        )
    )
    Y16 = (
        CONST001 * VAR18
        + CONST094 * VAR00
        - CONST139 * VAR15 * VAR22
        + CONST166 * VAR17 * VAR20
        + VAR02 * (CONST019 * VAR26 - CONST166 * VAR17)
        + VAR04 * (CONST022 * VAR24 + CONST104 * VAR17 * VAR26 + CONST139 * VAR15)
        + VAR06 * (-CONST049 * VAR15 * VAR26 + CONST171 * VAR22)
        + VAR08
        * (CONST049 * VAR15 * VAR24 + CONST106 * VAR17 * VAR22 + CONST172 * VAR20)
    )
    Y17 = VAR16 * (
        CONST050 * VAR21
        - CONST133 * VAR06 * VAR25
        + CONST154 * VAR08 * VAR23
        + CONST170 * VAR04 * z
    ) + y * (
        CONST058 * VAR02 * z
        + CONST060 * VAR04 * VAR25
        - CONST095 * VAR08 * VAR21
        + CONST119 * VAR06 * VAR23
        + CONST178 * VAR19
    )
    Y18 = (
        CONST034 * VAR02 * VAR26
        + CONST035 * VAR08 * VAR20
        + CONST082 * VAR00
        + CONST087 * VAR18
        + CONST155 * VAR04 * VAR24
        + CONST156 * VAR06 * VAR22
        + VAR17
        * (
            CONST025 * VAR04 * VAR26
            + CONST025 * VAR08 * VAR22
            + CONST028 * VAR02
            + CONST028 * VAR20
            + CONST161 * VAR06 * VAR24
        )
    )
    Y19 = y * (
        CONST002 * VAR04 * VAR25
        + CONST010 * VAR19
        + CONST045 * VAR08 * VAR21
        + CONST061 * VAR02 * z
        + CONST181 * VAR06 * VAR23
    )
    Y20 = (
        -CONST143 * VAR02 * VAR26
        + CONST143 * VAR08 * VAR20
        + CONST165 * VAR04 * VAR24
        - CONST165 * VAR06 * VAR22
        + CONST183 * VAR00
        - CONST183 * VAR18
    )
    output_stride = 21  # [2l + 1]
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id)
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
    tl.store(
        output_ptr + output_row_offset + 15,
        Y15,
        mask=output_row_offset + 15 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 16,
        Y16,
        mask=output_row_offset + 16 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 17,
        Y17,
        mask=output_row_offset + 17 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 18,
        Y18,
        mask=output_row_offset + 18 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 19,
        Y19,
        mask=output_row_offset + 19 < output_numel,
    )
    tl.store(
        output_ptr + output_row_offset + 20,
        Y20,
        mask=output_row_offset + 20 < output_numel,
    )


@triton.jit
def tenth_order_bwd(
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
    output_stride = 21  # [2l + 1]
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id)
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
    g_15 = tl.load(
        sph_grad_ptr + output_row_offset + 15,
        mask=output_row_offset + 15 < output_numel,
    )
    g_16 = tl.load(
        sph_grad_ptr + output_row_offset + 16,
        mask=output_row_offset + 16 < output_numel,
    )
    g_17 = tl.load(
        sph_grad_ptr + output_row_offset + 17,
        mask=output_row_offset + 17 < output_numel,
    )
    g_18 = tl.load(
        sph_grad_ptr + output_row_offset + 18,
        mask=output_row_offset + 18 < output_numel,
    )
    g_19 = tl.load(
        sph_grad_ptr + output_row_offset + 19,
        mask=output_row_offset + 19 < output_numel,
    )
    g_20 = tl.load(
        sph_grad_ptr + output_row_offset + 20,
        mask=output_row_offset + 20 < output_numel,
    )
    # -------------------- variable and constant definitions
    CONST000 = 2.00000000000000
    CONST002 = 4.00000000000000
    CONST003 = 4.82870805793735
    CONST004 = 6.00000000000000
    CONST005 = 4.97432985632550
    CONST006 = 8.00000000000000
    CONST007 = 4.97432985632550
    CONST008 = 10.5521471197994
    CONST009 = 3.00000000000000
    CONST010 = 5.00000000000000
    CONST011 = 7.00000000000000
    CONST012 = 13.2648796168680
    CONST014 = 12.1657520803952
    CONST015 = 16.7271353825295
    CONST016 = -2030.35546709287
    CONST017 = 19.3148322317494
    CONST018 = -6131.53904851919
    CONST019 = 22.8629854262320
    CONST020 = 23.2135393295190
    CONST021 = 24.6216766128653
    CONST022 = 17.5869118663323
    CONST024 = 28.9722483476241
    CONST025 = 33.9852909359329
    CONST026 = 33.9852909359329
    CONST027 = 35.5238206489124
    CONST028 = 6180.74631415980
    CONST029 = 38.6296644634988
    CONST030 = 39.7946388506040
    CONST031 = 38.6296644634988
    CONST032 = -2007.25624590353
    CONST033 = -2007.25624590353
    CONST034 = 45.8257569495584
    CONST035 = 45.7259708524640
    CONST037 = 56.3871618715269
    CONST038 = 56.2781179722634
    CONST039 = -1989.33395633909
    CONST040 = -1989.33395633909
    CONST041 = 59.6919582759060
    CONST042 = 66.9085415301178
    CONST043 = 69.6406179885570
    CONST044 = -8121.42186837148
    CONST045 = 77.2593289269976
    CONST046 = 78.6510608948335
    CONST047 = -1969.73412902922
    CONST048 = 77.3468749368712
    CONST049 = -1969.73412902922
    CONST050 = -9.65741611587469
    CONST051 = 90.1358837481638
    CONST053 = 94.9693240781945
    CONST055 = 96.5741611587469
    CONST057 = 98.4867064514610
    CONST058 = 100.362812295177
    CONST059 = 101.517773354644
    CONST060 = 106.571461946737
    CONST061 = 106.571461946737
    CONST062 = 109.491768723557
    CONST063 = 109.491768723557
    CONST064 = 112.774323743054
    CONST065 = 112.774323743054
    CONST067 = 2165.26701586663
    CONST070 = 133.817083060236
    CONST071 = 139.281235977114
    CONST072 = 139.281235977114
    CONST073 = 141.571909610700
    CONST074 = 142.095282595650
    CONST075 = 147.730059677192
    CONST076 = 150.544218442765
    CONST077 = 150.074981259369
    CONST079 = 2202.22970505534
    CONST080 = -3939.46825805844
    CONST081 = -5968.00186901728
    CONST082 = 176.592751833137
    CONST083 = 176.178376404427
    CONST085 = 185.708314636152
    CONST087 = 196.973412902922
    CONST089 = 225.548647486108
    CONST090 = 225.548647486108
    CONST091 = 4330.53403173327
    CONST093 = 244.831037842559
    CONST094 = -1804.38917988886
    CONST095 = -1804.38917988886
    CONST097 = 2317.77986780993
    CONST098 = 278.562471954228
    CONST100 = 284.190565191299
    CONST101 = -1761.78376404427
    CONST103 = -9946.66978169547
    CONST104 = 9.94865971265100
    CONST108 = -7878.93651611688
    CONST111 = 338.322971229162
    CONST112 = 360.877835977772
    CONST114 = -1671.37483172537
    CONST116 = 2436.42656051144
    CONST119 = 393.946825805844
    CONST120 = -1648.19901710928
    CONST121 = 401.451249180707
    CONST122 = 406.071093418574
    CONST123 = 412.049754277320
    CONST125 = -1624.28437367430
    CONST126 = 426.285847786949
    CONST127 = 426.285847786948
    CONST128 = 2486.66744542387
    CONST130 = 451.097294972216
    CONST131 = 451.097294972216
    CONST132 = 451.097294972215
    CONST133 = 6606.68911516602
    CONST134 = 6606.68911516602
    CONST135 = -1575.78730322338
    CONST136 = -1575.78730322338
    CONST137 = -3608.77835977772
    CONST139 = -1545.18657853995
    CONST140 = -1545.18657853995
    CONST142 = 535.268332240943
    CONST143 = 4635.55973561985
    CONST144 = 541.428124558099
    CONST145 = -3545.52143225260
    CONST146 = 557.124943908456
    CONST147 = -3523.56752808854
    CONST148 = -5571.24943908456
    CONST151 = 15.7883647328499
    CONST153 = 2642.67564606641
    CONST154 = 2642.67564606641
    CONST155 = 2676.34166120471
    CONST156 = 629.208487158668
    CONST158 = 4727.36190967013
    CONST159 = -1392.81235977114
    CONST160 = -1390.66792068596
    CONST162 = 663.111318779698
    CONST163 = -3427.63452979582
    CONST164 = -1378.81389032045
    CONST165 = 676.645942458323
    CONST167 = -1338.17083060236
    CONST168 = -1338.17083060236
    CONST169 = 721.755671955545
    CONST171 = 2785.62471954228
    CONST173 = 772.593289269975
    CONST175 = 787.893651611688
    CONST176 = 787.893651611688
    CONST177 = 6.63243980843400
    CONST178 = 812.142186837148
    CONST180 = -1218.21328025572
    CONST181 = -1202.92611992591
    CONST182 = -1202.92611992591
    CONST183 = -3248.56874734859
    CONST184 = -3248.56874734859
    CONST185 = -5285.35129213281
    CONST186 = -1181.84047741753
    CONST190 = 2936.30627340712
    CONST192 = 2954.60119354383
    CONST193 = -1114.24988781691
    CONST194 = -16.5810995210850
    CONST195 = -1101.11485252767
    CONST196 = -1081.63060497797
    CONST197 = 15.7302121789667
    CONST199 = 984.867064514610
    CONST202 = -1027.70719569249
    CONST203 = -1021.92317475320
    CONST204 = -3065.76952425960
    CONST205 = -1015.17773354644
    CONST206 = 3090.37315707990
    CONST207 = -994.666978169547
    CONST208 = -984.867064514610
    CONST209 = -984.867064514610
    CONST210 = -979.324151370235
    CONST211 = 1070.53666448189
    CONST212 = -979.324151370235
    CONST213 = 3151.57460644675
    CONST216 = -927.111947123971
    CONST217 = -927.111947123970
    CONST218 = -5.63871618715269
    CONST219 = -2954.60119354383
    CONST220 = -902.194589944431
    CONST221 = -900.449887556215
    CONST222 = -880.891882022136
    CONST223 = -880.891882022136
    CONST224 = -875.934149788456
    CONST226 = -4944.59705132784
    CONST228 = 3248.56874734859
    CONST229 = -835.687415862684
    CONST230 = 1218.21328025572
    CONST231 = -824.099508554641
    CONST232 = -824.863625092051
    CONST233 = -824.863625092051
    CONST234 = -812.142186837148
    CONST235 = 5352.68332240943
    CONST236 = -787.893651611688
    CONST237 = -787.893651611688
    CONST238 = -772.593289269976
    CONST239 = -742.833258544608
    CONST240 = -2785.62471954228
    CONST241 = -734.076568351780
    CONST242 = 1321.33782303320
    CONST243 = 1321.33782303320
    CONST244 = -706.371007332549
    CONST245 = -696.406179885570
    CONST246 = 1353.29188491665
    CONST247 = -675.337415667161
    CONST248 = -675.337415667161
    CONST250 = 3427.63452979582
    CONST251 = -669.085415301178
    CONST252 = -669.085415301178
    CONST253 = -669.085415301178
    CONST255 = -663.111318779698
    CONST256 = -2707.14062279049
    CONST258 = 1392.81235977114
    CONST259 = 1412.74201466510
    CONST260 = -4727.36190967013
    CONST261 = -2676.34166120471
    CONST262 = -618.074631415980
    CONST263 = -611.735236846792
    CONST264 = -611.735236846792
    CONST265 = 1443.51134391109
    CONST266 = -590.920238708766
    CONST267 = -10828.5624911620
    CONST268 = -580.101562026534
    CONST269 = -2626.31217203896
    CONST272 = 5571.24943908456
    CONST273 = -12.8765548211663
    CONST274 = -557.124943908456
    CONST275 = -557.124943908456
    CONST277 = -541.428124558099
    CONST278 = -6685.49932690147
    CONST279 = 7664.42381064899
    CONST280 = -525.262434407792
    CONST281 = 1532.88476212980
    CONST283 = -497.333489084773
    CONST284 = -497.333489084773
    CONST285 = -492.433532257305
    CONST286 = 1575.78730322338
    CONST287 = 1575.78730322338
    CONST288 = -463.555973561985
    CONST289 = -450.224943778107
    CONST290 = -450.224943778107
    CONST291 = -450.224943778108
    CONST292 = -437.967074894228
    CONST293 = -2472.29852566392
    CONST294 = 1624.28437367430
    CONST295 = -2472.29852566392
    CONST296 = -406.071093418574
    CONST297 = -393.946825805844
    CONST298 = -393.946825805844
    CONST299 = -2436.42656051144
    CONST300 = -386.296644634988
    CONST301 = -386.296644634988
    CONST302 = -4456.99955126765
    CONST303 = -337.668707833581
    CONST304 = -337.668707833581
    CONST305 = -331.555659389849
    CONST306 = -331.555659389849
    CONST307 = -2363.68095483506
    CONST309 = -309.037315707990
    CONST310 = -4404.45941011068
    CONST311 = -309.037315707990
    CONST312 = -305.867618423396
    CONST313 = -305.867618423396
    CONST314 = -305.867618423396
    CONST315 = -300.731529981477
    CONST316 = 9946.66978169547
    CONST318 = -290.050781013267
    CONST319 = -284.190565191299
    CONST320 = -278.562471954228
    CONST321 = -278.562471954228
    CONST322 = -2317.77986780993
    CONST323 = -10505.2486881558
    CONST324 = -251.683394863467
    CONST325 = -251.683394863467
    CONST326 = -246.216766128653
    CONST327 = -244.831037842559
    CONST328 = -2285.08968653055
    CONST329 = -2285.08968653055
    CONST330 = 3862.96644634988
    CONST331 = -223.028471767059
    CONST332 = -220.222970505534
    CONST333 = -206.215906273013
    CONST334 = -203.035546709287
    CONST335 = -196.973412902922
    CONST336 = -196.973412902922
    CONST337 = -182.903883409856
    CONST338 = -2228.49977563382
    CONST340 = 16.4144510752435
    CONST341 = 3939.46825805844
    CONST342 = 3939.46825805844
    CONST343 = -154.518657853995
    CONST344 = -154.518657853995
    CONST345 = -150.074981259369
    CONST346 = -147.730059677191
    CONST347 = -146.815313670356
    CONST348 = -142.095282595650
    CONST349 = -131.315608601948
    CONST350 = -131.315608601948
    CONST351 = -130.522851455970
    CONST352 = -125.841697431734
    CONST353 = -125.841697431734
    CONST354 = -112.556235944527
    CONST355 = -103.107953136506
    CONST356 = -101.517773354644
    CONST357 = 1949.93730367960
    CONST358 = -98.4867064514610
    CONST359 = -98.4867064514610
    CONST360 = -2141.07332896377
    CONST361 = -2141.07332896377
    CONST362 = -92.8541573180760
    CONST363 = -88.2963759165686
    CONST366 = -77.3468749368713
    CONST367 = 8121.42186837148
    CONST369 = -67.6645942458323
    CONST372 = -59.6919582759060
    CONST373 = -49.2433532257305
    CONST374 = -49.2433532257305
    CONST375 = -45.1097294972216
    CONST376 = -45.1097294972216
    CONST377 = -42.2085884791976
    CONST378 = -27.2034486491732
    CONST379 = -24.6216766128653
    CONST380 = -22.8629854262320
    CONST381 = -19.7354559160624
    CONST383 = -17.5869118663323
    CONST384 = -16.4144510752435
    CONST385 = -16.0956935264578
    CONST386 = -14.5025390506634
    CONST388 = -16.5810995210850
    CONST389 = -15.7883647328499
    CONST390 = -14.0695294930659
    CONST391 = -11.2774323743054
    CONST392 = -11.2774323743054
    CONST393 = -13.2648796168680
    CONST394 = -6.63243980843400
    CONST395 = -5.63871618715269
    CONST396 = -4.82870805793735
    CONST397 = -3.21913870529156
    CONST398 = -11.2774323743054
    VAR05 = x * x * x * x * x
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR01 = VAR05 * VAR06
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR14 = y * y * y * y * y
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR10 = VAR14 * VAR15
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR23 = z * z * z * z * z
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR19 = VAR23 * VAR24
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    # -------------------- kernel implementations
    g_x = (
        g_0
        * (
            CONST093 * VAR02 * z
            + CONST210 * VAR08 * VAR21
            + CONST250 * VAR06 * VAR23
            + CONST328 * VAR04 * VAR25
            - CONST378 * VAR19
        )
        + g_1
        * y
        * (
            CONST062 * VAR20
            + CONST063 * VAR02
            + CONST204 * VAR04 * VAR26
            + CONST204 * VAR08 * VAR22
            + CONST279 * VAR06 * VAR24
        )
        + g_10
        * (
            CONST000
            * x
            * (
                CONST089 * VAR17 * VAR22
                + CONST169 * VAR13 * VAR26
                + CONST220 * VAR15 * VAR24
                + CONST355 * VAR11
                + CONST395 * VAR20
            )
            + CONST002
            * VAR07
            * (
                CONST111 * VAR17 * VAR24
                + CONST112 * VAR13
                + CONST220 * VAR15 * VAR26
                + CONST392 * VAR22
            )
            + CONST004
            * VAR05
            * (CONST090 * VAR17 * VAR26 + CONST315 * VAR15 + CONST392 * VAR24)
            + CONST006 * VAR03 * (CONST037 * VAR17 + CONST218 * VAR26)
            + CONST391 * VAR01
        )
        + g_11
        * (
            CONST070 * VAR21 * x * y
            + VAR23 * (CONST121 * VAR07 * y + CONST168 * VAR16 * x)
            + VAR25
            * (CONST121 * VAR05 * y + CONST261 * VAR07 * VAR16 - CONST361 * VAR14 * x)
            + z
            * (
                CONST070 * VAR03 * y
                + CONST167 * VAR05 * VAR16
                + CONST263 * VAR12 * x
                - CONST361 * VAR07 * VAR14
            )
        )
        + g_12
        * (
            CONST000
            * x
            * (
                CONST003 * VAR20
                - CONST301 * VAR15 * VAR24
                + CONST343 * VAR17 * VAR22
                + CONST363 * VAR11
            )
            + CONST002
            * VAR07
            * (CONST123 * VAR13 + CONST300 * VAR15 * VAR26 - CONST397 * VAR22)
            + CONST004
            * VAR05
            * (CONST301 * VAR15 - CONST344 * VAR17 * VAR26 + CONST397 * VAR24)
            + CONST006 * VAR03 * (CONST045 * VAR17 + CONST396 * VAR26)
            + CONST385 * VAR01
        )
        + g_13
        * (
            CONST221 * VAR12 * x * z
            + VAR14 * (-CONST260 * VAR07 * z + CONST286 * VAR25 * x)
            + VAR16
            * (CONST080 * VAR07 * VAR25 + CONST145 * VAR05 * z + CONST297 * VAR23 * x)
            + y
            * (
                -CONST237 * VAR05 * VAR25
                - CONST297 * VAR07 * VAR23
                - CONST298 * VAR03 * z
            )
        )
        + g_14
        * (
            CONST000
            * x
            * (
                CONST005 * VAR20
                - CONST159 * VAR15 * VAR24
                + CONST193 * VAR13 * VAR26
                + CONST320 * VAR17 * VAR22
            )
            + CONST002
            * VAR07
            * (
                CONST020 * VAR22
                + CONST085 * VAR13
                + CONST245 * VAR17 * VAR24
                + CONST258 * VAR15 * VAR26
            )
            + CONST004
            * VAR05
            * (CONST020 * VAR24 + CONST320 * VAR15 + CONST320 * VAR17 * VAR26)
            + CONST006 * VAR03 * (CONST007 * VAR26 + CONST043 * VAR17)
            + CONST388 * VAR01
        )
        + g_15
        * (
            VAR14 * (-CONST147 * VAR07 * z + CONST147 * VAR25 * x)
            + VAR16
            * (CONST153 * VAR23 * x + CONST190 * VAR07 * VAR25 + CONST310 * VAR05 * z)
            + y
            * (CONST156 * VAR03 * z + CONST222 * VAR07 * VAR23 + CONST324 * VAR21 * x)
        )
        + g_16
        * (
            CONST000
            * x
            * (CONST047 * VAR15 * VAR24 + CONST175 * VAR17 * VAR22 + CONST380 * VAR20)
            + CONST002 * VAR07 * (-CONST047 * VAR15 * VAR26 + CONST379 * VAR22)
            + CONST004
            * VAR05
            * (CONST021 * VAR24 + CONST236 * VAR17 * VAR26 + CONST349 * VAR15)
            + CONST006 * VAR03 * (CONST019 * VAR26 + CONST038 * VAR17)
            + CONST383 * VAR01
        )
        + g_17
        * (
            VAR16
            * (CONST183 * VAR23 * x + CONST184 * VAR05 * z - CONST267 * VAR07 * VAR25)
            + y
            * (
                CONST178 * VAR03 * z
                + CONST234 * VAR07 * VAR23
                - CONST268 * VAR21 * x
                + CONST299 * VAR05 * VAR25
            )
        )
        + g_18
        * (
            CONST060 * VAR20 * x
            + CONST126 * VAR03 * VAR26
            + CONST283 * VAR05 * VAR24
            + CONST305 * VAR07 * VAR22
            + CONST381 * VAR01
            + VAR17
            * (
                CONST039 * VAR22 * x
                + CONST081 * VAR05 * VAR26
                + CONST316 * VAR07 * VAR24
                - CONST319 * VAR03
            )
        )
        + g_19
        * y
        * (
            CONST018 * VAR05 * VAR25
            - CONST018 * VAR07 * VAR23
            - CONST224 * VAR03 * z
            + CONST224 * VAR21 * x
        )
        + g_2
        * (
            CONST074 * VAR02 * z
            + CONST100 * VAR08 * VAR21
            + CONST255 * VAR04 * VAR25
            + CONST389 * VAR19
            + VAR17
            * (
                CONST040 * VAR04 * z
                + CONST081 * VAR08 * VAR23
                - CONST103 * VAR06 * VAR25
                - CONST319 * VAR21
            )
        )
        + g_20
        * (
            CONST163 * VAR05 * VAR24
            - CONST212 * VAR03 * VAR26
            + CONST327 * VAR20 * x
            - CONST329 * VAR07 * VAR22
            + CONST378 * VAR01
        )
        + g_3
        * (
            VAR16
            * (
                CONST044 * VAR08 * VAR24
                + CONST144 * VAR22
                + CONST277 * VAR04
                + CONST367 * VAR06 * VAR26
            )
            + y
            * (
                CONST016 * VAR04 * VAR26
                - CONST205 * VAR06 * VAR24
                + CONST230 * VAR08 * VAR22
                - CONST351 * VAR02
                + CONST356 * VAR20
            )
        )
        + g_4
        * (
            CONST008 * VAR19
            + CONST009
            * VAR08
            * (CONST175 * VAR17 * VAR23 + CONST269 * VAR15 * VAR25 + CONST390 * VAR21)
            + CONST010
            * VAR06
            * (CONST175 * VAR15 * z + CONST176 * VAR17 * VAR25 + CONST373 * VAR23)
            + CONST011 * VAR04 * (CONST303 * VAR17 * z + CONST390 * VAR25)
            + CONST053 * VAR02 * z
            + CONST175 * VAR15 * VAR23
            + CONST304 * VAR17 * VAR21
        )
        + g_5
        * (
            VAR14 * (CONST185 * VAR08 * VAR26 - CONST222 * VAR06 - CONST223 * VAR24)
            + VAR16
            * (
                CONST079 * VAR08 * VAR24
                + CONST133 * VAR06 * VAR26
                + CONST202 * VAR04
                + CONST241 * VAR22
            )
            + y
            * (
                CONST046 * VAR20
                + CONST073 * VAR02
                + CONST195 * VAR06 * VAR24
                + CONST222 * VAR04 * VAR26
            )
        )
        + g_6
        * (
            CONST009
            * VAR08
            * (CONST098 * VAR17 * VAR23 + CONST239 * VAR13 * z + CONST393 * VAR21)
            + CONST010 * VAR06 * (-CONST193 * VAR15 * z + CONST320 * VAR17 * VAR25)
            + CONST011 * VAR04 * (CONST012 * VAR25 + CONST321 * VAR17 * z)
            + CONST041 * VAR02 * z
            + CONST098 * VAR17 * VAR21
            + CONST193 * VAR15 * VAR23
            - CONST239 * VAR13 * VAR25
            + CONST394 * VAR19
        )
        + g_7
        * (
            VAR12 * (CONST289 * VAR08 - CONST290 * VAR26)
            + VAR14 * (-CONST049 * VAR06 + CONST186 * VAR24 + CONST307 * VAR08 * VAR26)
            + VAR16
            * (
                CONST164 * VAR04
                + CONST192 * VAR08 * VAR24
                + CONST199 * VAR06 * VAR26
                - CONST266 * VAR22
            )
            + y
            * (
                CONST075 * VAR02
                + CONST285 * VAR06 * VAR24
                + CONST297 * VAR08 * VAR22
                + CONST374 * VAR20
            )
        )
        + g_8
        * (
            CONST009
            * VAR08
            * (
                -CONST140 * VAR15 * VAR25
                + CONST231 * VAR13 * z
                - CONST273 * VAR21
                + CONST288 * VAR17 * VAR23
            )
            + CONST010
            * VAR06
            * (CONST017 * VAR23 + CONST173 * VAR15 * z + CONST288 * VAR17 * VAR25)
            + CONST011 * VAR04 * (-CONST273 * VAR25 + CONST344 * VAR17 * z)
            + CONST024 * VAR02 * z
            + CONST082 * VAR11 * z
            + CONST173 * VAR15 * VAR23
            + CONST231 * VAR13 * VAR25
            + CONST344 * VAR17 * VAR21
            - CONST397 * VAR19
        )
        + g_9
        * (
            CONST009
            * VAR08
            * (
                CONST042 * VAR22 * y
                + CONST211 * VAR14 * VAR26
                + CONST251 * VAR16 * VAR24
                + CONST312 * VAR12
            )
            + CONST010
            * VAR06
            * (CONST058 * VAR24 * y + CONST142 * VAR14 + CONST252 * VAR16 * VAR26)
            + CONST011 * VAR04 * (CONST042 * VAR26 * y + CONST331 * VAR16)
            + CONST015 * VAR20 * y
            + CONST025 * VAR10
            + CONST076 * VAR02 * y
            + CONST142 * VAR14 * VAR24
            + CONST312 * VAR12 * VAR26
            + CONST331 * VAR16 * VAR22
        )
    )
    g_y = (
        CONST000
        * g_18
        * y
        * (
            CONST027 * VAR02
            + CONST027 * VAR20
            + CONST128 * VAR06 * VAR24
            + CONST207 * VAR04 * VAR26
            + CONST207 * VAR08 * VAR22
        )
        + CONST000
        * g_2
        * y
        * (
            -CONST039 * VAR05 * VAR25
            + CONST039 * VAR07 * VAR23
            + CONST319 * VAR03 * z
            - CONST319 * VAR21 * x
        )
        + g_1
        * (
            CONST014 * VAR01
            + CONST062 * VAR20 * x
            + CONST203 * VAR07 * VAR22
            + CONST281 * VAR05 * VAR24
            + CONST292 * VAR03 * VAR26
        )
        + g_10
        * (
            CONST034 * VAR10
            + CONST064 * VAR20 * y
            + CONST065 * VAR02 * y
            + CONST067 * VAR14 * VAR24
            + CONST182 * VAR16 * VAR22
            + CONST233 * VAR12 * VAR26
            + VAR04 * (CONST131 * VAR26 * y + CONST181 * VAR16)
            + VAR06
            * (CONST067 * VAR14 + CONST137 * VAR16 * VAR26 + CONST165 * VAR24 * y)
            + VAR08
            * (
                CONST091 * VAR14 * VAR26
                + CONST130 * VAR22 * y
                + CONST137 * VAR16 * VAR24
                + CONST232 * VAR12
            )
        )
        + g_11
        * (
            CONST015 * VAR19
            + VAR21 * (CONST042 * VAR08 + CONST253 * VAR17)
            + VAR23 * (CONST033 * VAR08 * VAR17 + CONST058 * VAR06 + CONST155 * VAR15)
            + VAR25
            * (
                CONST032 * VAR06 * VAR17
                + CONST042 * VAR04
                + CONST235 * VAR08 * VAR15
                + CONST361 * VAR13
            )
            + z
            * (
                CONST015 * VAR02
                + CONST155 * VAR06 * VAR15
                + CONST253 * VAR04 * VAR17
                - CONST312 * VAR11
                + CONST360 * VAR08 * VAR13
            )
        )
        + g_12
        * (
            -CONST140 * VAR16 * VAR22
            - CONST244 * VAR12 * VAR26
            + CONST293 * VAR14 * VAR24
            + CONST343 * VAR20 * y
            - CONST344 * VAR02 * y
            + VAR04 * (CONST140 * VAR16 - CONST311 * VAR26 * y)
            + VAR06 * (CONST139 * VAR16 * VAR26 - CONST295 * VAR14)
            + VAR08
            * (-CONST140 * VAR16 * VAR24 + CONST244 * VAR12 + CONST309 * VAR22 * y)
        )
        + g_13
        * (
            CONST009
            * VAR17
            * (
                CONST208 * VAR06 * VAR25
                + CONST266 * VAR04 * z
                + CONST335 * VAR08 * VAR23
                - CONST336 * VAR21
            )
            + CONST010
            * VAR15
            * (CONST176 * VAR08 * VAR25 - CONST186 * VAR06 * z + CONST298 * VAR23)
            + CONST011 * VAR13 * (CONST077 * VAR25 + CONST290 * VAR08 * z)
            - CONST350 * VAR04 * VAR25
            - CONST358 * VAR06 * VAR23
            - CONST374 * VAR02 * z
            + CONST384 * VAR19
        )
        + g_14
        * (
            CONST071 * VAR02 * y
            + CONST072 * VAR20 * y
            - CONST193 * VAR14 * VAR24
            + CONST193 * VAR16 * VAR22
            + VAR04 * (CONST193 * VAR16 + CONST274 * VAR26 * y)
            + VAR06
            * (CONST159 * VAR24 * y - CONST193 * VAR14 + CONST272 * VAR16 * VAR26)
            + VAR08
            * (
                -CONST148 * VAR16 * VAR24
                + CONST274 * VAR22 * y
                + CONST278 * VAR14 * VAR26
            )
        )
        + g_15
        * (
            CONST009
            * VAR17
            * (
                CONST241 * VAR04 * z
                - CONST241 * VAR06 * VAR25
                + CONST242 * VAR08 * VAR23
                + CONST347 * VAR21
            )
            + CONST010
            * VAR15
            * (CONST083 * VAR23 + CONST101 * VAR08 * VAR25 - CONST223 * VAR06 * z)
            + CONST046 * VAR02 * z
            + CONST197 * VAR19
            + CONST332 * VAR06 * VAR23
            + CONST352 * VAR08 * VAR21
        )
        + g_16
        * (
            -CONST108 * VAR06 * VAR16 * VAR26
            - CONST280 * VAR16 * VAR22
            - CONST354 * VAR02 * y
            + CONST354 * VAR20 * y
            + VAR04 * (CONST135 * VAR26 * y + CONST280 * VAR16)
            + VAR08 * (CONST108 * VAR16 * VAR24 + CONST287 * VAR22 * y)
        )
        + g_17
        * (
            CONST009
            * VAR17
            * (
                CONST048 * VAR21
                + CONST125 * VAR08 * VAR23
                - CONST256 * VAR06 * VAR25
                + CONST277 * VAR04 * z
            )
            + CONST059 * VAR02 * z
            + CONST296 * VAR04 * VAR25
            - CONST318 * VAR08 * VAR21
            + CONST334 * VAR06 * VAR23
            + CONST386 * VAR19
        )
        + g_19
        * (
            CONST014 * VAR19
            + CONST062 * VAR02 * z
            + CONST203 * VAR04 * VAR25
            + CONST281 * VAR06 * VAR23
            + CONST292 * VAR08 * VAR21
        )
        + g_3
        * (
            CONST009
            * VAR17
            * (
                CONST144 * VAR22 * x
                + CONST256 * VAR07 * VAR24
                + CONST294 * VAR05 * VAR26
                + CONST366 * VAR03
            )
            + CONST122 * VAR07 * VAR22
            + CONST318 * VAR03 * VAR26
            - CONST334 * VAR05 * VAR24
            + CONST356 * VAR20 * x
            - CONST386 * VAR01
        )
        + g_4
        * (
            CONST248 * VAR03 * y * z
            + VAR05 * (CONST213 * VAR16 * z + CONST286 * VAR25 * y)
            + VAR07 * (CONST287 * VAR23 * y + CONST323 * VAR16 * VAR25)
            + x * (CONST213 * VAR16 * VAR23 + CONST247 * VAR21 * y)
        )
        + g_5
        * (
            CONST009
            * VAR17
            * (
                -CONST241 * VAR07 * VAR24
                + CONST241 * VAR22 * x
                + CONST243 * VAR05 * VAR26
                + CONST347 * VAR03
            )
            + CONST010
            * VAR15
            * (CONST083 * VAR05 + CONST101 * VAR07 * VAR26 - CONST223 * VAR24 * x)
            + CONST046 * VAR20 * x
            + CONST197 * VAR01
            + CONST332 * VAR05 * VAR24
            + CONST353 * VAR03 * VAR26
        )
        + g_6
        * (
            CONST275 * VAR03 * y * z
            + VAR05 * (CONST274 * VAR25 * y - CONST302 * VAR16 * z)
            + VAR07 * (CONST146 * VAR23 * y + CONST302 * VAR14 * z)
            + x
            * (
                CONST146 * VAR21 * y
                - CONST302 * VAR14 * VAR25
                + CONST302 * VAR16 * VAR23
            )
        )
        + g_7
        * (
            CONST009
            * VAR17
            * (
                CONST087 * VAR05 * VAR26
                - CONST209 * VAR07 * VAR24
                - CONST266 * VAR22 * x
                + CONST336 * VAR03
            )
            + CONST010
            * VAR15
            * (CONST186 * VAR24 * x + CONST237 * VAR07 * VAR26 - CONST298 * VAR05)
            + CONST011 * VAR13 * (-CONST290 * VAR26 * x + CONST345 * VAR07)
            + CONST340 * VAR01
            + CONST350 * VAR07 * VAR22
            + CONST358 * VAR05 * VAR24
            + CONST374 * VAR20 * x
        )
        + g_8
        * (
            CONST311 * VAR03 * y * z
            + VAR05 * (CONST206 * VAR16 * z + CONST216 * VAR25 * y)
            + VAR07
            * (CONST028 * VAR16 * VAR25 + CONST216 * VAR23 * y + CONST226 * VAR14 * z)
            + x
            * (
                CONST206 * VAR16 * VAR23
                + CONST226 * VAR14 * VAR25
                + CONST259 * VAR12 * z
                + CONST311 * VAR21 * y
            )
        )
        + g_9
        * (
            CONST015 * VAR01
            + VAR03 * (CONST042 * VAR26 + CONST253 * VAR17)
            + VAR05 * (CONST033 * VAR17 * VAR26 + CONST058 * VAR24 + CONST155 * VAR15)
            + VAR07
            * (
                CONST032 * VAR17 * VAR24
                + CONST042 * VAR22
                + CONST235 * VAR15 * VAR26
                + CONST361 * VAR13
            )
            + x
            * (
                CONST015 * VAR20
                + CONST155 * VAR15 * VAR24
                + CONST253 * VAR17 * VAR22
                - CONST314 * VAR11
                + CONST361 * VAR13 * VAR26
            )
        )
    )
    g_z = (
        g_0
        * (
            CONST093 * VAR20 * x
            + CONST210 * VAR03 * VAR26
            + CONST250 * VAR05 * VAR24
            + CONST328 * VAR07 * VAR22
            - CONST378 * VAR01
        )
        + g_1
        * y
        * (
            -CONST018 * VAR05 * VAR25
            + CONST018 * VAR07 * VAR23
            + CONST224 * VAR03 * z
            - CONST224 * VAR21 * x
        )
        + g_10
        * (
            CONST095 * VAR15 * VAR23
            + CONST132 * VAR17 * VAR21
            + CONST265 * VAR13 * VAR25
            + CONST333 * VAR11 * z
            + CONST391 * VAR19
            + CONST398 * VAR02 * z
            + VAR04 * (CONST131 * VAR17 * z + CONST376 * VAR25)
            + VAR06
            * (CONST094 * VAR15 * z + CONST246 * VAR17 * VAR25 + CONST369 * VAR23)
            + VAR08
            * (
                CONST137 * VAR15 * VAR25
                + CONST246 * VAR17 * VAR23
                + CONST265 * VAR13 * z
                + CONST375 * VAR21
            )
        )
        + g_11
        * (
            CONST009
            * VAR26
            * (
                CONST042 * VAR04 * y
                + CONST211 * VAR08 * VAR14
                + CONST251 * VAR06 * VAR16
                + CONST313 * VAR12
            )
            + CONST010
            * VAR24
            * (CONST058 * VAR06 * y + CONST142 * VAR14 + CONST252 * VAR08 * VAR16)
            + CONST011 * VAR22 * (CONST042 * VAR08 * y + CONST331 * VAR16)
            + CONST015 * VAR02 * y
            + CONST026 * VAR10
            + CONST076 * VAR20 * y
            + CONST142 * VAR06 * VAR14
            + CONST314 * VAR08 * VAR12
            + CONST331 * VAR04 * VAR16
        )
        + g_12
        * (
            CONST050 * VAR02 * z
            + CONST082 * VAR11 * z
            + CONST097 * VAR15 * VAR23
            + CONST120 * VAR13 * VAR25
            + CONST262 * VAR17 * VAR21
            - CONST385 * VAR19
            + VAR04 * (CONST273 * VAR25 - CONST311 * VAR17 * z)
            + VAR06 * (CONST017 * VAR23 + CONST238 * VAR15 * z)
            + VAR08
            * (CONST029 * VAR21 - CONST140 * VAR15 * VAR25 + CONST217 * VAR17 * VAR23)
        )
        + g_13
        * (
            VAR12 * (CONST290 * VAR08 - CONST290 * VAR26)
            + VAR14 * (CONST049 * VAR24 - CONST186 * VAR06 - CONST307 * VAR08 * VAR26)
            + VAR16
            * (
                -CONST164 * VAR22
                + CONST209 * VAR08 * VAR24
                + CONST219 * VAR06 * VAR26
                + CONST266 * VAR04
            )
            + y
            * (
                -CONST285 * VAR06 * VAR24
                - CONST297 * VAR04 * VAR26
                + CONST346 * VAR20
                - CONST374 * VAR02
            )
        )
        + g_14
        * (
            CONST104 * VAR02 * z
            + CONST114 * VAR15 * VAR23
            + CONST146 * VAR17 * VAR21
            + CONST194 * VAR19
            - CONST239 * VAR13 * VAR25
            + VAR04 * (CONST274 * VAR17 * z - CONST362 * VAR25)
            + VAR06
            * (CONST072 * VAR23 + CONST171 * VAR15 * z + CONST240 * VAR17 * VAR25)
            + VAR08
            * (
                CONST030 * VAR21
                + CONST114 * VAR17 * VAR23
                - CONST148 * VAR15 * VAR25
                + CONST338 * VAR13 * z
            )
        )
        + g_15
        * (
            VAR14 * (CONST185 * VAR08 * VAR26 - CONST222 * VAR24 - CONST223 * VAR06)
            + VAR16
            * (
                CONST079 * VAR06 * VAR26
                + CONST134 * VAR08 * VAR24
                + CONST202 * VAR22
                + CONST241 * VAR04
            )
            + y
            * (
                CONST046 * VAR02
                + CONST073 * VAR20
                + CONST195 * VAR06 * VAR24
                + CONST223 * VAR08 * VAR22
            )
        )
        + g_16
        * (
            CONST022 * VAR19
            + CONST035 * VAR02 * z
            + CONST175 * VAR15 * VAR23
            + CONST291 * VAR17 * VAR21
            + VAR04 * (CONST057 * VAR25 + CONST135 * VAR17 * z)
            + VAR06 * (CONST341 * VAR15 * z + CONST346 * VAR23)
            + VAR08
            * (CONST108 * VAR15 * VAR25 + CONST158 * VAR17 * VAR23 + CONST337 * VAR21)
        )
        + g_17
        * (
            VAR16
            * (
                -CONST044 * VAR06 * VAR26
                + CONST044 * VAR08 * VAR24
                + CONST144 * VAR22
                + CONST277 * VAR04
            )
            + y
            * (
                -CONST016 * VAR08 * VAR22
                + CONST059 * VAR02
                + CONST180 * VAR04 * VAR26
                + CONST205 * VAR06 * VAR24
                + CONST351 * VAR20
            )
        )
        + g_18
        * (
            CONST061 * VAR02 * z
            + CONST127 * VAR08 * VAR21
            + CONST284 * VAR06 * VAR23
            + CONST306 * VAR04 * VAR25
            + CONST381 * VAR19
            + VAR17
            * (
                CONST039 * VAR04 * z
                + CONST081 * VAR08 * VAR23
                + CONST316 * VAR06 * VAR25
                - CONST319 * VAR21
            )
        )
        + g_19
        * y
        * (
            CONST062 * VAR02
            + CONST063 * VAR20
            + CONST204 * VAR04 * VAR26
            + CONST204 * VAR08 * VAR22
            + CONST279 * VAR06 * VAR24
        )
        + g_2
        * (
            CONST151 * VAR01
            + CONST162 * VAR07 * VAR22
            + CONST319 * VAR03 * VAR26
            + CONST348 * VAR20 * x
            + VAR17
            * (
                -CONST040 * VAR22 * x
                - CONST081 * VAR05 * VAR26
                + CONST103 * VAR07 * VAR24
                + CONST319 * VAR03
            )
        )
        + g_20
        * (
            -CONST163 * VAR06 * VAR23
            + CONST212 * VAR08 * VAR21
            - CONST327 * VAR02 * z
            + CONST329 * VAR04 * VAR25
            - CONST378 * VAR19
        )
        + g_3
        * (
            VAR16
            * (-CONST183 * VAR23 * x + CONST228 * VAR05 * z + CONST267 * VAR07 * VAR25)
            + y
            * (
                CONST116 * VAR07 * VAR23
                - CONST234 * VAR05 * VAR25
                + CONST234 * VAR21 * x
                + CONST268 * VAR03 * z
            )
        )
        + g_4
        * (
            CONST008 * VAR01
            + VAR03 * (CONST303 * VAR17 + CONST377 * VAR26)
            + VAR05 * (CONST175 * VAR15 - CONST307 * VAR17 * VAR26 + CONST326 * VAR24)
            + VAR07
            * (CONST108 * VAR15 * VAR26 + CONST341 * VAR17 * VAR24 + CONST359 * VAR22)
            + x
            * (CONST053 * VAR20 + CONST307 * VAR17 * VAR22 + CONST341 * VAR15 * VAR24)
        )
        + g_5
        * (
            VAR14 * (CONST147 * VAR07 * z - CONST147 * VAR25 * x)
            + VAR16
            * (CONST154 * VAR05 * z + CONST190 * VAR07 * VAR25 + CONST310 * VAR23 * x)
            + y
            * (CONST156 * VAR21 * x + CONST222 * VAR05 * VAR25 + CONST325 * VAR03 * z)
        )
        + g_6
        * (
            CONST177 * VAR01
            + VAR03 * (CONST030 * VAR26 + CONST321 * VAR17)
            + VAR05 * (-CONST193 * VAR15 + CONST229 * VAR17 * VAR26)
            + VAR07 * (CONST239 * VAR13 + CONST258 * VAR17 * VAR24 + CONST362 * VAR22)
            + x
            * (
                CONST148 * VAR15 * VAR24
                - CONST338 * VAR13 * VAR26
                + CONST357 * VAR17 * VAR22
                + CONST372 * VAR20
            )
        )
        + g_7
        * (
            -CONST221 * VAR12 * x * z
            + VAR14 * (CONST136 * VAR07 * z + CONST260 * VAR25 * x)
            + VAR16
            * (CONST119 * VAR05 * z - CONST145 * VAR23 * x + CONST342 * VAR07 * VAR25)
            + y
            * (
                CONST237 * VAR07 * VAR23
                + CONST297 * VAR05 * VAR25
                + CONST298 * VAR21 * x
            )
        )
        + g_8
        * (
            -CONST397 * VAR01
            + VAR03 * (CONST031 * VAR26 + CONST344 * VAR17)
            + VAR05 * (CONST055 * VAR24 + CONST160 * VAR17 * VAR26 + CONST173 * VAR15)
            + VAR07
            * (
                CONST051 * VAR22
                + CONST143 * VAR15 * VAR26
                + CONST231 * VAR13
                + CONST322 * VAR17 * VAR24
            )
            + x
            * (
                CONST024 * VAR20
                + CONST082 * VAR11
                + CONST196 * VAR17 * VAR22
                + CONST295 * VAR13 * VAR26
                + CONST330 * VAR15 * VAR24
            )
        )
        + g_9
        * (
            CONST070 * VAR03 * y * z
            + VAR05 * (CONST121 * VAR25 * y + CONST168 * VAR16 * z)
            + VAR07
            * (CONST121 * VAR23 * y + CONST261 * VAR16 * VAR25 - CONST361 * VAR14 * z)
            + x
            * (
                CONST070 * VAR21 * y
                + CONST167 * VAR16 * VAR23
                + CONST264 * VAR12 * z
                - CONST361 * VAR14 * VAR25
            )
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
