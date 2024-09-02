import triton
import torch
from triton import language as tl

from equitriton.utils import calculate_lastdim_num_blocks

__all__ = ["NinthOrderSphericalHarmonic"]


class NinthOrderSphericalHarmonic(torch.autograd.Function):
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
                (*coords.shape[:-1], 19), dtype=coords.dtype, device=coords.device
            )
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        # apply the kernel
        ninth_order_fwd[num_blocks,](
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
        ninth_order_bwd[num_blocks,](
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
    CONST000 = 1.93163963757558
    CONST001 = 2.65478475211798
    CONST002 = 1.72771101506082
    CONST004 = 1.59908344719522
    CONST005 = 6.39633378878088
    CONST006 = 6.39633378878088
    CONST007 = 8.63855507530412
    CONST008 = 9.59450068317133
    CONST009 = 4.35889894354067
    CONST010 = 10.7269778688696
    CONST011 = 10.7269778688696
    CONST012 = 6.39633378878088
    CONST013 = 15.0007324039945
    CONST014 = 13.0937127087774
    CONST016 = 14.4550674370400
    CONST017 = 14.4550674370400
    CONST018 = 13.3827919767794
    CONST019 = 13.5214774630291
    CONST020 = 23.8930627690618
    CONST021 = 27.0429549260581
    CONST022 = 29.2403830344269
    CONST023 = 29.2403830344269
    CONST024 = 30.0014648079890
    CONST025 = -480.023436927823
    CONST026 = -480.023436927823
    CONST029 = 42.9079114754785
    CONST030 = -462.562157985281
    CONST032 = -967.518168434061
    CONST034 = 57.8202697481601
    CONST035 = 58.9217071894985
    CONST036 = 58.9217071894985
    CONST037 = 62.4530292249704
    CONST038 = 1081.71819704233
    CONST039 = 64.3618672132178
    CONST040 = 578.202697481601
    CONST044 = 600.029296159779
    CONST045 = -936.795438374555
    CONST047 = 96.7518168434061
    CONST049 = 115.640539496320
    CONST051 = -392.811381263323
    CONST053 = 137.149553407950
    CONST055 = 150.007324039945
    CONST056 = -343.263291803828
    CONST058 = 11.2632978048796
    CONST061 = -315.372338536630
    CONST062 = -314.249105010659
    CONST063 = 205.957975082297
    CONST065 = -294.608535947493
    CONST066 = 240.011718463912
    CONST068 = 241.879542108515
    CONST069 = 255.853351551235
    CONST070 = 255.853351551235
    CONST071 = -241.879542108515
    CONST072 = -240.011718463912
    CONST073 = -241.879542108515
    CONST074 = 788.430846341574
    CONST075 = 1.72771101506082
    CONST076 = -1.93163963757558
    CONST077 = -1249.06058449941
    CONST078 = -223.001919177910
    CONST080 = -216.343639408465
    CONST081 = 300.014648079890
    CONST082 = -204.682681240988
    CONST083 = -204.682681240988
    CONST084 = -204.682681240988
    CONST086 = -196.405690631662
    CONST087 = -191.890013663426
    CONST088 = -191.890013663427
    CONST089 = -187.359087674911
    CONST090 = -693.843236977922
    CONST091 = 334.502878766866
    CONST092 = -176.765121568496
    CONST093 = -150.007324039945
    CONST094 = -144.550674370400
    CONST095 = 374.718175349822
    CONST096 = 374.718175349822
    CONST097 = -649.030918225395
    CONST099 = -630.744677073259
    CONST100 = -115.640539496320
    CONST101 = -114.421097267943
    CONST102 = -115.640539496320
    CONST103 = -104.749701670220
    CONST104 = 411.915950164594
    CONST105 = -95.5722510762473
    CONST106 = -90.1063824390370
    CONST107 = -90.0043944239669
    CONST109 = -80.2967518606762
    CONST110 = -78.4601809837321
    CONST111 = 435.383175795327
    CONST112 = -589.217071894985
    CONST113 = -78.4601809837321
    CONST114 = 435.383175795328
    CONST115 = -68.5747767039748
    CONST116 = -63.9633378878088
    CONST117 = -63.9633378878088
    CONST118 = -62.4530292249704
    CONST119 = -58.9217071894985
    CONST120 = -1081.71819704233
    CONST121 = -57.8202697481601
    CONST122 = -57.8202697481601
    CONST123 = -58.9217071894985
    CONST124 = -54.0859098521163
    CONST125 = 462.562157985281
    CONST127 = -48.3759084217031
    CONST128 = -48.3759084217030
    CONST129 = -38.6327927515116
    CONST130 = -30.9062342012093
    CONST131 = 483.759084217031
    CONST132 = -30.0014648079890
    CONST133 = -30.0014648079890
    CONST134 = -27.0429549260581
    CONST135 = -24.1879542108515
    CONST136 = -24.1879542108515
    CONST137 = -1.63671408859718
    CONST138 = -15.0007324039945
    CONST139 = -13.5214774630291
    CONST140 = -13.8216881204866
    CONST141 = -13.0937127087774
    CONST142 = -13.3827919767794
    CONST143 = -9.82028453158308
    CONST144 = -4.91014226579154
    CONST145 = 511.706703102471
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR01 = VAR07 * VAR07 * VAR07
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR10 = VAR16 * VAR16 * VAR16
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR19 = VAR25 * VAR25 * VAR25
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    # -------------------- kernel implementations
    Y00 = (
        CONST001 * VAR01
        + CONST020 * VAR20 * x
        + CONST078 * VAR07 * VAR22
        + CONST091 * VAR05 * VAR24
        + CONST105 * VAR03 * VAR26
    )
    Y01 = y * (
        -CONST099 * VAR05 * VAR25
        + CONST099 * VAR07 * VAR23
        + CONST106 * VAR03 * z
        - CONST106 * VAR21 * x
    )
    Y02 = (
        CONST000 * VAR01
        + VAR03 * (CONST129 * VAR26 + CONST130 * VAR17)
        + VAR05 * (CONST021 * VAR24 - CONST097 * VAR17 * VAR26)
        + VAR07 * (CONST120 * VAR17 * VAR24 - CONST124 * VAR22)
        + x * (-CONST080 * VAR17 * VAR22 + CONST139 * VAR20)
    )
    Y03 = VAR16 * (
        CONST077 * VAR07 * VAR25 + CONST095 * VAR05 * z + CONST096 * VAR23 * x
    ) + y * (
        -CONST089 * VAR05 * VAR25
        - CONST089 * VAR07 * VAR23
        + CONST109 * VAR03 * z
        + CONST109 * VAR21 * x
    )
    Y04 = (
        CONST002 * VAR01
        + CONST007 * VAR20 * x
        + CONST135 * VAR05 * VAR24
        + CONST140 * VAR03 * VAR26
        + VAR15 * (CONST032 * VAR07 * VAR26 + CONST047 * VAR05 + CONST131 * VAR24 * x)
        + VAR17
        * (
            -CONST071 * VAR07 * VAR24
            + CONST071 * VAR22 * x
            + CONST111 * VAR05 * VAR26
            + CONST127 * VAR03
        )
    )
    Y05 = (
        VAR14 * (CONST030 * VAR07 * z - CONST030 * VAR25 * x)
        + VAR16 * (CONST030 * VAR23 * x + CONST125 * VAR05 * z)
        + y
        * (
            CONST034 * VAR07 * VAR23
            + CONST121 * VAR05 * VAR25
            - CONST121 * VAR21 * x
            + CONST122 * VAR03 * z
        )
    )
    Y06 = (
        CONST119 * VAR03 * VAR17
        - CONST137 * VAR01
        + VAR05 * (CONST035 * VAR17 * VAR26 - CONST086 * VAR15 + CONST143 * VAR24)
        + VAR07
        * (
            CONST051 * VAR15 * VAR26
            - CONST065 * VAR17 * VAR24
            + CONST103 * VAR13
            + CONST141 * VAR22
        )
        + x
        * (
            -CONST062 * VAR13 * VAR26
            - CONST092 * VAR17 * VAR22
            + CONST112 * VAR15 * VAR24
            + CONST144 * VAR20
        )
    )
    Y07 = (
        CONST132 * VAR03 * y * z
        + VAR05 * (CONST081 * VAR16 * z + CONST107 * VAR25 * y)
        + VAR07
        * (CONST026 * VAR14 * z + CONST044 * VAR16 * VAR25 + CONST107 * VAR23 * y)
        + x
        * (
            CONST025 * VAR14 * VAR25
            + CONST053 * VAR12 * z
            + CONST081 * VAR16 * VAR23
            + CONST132 * VAR21 * y
        )
    )
    Y08 = (
        CONST004 * VAR01
        + VAR03 * (CONST006 * VAR26 + CONST116 * VAR17)
        + VAR05 * (CONST008 * VAR24 + CONST069 * VAR15 + CONST087 * VAR17 * VAR26)
        + VAR07
        * (
            CONST005 * VAR22
            + CONST083 * VAR13
            + CONST087 * VAR17 * VAR24
            + CONST145 * VAR15 * VAR26
        )
        + x
        * (
            CONST004 * VAR20
            + CONST022 * VAR11
            + CONST069 * VAR15 * VAR24
            + CONST082 * VAR13 * VAR26
            + CONST116 * VAR17 * VAR22
        )
    )
    Y09 = (
        CONST009 * VAR10
        + VAR12 * (CONST110 * VAR26 + CONST113 * VAR08)
        + VAR14 * (CONST063 * VAR06 + CONST063 * VAR24 + CONST104 * VAR08 * VAR26)
        + VAR16
        * (
            CONST056 * VAR06 * VAR26
            + CONST056 * VAR08 * VAR24
            + CONST101 * VAR04
            + CONST101 * VAR22
        )
        + y
        * (
            CONST010 * VAR20
            + CONST011 * VAR02
            + CONST029 * VAR04 * VAR26
            + CONST029 * VAR08 * VAR22
            + CONST039 * VAR06 * VAR24
        )
    )
    Y10 = (
        CONST004 * VAR19
        + VAR21 * (CONST005 * VAR08 + CONST117 * VAR17)
        + VAR23 * (CONST008 * VAR06 + CONST070 * VAR15 + CONST088 * VAR08 * VAR17)
        + VAR25
        * (
            CONST012 * VAR04
            + CONST082 * VAR13
            + CONST087 * VAR06 * VAR17
            + CONST145 * VAR08 * VAR15
        )
        + z
        * (
            CONST004 * VAR02
            + CONST023 * VAR11
            + CONST070 * VAR06 * VAR15
            + CONST084 * VAR08 * VAR13
            + CONST117 * VAR04 * VAR17
        )
    )
    Y11 = (
        VAR12 * (CONST115 * VAR08 - CONST115 * VAR26)
        + VAR14 * (CONST066 * VAR06 + CONST072 * VAR24)
        + VAR16
        * (
            CONST055 * VAR08 * VAR24
            + CONST093 * VAR04
            + CONST093 * VAR06 * VAR26
            - CONST093 * VAR22
        )
        + y
        * (
            CONST013 * VAR02
            + CONST024 * VAR04 * VAR26
            + CONST133 * VAR08 * VAR22
            + CONST138 * VAR20
        )
    )
    Y12 = (
        CONST036 * VAR17 * VAR21
        + CONST137 * VAR19
        + VAR23 * (CONST086 * VAR15 + CONST123 * VAR08 * VAR17 - CONST143 * VAR06)
        + VAR25
        * (
            CONST014 * VAR04
            - CONST051 * VAR08 * VAR15
            + CONST065 * VAR06 * VAR17
            - CONST103 * VAR13
        )
        + z
        * (
            CONST062 * VAR08 * VAR13
            + CONST092 * VAR04 * VAR17
            - CONST112 * VAR06 * VAR15
            - CONST144 * VAR02
        )
    )
    Y13 = (
        VAR14 * (CONST049 * VAR06 + CONST049 * VAR24 + CONST090 * VAR08 * VAR26)
        + VAR16
        * (
            CONST040 * VAR06 * VAR26
            + CONST040 * VAR08 * VAR24
            + CONST100 * VAR22
            + CONST102 * VAR04
        )
        + y
        * (
            CONST016 * VAR20
            + CONST017 * VAR02
            + CONST094 * VAR06 * VAR24
            + CONST121 * VAR04 * VAR26
            + CONST122 * VAR08 * VAR22
        )
    )
    Y14 = (
        CONST007 * VAR02 * z
        + CONST075 * VAR19
        + CONST136 * VAR06 * VAR23
        + CONST140 * VAR08 * VAR21
        + VAR15 * (CONST032 * VAR08 * VAR25 + CONST047 * VAR23 + CONST131 * VAR06 * z)
        + VAR17
        * (
            CONST068 * VAR06 * VAR25
            + CONST073 * VAR04 * z
            + CONST114 * VAR08 * VAR23
            + CONST128 * VAR21
        )
    )
    Y15 = VAR16 * (
        CONST037 * VAR22
        - CONST045 * VAR06 * VAR26
        + CONST045 * VAR08 * VAR24
        + CONST118 * VAR04
    ) + y * (
        CONST018 * VAR02
        + CONST089 * VAR04 * VAR26
        - CONST089 * VAR08 * VAR22
        + CONST142 * VAR20
    )
    Y16 = (
        CONST019 * VAR02 * z
        + CONST076 * VAR19
        + CONST124 * VAR04 * VAR25
        - CONST129 * VAR08 * VAR21
        + CONST134 * VAR06 * VAR23
        + VAR17
        * (
            CONST038 * VAR06 * VAR25
            + CONST080 * VAR04 * z
            + CONST097 * VAR08 * VAR23
            - CONST130 * VAR21
        )
    )
    Y17 = y * (
        CONST058 * VAR02
        + CONST058 * VAR20
        + CONST061 * VAR04 * VAR26
        + CONST061 * VAR08 * VAR22
        + CONST074 * VAR06 * VAR24
    )
    Y18 = (
        CONST001 * VAR19
        + CONST020 * VAR02 * z
        + CONST078 * VAR04 * VAR25
        + CONST091 * VAR06 * VAR23
        + CONST105 * VAR08 * VAR21
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
    ]
    return torch.cat(tensors, dim=-1)


@triton.jit
def ninth_order_fwd(
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
    CONST000 = 1.93163963757558
    CONST001 = 2.65478475211798
    CONST002 = 1.72771101506082
    CONST004 = 1.59908344719522
    CONST005 = 6.39633378878088
    CONST006 = 6.39633378878088
    CONST007 = 8.63855507530412
    CONST008 = 9.59450068317133
    CONST009 = 4.35889894354067
    CONST010 = 10.7269778688696
    CONST011 = 10.7269778688696
    CONST012 = 6.39633378878088
    CONST013 = 15.0007324039945
    CONST014 = 13.0937127087774
    CONST016 = 14.4550674370400
    CONST017 = 14.4550674370400
    CONST018 = 13.3827919767794
    CONST019 = 13.5214774630291
    CONST020 = 23.8930627690618
    CONST021 = 27.0429549260581
    CONST022 = 29.2403830344269
    CONST023 = 29.2403830344269
    CONST024 = 30.0014648079890
    CONST025 = -480.023436927823
    CONST026 = -480.023436927823
    CONST029 = 42.9079114754785
    CONST030 = -462.562157985281
    CONST032 = -967.518168434061
    CONST034 = 57.8202697481601
    CONST035 = 58.9217071894985
    CONST036 = 58.9217071894985
    CONST037 = 62.4530292249704
    CONST038 = 1081.71819704233
    CONST039 = 64.3618672132178
    CONST040 = 578.202697481601
    CONST044 = 600.029296159779
    CONST045 = -936.795438374555
    CONST047 = 96.7518168434061
    CONST049 = 115.640539496320
    CONST051 = -392.811381263323
    CONST053 = 137.149553407950
    CONST055 = 150.007324039945
    CONST056 = -343.263291803828
    CONST058 = 11.2632978048796
    CONST061 = -315.372338536630
    CONST062 = -314.249105010659
    CONST063 = 205.957975082297
    CONST065 = -294.608535947493
    CONST066 = 240.011718463912
    CONST068 = 241.879542108515
    CONST069 = 255.853351551235
    CONST070 = 255.853351551235
    CONST071 = -241.879542108515
    CONST072 = -240.011718463912
    CONST073 = -241.879542108515
    CONST074 = 788.430846341574
    CONST075 = 1.72771101506082
    CONST076 = -1.93163963757558
    CONST077 = -1249.06058449941
    CONST078 = -223.001919177910
    CONST080 = -216.343639408465
    CONST081 = 300.014648079890
    CONST082 = -204.682681240988
    CONST083 = -204.682681240988
    CONST084 = -204.682681240988
    CONST086 = -196.405690631662
    CONST087 = -191.890013663426
    CONST088 = -191.890013663427
    CONST089 = -187.359087674911
    CONST090 = -693.843236977922
    CONST091 = 334.502878766866
    CONST092 = -176.765121568496
    CONST093 = -150.007324039945
    CONST094 = -144.550674370400
    CONST095 = 374.718175349822
    CONST096 = 374.718175349822
    CONST097 = -649.030918225395
    CONST099 = -630.744677073259
    CONST100 = -115.640539496320
    CONST101 = -114.421097267943
    CONST102 = -115.640539496320
    CONST103 = -104.749701670220
    CONST104 = 411.915950164594
    CONST105 = -95.5722510762473
    CONST106 = -90.1063824390370
    CONST107 = -90.0043944239669
    CONST109 = -80.2967518606762
    CONST110 = -78.4601809837321
    CONST111 = 435.383175795327
    CONST112 = -589.217071894985
    CONST113 = -78.4601809837321
    CONST114 = 435.383175795328
    CONST115 = -68.5747767039748
    CONST116 = -63.9633378878088
    CONST117 = -63.9633378878088
    CONST118 = -62.4530292249704
    CONST119 = -58.9217071894985
    CONST120 = -1081.71819704233
    CONST121 = -57.8202697481601
    CONST122 = -57.8202697481601
    CONST123 = -58.9217071894985
    CONST124 = -54.0859098521163
    CONST125 = 462.562157985281
    CONST127 = -48.3759084217031
    CONST128 = -48.3759084217030
    CONST129 = -38.6327927515116
    CONST130 = -30.9062342012093
    CONST131 = 483.759084217031
    CONST132 = -30.0014648079890
    CONST133 = -30.0014648079890
    CONST134 = -27.0429549260581
    CONST135 = -24.1879542108515
    CONST136 = -24.1879542108515
    CONST137 = -1.63671408859718
    CONST138 = -15.0007324039945
    CONST139 = -13.5214774630291
    CONST140 = -13.8216881204866
    CONST141 = -13.0937127087774
    CONST142 = -13.3827919767794
    CONST143 = -9.82028453158308
    CONST144 = -4.91014226579154
    CONST145 = 511.706703102471
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR01 = VAR07 * VAR07 * VAR07
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR10 = VAR16 * VAR16 * VAR16
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR19 = VAR25 * VAR25 * VAR25
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    # -------------------- kernel implementations
    Y00 = (
        CONST001 * VAR01
        + CONST020 * VAR20 * x
        + CONST078 * VAR07 * VAR22
        + CONST091 * VAR05 * VAR24
        + CONST105 * VAR03 * VAR26
    )
    Y01 = y * (
        -CONST099 * VAR05 * VAR25
        + CONST099 * VAR07 * VAR23
        + CONST106 * VAR03 * z
        - CONST106 * VAR21 * x
    )
    Y02 = (
        CONST000 * VAR01
        + VAR03 * (CONST129 * VAR26 + CONST130 * VAR17)
        + VAR05 * (CONST021 * VAR24 - CONST097 * VAR17 * VAR26)
        + VAR07 * (CONST120 * VAR17 * VAR24 - CONST124 * VAR22)
        + x * (-CONST080 * VAR17 * VAR22 + CONST139 * VAR20)
    )
    Y03 = VAR16 * (
        CONST077 * VAR07 * VAR25 + CONST095 * VAR05 * z + CONST096 * VAR23 * x
    ) + y * (
        -CONST089 * VAR05 * VAR25
        - CONST089 * VAR07 * VAR23
        + CONST109 * VAR03 * z
        + CONST109 * VAR21 * x
    )
    Y04 = (
        CONST002 * VAR01
        + CONST007 * VAR20 * x
        + CONST135 * VAR05 * VAR24
        + CONST140 * VAR03 * VAR26
        + VAR15 * (CONST032 * VAR07 * VAR26 + CONST047 * VAR05 + CONST131 * VAR24 * x)
        + VAR17
        * (
            -CONST071 * VAR07 * VAR24
            + CONST071 * VAR22 * x
            + CONST111 * VAR05 * VAR26
            + CONST127 * VAR03
        )
    )
    Y05 = (
        VAR14 * (CONST030 * VAR07 * z - CONST030 * VAR25 * x)
        + VAR16 * (CONST030 * VAR23 * x + CONST125 * VAR05 * z)
        + y
        * (
            CONST034 * VAR07 * VAR23
            + CONST121 * VAR05 * VAR25
            - CONST121 * VAR21 * x
            + CONST122 * VAR03 * z
        )
    )
    Y06 = (
        CONST119 * VAR03 * VAR17
        - CONST137 * VAR01
        + VAR05 * (CONST035 * VAR17 * VAR26 - CONST086 * VAR15 + CONST143 * VAR24)
        + VAR07
        * (
            CONST051 * VAR15 * VAR26
            - CONST065 * VAR17 * VAR24
            + CONST103 * VAR13
            + CONST141 * VAR22
        )
        + x
        * (
            -CONST062 * VAR13 * VAR26
            - CONST092 * VAR17 * VAR22
            + CONST112 * VAR15 * VAR24
            + CONST144 * VAR20
        )
    )
    Y07 = (
        CONST132 * VAR03 * y * z
        + VAR05 * (CONST081 * VAR16 * z + CONST107 * VAR25 * y)
        + VAR07
        * (CONST026 * VAR14 * z + CONST044 * VAR16 * VAR25 + CONST107 * VAR23 * y)
        + x
        * (
            CONST025 * VAR14 * VAR25
            + CONST053 * VAR12 * z
            + CONST081 * VAR16 * VAR23
            + CONST132 * VAR21 * y
        )
    )
    Y08 = (
        CONST004 * VAR01
        + VAR03 * (CONST006 * VAR26 + CONST116 * VAR17)
        + VAR05 * (CONST008 * VAR24 + CONST069 * VAR15 + CONST087 * VAR17 * VAR26)
        + VAR07
        * (
            CONST005 * VAR22
            + CONST083 * VAR13
            + CONST087 * VAR17 * VAR24
            + CONST145 * VAR15 * VAR26
        )
        + x
        * (
            CONST004 * VAR20
            + CONST022 * VAR11
            + CONST069 * VAR15 * VAR24
            + CONST082 * VAR13 * VAR26
            + CONST116 * VAR17 * VAR22
        )
    )
    Y09 = (
        CONST009 * VAR10
        + VAR12 * (CONST110 * VAR26 + CONST113 * VAR08)
        + VAR14 * (CONST063 * VAR06 + CONST063 * VAR24 + CONST104 * VAR08 * VAR26)
        + VAR16
        * (
            CONST056 * VAR06 * VAR26
            + CONST056 * VAR08 * VAR24
            + CONST101 * VAR04
            + CONST101 * VAR22
        )
        + y
        * (
            CONST010 * VAR20
            + CONST011 * VAR02
            + CONST029 * VAR04 * VAR26
            + CONST029 * VAR08 * VAR22
            + CONST039 * VAR06 * VAR24
        )
    )
    Y10 = (
        CONST004 * VAR19
        + VAR21 * (CONST005 * VAR08 + CONST117 * VAR17)
        + VAR23 * (CONST008 * VAR06 + CONST070 * VAR15 + CONST088 * VAR08 * VAR17)
        + VAR25
        * (
            CONST012 * VAR04
            + CONST082 * VAR13
            + CONST087 * VAR06 * VAR17
            + CONST145 * VAR08 * VAR15
        )
        + z
        * (
            CONST004 * VAR02
            + CONST023 * VAR11
            + CONST070 * VAR06 * VAR15
            + CONST084 * VAR08 * VAR13
            + CONST117 * VAR04 * VAR17
        )
    )
    Y11 = (
        VAR12 * (CONST115 * VAR08 - CONST115 * VAR26)
        + VAR14 * (CONST066 * VAR06 + CONST072 * VAR24)
        + VAR16
        * (
            CONST055 * VAR08 * VAR24
            + CONST093 * VAR04
            + CONST093 * VAR06 * VAR26
            - CONST093 * VAR22
        )
        + y
        * (
            CONST013 * VAR02
            + CONST024 * VAR04 * VAR26
            + CONST133 * VAR08 * VAR22
            + CONST138 * VAR20
        )
    )
    Y12 = (
        CONST036 * VAR17 * VAR21
        + CONST137 * VAR19
        + VAR23 * (CONST086 * VAR15 + CONST123 * VAR08 * VAR17 - CONST143 * VAR06)
        + VAR25
        * (
            CONST014 * VAR04
            - CONST051 * VAR08 * VAR15
            + CONST065 * VAR06 * VAR17
            - CONST103 * VAR13
        )
        + z
        * (
            CONST062 * VAR08 * VAR13
            + CONST092 * VAR04 * VAR17
            - CONST112 * VAR06 * VAR15
            - CONST144 * VAR02
        )
    )
    Y13 = (
        VAR14 * (CONST049 * VAR06 + CONST049 * VAR24 + CONST090 * VAR08 * VAR26)
        + VAR16
        * (
            CONST040 * VAR06 * VAR26
            + CONST040 * VAR08 * VAR24
            + CONST100 * VAR22
            + CONST102 * VAR04
        )
        + y
        * (
            CONST016 * VAR20
            + CONST017 * VAR02
            + CONST094 * VAR06 * VAR24
            + CONST121 * VAR04 * VAR26
            + CONST122 * VAR08 * VAR22
        )
    )
    Y14 = (
        CONST007 * VAR02 * z
        + CONST075 * VAR19
        + CONST136 * VAR06 * VAR23
        + CONST140 * VAR08 * VAR21
        + VAR15 * (CONST032 * VAR08 * VAR25 + CONST047 * VAR23 + CONST131 * VAR06 * z)
        + VAR17
        * (
            CONST068 * VAR06 * VAR25
            + CONST073 * VAR04 * z
            + CONST114 * VAR08 * VAR23
            + CONST128 * VAR21
        )
    )
    Y15 = VAR16 * (
        CONST037 * VAR22
        - CONST045 * VAR06 * VAR26
        + CONST045 * VAR08 * VAR24
        + CONST118 * VAR04
    ) + y * (
        CONST018 * VAR02
        + CONST089 * VAR04 * VAR26
        - CONST089 * VAR08 * VAR22
        + CONST142 * VAR20
    )
    Y16 = (
        CONST019 * VAR02 * z
        + CONST076 * VAR19
        + CONST124 * VAR04 * VAR25
        - CONST129 * VAR08 * VAR21
        + CONST134 * VAR06 * VAR23
        + VAR17
        * (
            CONST038 * VAR06 * VAR25
            + CONST080 * VAR04 * z
            + CONST097 * VAR08 * VAR23
            - CONST130 * VAR21
        )
    )
    Y17 = y * (
        CONST058 * VAR02
        + CONST058 * VAR20
        + CONST061 * VAR04 * VAR26
        + CONST061 * VAR08 * VAR22
        + CONST074 * VAR06 * VAR24
    )
    Y18 = (
        CONST001 * VAR19
        + CONST020 * VAR02 * z
        + CONST078 * VAR04 * VAR25
        + CONST091 * VAR06 * VAR23
        + CONST105 * VAR08 * VAR21
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


@triton.jit
def ninth_order_bwd(
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
    # -------------------- variable and constant definitions
    CONST000 = 1.59908344719522
    CONST001 = 2.00000000000000
    CONST002 = 3.00000000000000
    CONST003 = 4.00000000000000
    CONST004 = 5.00000000000000
    CONST005 = 6.39633378878088
    CONST006 = 7.00000000000000
    CONST007 = 8.63855507530412
    CONST008 = 9.59450068317133
    CONST009 = 6.39633378878088
    CONST011 = 12.7926675775618
    CONST012 = 12.7926675775618
    CONST014 = 15.5493991355474
    CONST015 = 14.3917510247570
    CONST017 = 15.0007324039945
    CONST018 = 14.4550674370400
    CONST019 = 14.4550674370400
    CONST020 = 13.3827919767794
    CONST021 = 23.8930627690618
    CONST022 = 23.8930627690618
    CONST023 = 27.0429549260581
    CONST024 = 29.2403830344269
    CONST025 = 30.0014648079890
    CONST027 = 29.2403830344269
    CONST028 = 38.3780027326853
    CONST031 = 39.2300904918661
    CONST032 = 42.9079114754785
    CONST033 = 10.7269778688696
    CONST034 = 54.0859098521163
    CONST036 = 58.9217071894985
    CONST037 = 57.8202697481601
    CONST038 = 60.0029296159779
    CONST039 = 62.4530292249704
    CONST040 = 64.3618672132178
    CONST042 = 69.1084406024329
    CONST044 = 78.5622762526647
    CONST045 = 85.8158229509570
    CONST046 = 85.8158229509570
    CONST050 = 107.062335814235
    CONST052 = 108.171819704233
    CONST053 = -1935.03633686812
    CONST055 = 115.640539496320
    CONST056 = 117.843414378997
    CONST057 = 117.843414378997
    CONST059 = 120.005859231956
    CONST060 = 2176.91587897664
    CONST061 = 2176.91587897664
    CONST064 = 150.007324039945
    CONST065 = -1892.23403121978
    CONST066 = -1885.49463006395
    CONST067 = 173.460809244480
    CONST068 = -1873.59087674911
    CONST070 = 10.7269778688696
    CONST071 = 180.008788847934
    CONST074 = 13.5214774630291
    CONST076 = 205.957975082297
    CONST078 = 216.343639408465
    CONST079 = 4326.87278816930
    CONST080 = 233.923064275415
    CONST081 = 233.923064275415
    CONST082 = 240.011718463912
    CONST083 = 241.879542108515
    CONST085 = 255.853351551235
    CONST086 = 255.853351551235
    CONST087 = 257.447468852871
    CONST088 = 257.447468852871
    CONST090 = 270.429549260581
    CONST091 = 289.101348740801
    CONST093 = 300.014648079890
    CONST097 = 13.0937127087774
    CONST099 = -3747.18175349822
    CONST100 = 6.39633378878088
    CONST103 = 374.718175349822
    CONST105 = 404.741888237121
    CONST106 = 411.915950164594
    CONST107 = 412.451950326490
    CONST108 = 432.687278816930
    CONST109 = 435.383175795328
    CONST110 = 435.383175795327
    CONST112 = 462.562157985281
    CONST113 = -1571.24552505329
    CONST114 = 483.759084217031
    CONST115 = 511.706703102471
    CONST116 = 562.077263024733
    CONST117 = 578.202697481601
    CONST119 = -1451.27725265109
    CONST121 = -1451.27725265109
    CONST123 = 600.029296159779
    CONST124 = -1440.07031078347
    CONST129 = -1387.68647395584
    CONST130 = -1387.68647395584
    CONST131 = -1373.05316721531
    CONST132 = -1338.01151506746
    CONST133 = 725.638626325546
    CONST134 = -1298.06183645079
    CONST137 = 788.430846341574
    CONST138 = -1249.06058449941
    CONST139 = -1228.09608744593
    CONST140 = -1228.09608744593
    CONST141 = 823.831900329187
    CONST142 = -3245.15459112698
    CONST143 = -1178.43414378997
    CONST144 = 870.766351590655
    CONST145 = 870.766351590655
    CONST147 = -1124.15452604947
    CONST149 = -3153.72338536630
    CONST150 = 960.046873855647
    CONST151 = 960.046873855647
    CONST152 = 967.518168434061
    CONST153 = -1081.71819704233
    CONST154 = 967.518168434061
    CONST155 = -1060.59072941097
    CONST156 = 1023.41340620494
    CONST157 = 1023.41340620494
    CONST159 = -967.518168434061
    CONST160 = 1081.71819704233
    CONST161 = -960.046873855647
    CONST163 = -936.795438374555
    CONST165 = -900.043944239669
    CONST166 = 1156.40539496320
    CONST168 = -2902.55450530218
    CONST170 = 11.2632978048796
    CONST171 = -785.622762526647
    CONST172 = -785.622762526647
    CONST173 = -767.560054653706
    CONST175 = 1338.01151506746
    CONST176 = -693.843236977922
    CONST177 = -693.843236977921
    CONST178 = -686.526583607656
    CONST179 = -669.005757533731
    CONST180 = -669.005757533731
    CONST182 = -649.030918225395
    CONST183 = -630.744677073259
    CONST184 = -628.498210021318
    CONST185 = -628.498210021317
    CONST186 = -600.029296159779
    CONST187 = -589.217071894985
    CONST188 = -578.202697481601
    CONST189 = 15.5493991355474
    CONST190 = -562.077263024733
    CONST191 = 1500.07324039945
    CONST192 = -480.023436927823
    CONST193 = -480.023436927823
    CONST195 = -462.562157985281
    CONST196 = -450.021972119834
    CONST197 = -412.451950326490
    CONST198 = -409.365362481977
    CONST199 = -409.365362481976
    CONST200 = -404.741888237121
    CONST201 = -392.811381263323
    CONST202 = -383.780027326853
    CONST203 = -383.780027326853
    CONST204 = 1672.51439383433
    CONST205 = -374.718175349822
    CONST206 = -353.530243136991
    CONST207 = -2400.11718463912
    CONST209 = -346.921618488961
    CONST210 = -346.921618488961
    CONST211 = -343.263291803828
    CONST212 = -338.631358951921
    CONST213 = -338.631358951921
    CONST214 = -324.515459112698
    CONST215 = -315.372338536630
    CONST216 = -314.249105010659
    CONST217 = -2356.86828757994
    CONST218 = -300.014648079890
    CONST219 = -294.608535947493
    CONST220 = -289.101348740801
    CONST221 = -270.013183271901
    CONST222 = -2312.81078992641
    CONST223 = 1800.08788847934
    CONST224 = -241.879542108515
    CONST225 = -240.011718463912
    CONST226 = -241.879542108515
    CONST227 = -4326.87278816930
    CONST228 = -216.343639408465
    CONST229 = -210.010253655923
    CONST230 = -204.682681240988
    CONST231 = -204.682681240988
    CONST232 = -204.682681240988
    CONST233 = -196.405690631662
    CONST234 = -191.144502152495
    CONST235 = -191.890013663426
    CONST236 = -191.890013663427
    CONST237 = -187.359087674911
    CONST238 = -180.008788847934
    CONST239 = -176.765121568496
    CONST241 = 1873.59087674911
    CONST242 = -173.460809244480
    CONST244 = -162.257729556349
    CONST245 = -156.920361967464
    CONST246 = -156.920361967464
    CONST248 = -150.007324039945
    CONST249 = -144.550674370400
    CONST250 = -137.149553407950
    CONST251 = -135.214774630291
    CONST252 = -127.926675775618
    CONST253 = -127.926675775618
    CONST254 = -120.939771054258
    CONST255 = -120.005859231956
    CONST256 = -120.939771054258
    CONST257 = -117.843414378997
    CONST258 = -117.843414378997
    CONST259 = -115.640539496320
    CONST260 = -115.640539496320
    CONST261 = 1935.03633686812
    CONST262 = -2163.43639408465
    CONST263 = -114.421097267943
    CONST264 = -108.171819704233
    CONST265 = -107.062335814235
    CONST266 = -108.171819704233
    CONST267 = -104.749701670220
    CONST268 = -96.7518168434061
    CONST269 = -96.7518168434061
    CONST270 = -90.0043944239669
    CONST271 = -90.1063824390370
    CONST272 = -80.2967518606762
    CONST273 = -78.4601809837321
    CONST274 = -78.4601809837321
    CONST275 = -77.2655855030233
    CONST276 = -78.5622762526647
    CONST277 = -68.5747767039748
    CONST278 = -63.9633378878088
    CONST279 = -62.4530292249704
    CONST280 = -61.8124684024186
    CONST281 = -60.0029296159779
    CONST282 = -63.9633378878088
    CONST283 = -58.9217071894985
    CONST284 = -57.8202697481601
    CONST285 = -57.8202697481601
    CONST286 = -48.3759084217030
    CONST287 = -48.3759084217031
    CONST288 = -39.2811381263323
    CONST289 = -38.6327927515116
    CONST290 = -39.2811381263323
    CONST291 = -30.9062342012093
    CONST292 = -30.0014648079890
    CONST293 = -30.0014648079890
    CONST294 = -27.6433762409732
    CONST295 = -17.3847567381802
    CONST296 = -15.0007324039945
    CONST297 = -14.7304267973746
    CONST298 = -13.5214774630291
    CONST299 = -13.0937127087774
    CONST300 = -13.3827919767794
    CONST301 = -9.82028453158308
    CONST302 = -4.91014226579154
    CONST303 = 2046.82681240988
    VAR06 = x * x * x * x
    VAR07 = x * x * x
    VAR08 = x * x
    VAR02 = VAR06 * VAR06
    VAR03 = VAR06 * VAR07
    VAR04 = VAR07 * VAR07
    VAR05 = VAR07 * VAR08
    VAR15 = y * y * y * y
    VAR16 = y * y * y
    VAR17 = y * y
    VAR11 = VAR15 * VAR15
    VAR12 = VAR15 * VAR16
    VAR13 = VAR16 * VAR16
    VAR14 = VAR16 * VAR17
    VAR24 = z * z * z * z
    VAR25 = z * z * z
    VAR26 = z * z
    VAR20 = VAR24 * VAR24
    VAR21 = VAR24 * VAR25
    VAR22 = VAR25 * VAR25
    VAR23 = VAR25 * VAR26
    # -------------------- kernel implementations
    g_x = (
        g_0
        * (
            CONST021 * VAR20
            + CONST022 * VAR02
            + CONST179 * VAR04 * VAR26
            + CONST180 * VAR08 * VAR22
            + CONST204 * VAR06 * VAR24
        )
        + g_1
        * y
        * (
            CONST065 * VAR08 * VAR23
            - CONST149 * VAR06 * VAR25
            + CONST183 * VAR04 * z
            - CONST271 * VAR21
        )
        + g_10
        * (
            CONST012 * VAR21 * x
            + VAR23 * (CONST028 * VAR07 + CONST203 * VAR17 * x)
            + VAR25
            * (CONST028 * VAR05 + CONST157 * VAR15 * x + CONST173 * VAR07 * VAR17)
            + z
            * (
                CONST011 * VAR03
                + CONST157 * VAR07 * VAR15
                + CONST198 * VAR13 * x
                + CONST202 * VAR05 * VAR17
            )
        )
        + g_11
        * (
            CONST150 * VAR07 * VAR14
            + CONST250 * VAR12 * x
            + VAR16
            * (CONST093 * VAR24 * x + CONST165 * VAR05 + CONST186 * VAR07 * VAR26)
            + y * (CONST059 * VAR03 + CONST071 * VAR05 * VAR26 + CONST281 * VAR22 * x)
        )
        + g_12
        * (
            VAR23 * (CONST257 * VAR17 * x - CONST290 * VAR07)
            + VAR25
            * (CONST044 * VAR05 + CONST143 * VAR07 * VAR17 - CONST172 * VAR15 * x)
            + z
            * (
                CONST155 * VAR05 * VAR17
                + CONST184 * VAR13 * x
                - CONST217 * VAR07 * VAR15
                - CONST288 * VAR03
            )
        )
        + g_13
        * (
            VAR14 * (CONST129 * VAR26 * x - CONST195 * VAR07)
            + VAR16
            * (CONST166 * VAR24 * x + CONST176 * VAR05 - CONST222 * VAR07 * VAR26)
            + y
            * (
                CONST188 * VAR07 * VAR24
                + CONST209 * VAR05 * VAR26
                - CONST259 * VAR03
                + CONST259 * VAR22 * x
            )
        )
        + g_14
        * (
            CONST042 * VAR03 * z
            + CONST268 * VAR07 * VAR23
            + CONST294 * VAR21 * x
            + VAR15 * (CONST053 * VAR25 * x + CONST261 * VAR07 * z)
            + VAR17
            * (CONST119 * VAR05 * z + CONST144 * VAR23 * x + CONST152 * VAR07 * VAR25)
        )
        + g_15
        * (
            VAR16 * (CONST068 * VAR24 * x - CONST099 * VAR07 * VAR26 + CONST205 * VAR05)
            + y * (CONST050 * VAR03 + CONST147 * VAR05 * VAR26 - CONST205 * VAR22 * x)
        )
        + g_16
        * (
            CONST214 * VAR05 * VAR25
            - CONST264 * VAR03 * z
            + CONST264 * VAR07 * VAR23
            - CONST275 * VAR21 * x
            + VAR17
            * (CONST079 * VAR07 * VAR25 + CONST134 * VAR05 * z + CONST134 * VAR23 * x)
        )
        + g_17
        * y
        * (
            CONST065 * VAR05 * VAR26
            - CONST149 * VAR07 * VAR24
            + CONST183 * VAR22 * x
            - CONST271 * VAR03
        )
        + g_18
        * (
            CONST132 * VAR05 * VAR25
            + CONST175 * VAR07 * VAR23
            - CONST234 * VAR03 * z
            + CONST234 * VAR21 * x
        )
        + g_2
        * (
            CONST002 * VAR08 * (CONST034 * VAR22 + CONST153 * VAR17 * VAR24)
            + CONST004 * VAR06 * (CONST023 * VAR24 - CONST182 * VAR17 * VAR26)
            + CONST006 * VAR04 * (CONST289 * VAR26 + CONST291 * VAR17)
            - CONST228 * VAR17 * VAR22
            - CONST295 * VAR02
            + CONST298 * VAR20
        )
        + g_3
        * (
            VAR16
            * (-CONST068 * VAR06 * z + CONST099 * VAR08 * VAR25 + CONST103 * VAR23)
            + y
            * (
                CONST116 * VAR08 * VAR23
                - CONST163 * VAR06 * VAR25
                + CONST190 * VAR04 * z
                + CONST272 * VAR21
            )
        )
        + g_4
        * (
            CONST007 * VAR20
            + CONST014 * VAR02
            + CONST254 * VAR06 * VAR24
            + CONST269 * VAR04 * VAR26
            + VAR15 * (CONST114 * VAR06 + CONST114 * VAR24 + CONST168 * VAR08 * VAR26)
            + VAR17
            * (
                CONST060 * VAR06 * VAR26
                + CONST133 * VAR08 * VAR24
                + CONST212 * VAR04
                + CONST224 * VAR22
            )
        )
        + g_5
        * (
            VAR14 * (CONST130 * VAR08 * z - CONST195 * VAR25)
            + VAR16 * (CONST195 * VAR23 - CONST222 * VAR06 * z)
            + y
            * (
                CONST067 * VAR08 * VAR23
                + CONST200 * VAR04 * z
                + CONST220 * VAR06 * VAR25
                - CONST284 * VAR21
            )
        )
        + g_6
        * (
            CONST002
            * VAR08
            * (
                CONST201 * VAR15 * VAR26
                - CONST219 * VAR17 * VAR24
                + CONST267 * VAR13
                + CONST299 * VAR22
            )
            + CONST004
            * VAR06
            * (CONST036 * VAR17 * VAR26 - CONST233 * VAR15 + CONST301 * VAR24)
            + CONST187 * VAR15 * VAR24
            + CONST197 * VAR04 * VAR17
            - CONST216 * VAR13 * VAR26
            - CONST239 * VAR17 * VAR22
            - CONST297 * VAR02
            + CONST302 * VAR20
        )
        + g_7
        * (
            CONST002
            * VAR08
            * (-CONST186 * VAR16 * VAR25 + CONST192 * VAR14 * z + CONST270 * VAR23 * y)
            + CONST004 * VAR06 * (-CONST218 * VAR16 * z + CONST270 * VAR25 * y)
            + CONST193 * VAR14 * VAR25
            - CONST218 * VAR16 * VAR23
            + CONST229 * VAR04 * y * z
            - CONST250 * VAR12 * z
            + CONST292 * VAR21 * y
        )
        + g_8
        * (
            CONST000 * VAR20
            + CONST002
            * VAR08
            * (
                CONST005 * VAR22
                + CONST115 * VAR15 * VAR26
                + CONST230 * VAR13
                + CONST235 * VAR17 * VAR24
            )
            + CONST004
            * VAR06
            * (CONST008 * VAR24 + CONST085 * VAR15 + CONST235 * VAR17 * VAR26)
            + CONST006 * VAR04 * (CONST009 * VAR26 + CONST278 * VAR17)
            + CONST015 * VAR02
            + CONST024 * VAR11
            + CONST085 * VAR15 * VAR24
            + CONST231 * VAR13 * VAR26
            + CONST278 * VAR17 * VAR22
        )
        + g_9
        * (
            CONST245 * VAR12 * x
            + VAR14 * (CONST141 * VAR07 + CONST141 * VAR26 * x)
            + VAR16
            * (CONST131 * VAR07 * VAR26 + CONST178 * VAR05 + CONST178 * VAR24 * x)
            + y
            * (
                CONST045 * VAR03
                + CONST046 * VAR22 * x
                + CONST087 * VAR05 * VAR26
                + CONST088 * VAR07 * VAR24
            )
        )
    )
    g_y = (
        CONST001
        * g_16
        * y
        * (
            CONST160 * VAR06 * VAR25
            + CONST182 * VAR08 * VAR23
            + CONST228 * VAR04 * z
            - CONST291 * VAR21
        )
        + g_1
        * (
            -CONST183 * VAR05 * VAR25
            + CONST183 * VAR07 * VAR23
            + CONST271 * VAR03 * z
            - CONST271 * VAR21 * x
        )
        + g_10
        * (
            CONST252 * VAR21 * y
            + VAR23 * (CONST157 * VAR16 + CONST203 * VAR08 * y)
            + VAR25
            * (CONST140 * VAR14 + CONST202 * VAR06 * y + CONST303 * VAR08 * VAR16)
            + z
            * (
                CONST080 * VAR12
                + CONST139 * VAR08 * VAR14
                + CONST157 * VAR06 * VAR16
                + CONST252 * VAR04 * y
            )
        )
        + g_11
        * (
            CONST002
            * VAR17
            * (
                CONST064 * VAR08 * VAR24
                + CONST248 * VAR04
                + CONST248 * VAR06 * VAR26
                - CONST248 * VAR22
            )
            + CONST004 * VAR15 * (CONST082 * VAR06 + CONST225 * VAR24)
            + CONST006 * VAR13 * (CONST277 * VAR08 - CONST277 * VAR26)
            + CONST017 * VAR02
            + CONST025 * VAR04 * VAR26
            + CONST293 * VAR08 * VAR22
            + CONST296 * VAR20
        )
        + g_12
        * (
            CONST056 * VAR21 * y
            + VAR23 * (CONST171 * VAR16 + CONST257 * VAR08 * y)
            + VAR25
            * (-CONST113 * VAR08 * VAR16 - CONST185 * VAR14 + CONST187 * VAR06 * y)
            + z
            * (
                CONST066 * VAR08 * VAR14
                + CONST206 * VAR04 * y
                - CONST217 * VAR06 * VAR16
            )
        )
        + g_13
        * (
            CONST002
            * VAR17
            * (
                CONST117 * VAR06 * VAR26
                + CONST117 * VAR08 * VAR24
                + CONST259 * VAR04
                + CONST260 * VAR22
            )
            + CONST004
            * VAR15
            * (CONST055 * VAR06 + CONST055 * VAR24 + CONST176 * VAR08 * VAR26)
            + CONST018 * VAR20
            + CONST019 * VAR02
            + CONST249 * VAR06 * VAR24
            + CONST284 * VAR04 * VAR26
            + CONST285 * VAR08 * VAR22
        )
        + g_14
        * (
            CONST001
            * y
            * (
                CONST083 * VAR06 * VAR25
                + CONST109 * VAR08 * VAR23
                + CONST226 * VAR04 * z
                + CONST286 * VAR21
            )
            + CONST003
            * VAR16
            * (CONST114 * VAR06 * z + CONST159 * VAR08 * VAR25 - CONST269 * VAR23)
        )
        + g_15
        * (
            CONST002
            * VAR17
            * (
                CONST039 * VAR22
                - CONST163 * VAR06 * VAR26
                + CONST163 * VAR08 * VAR24
                + CONST279 * VAR04
            )
            + CONST020 * VAR02
            + CONST237 * VAR04 * VAR26
            - CONST237 * VAR08 * VAR22
            + CONST300 * VAR20
        )
        + g_17
        * (
            CONST137 * VAR06 * VAR24
            + CONST170 * VAR02
            + CONST170 * VAR20
            + CONST215 * VAR04 * VAR26
            + CONST215 * VAR08 * VAR22
        )
        + g_2
        * (
            CONST108 * VAR22 * x * y
            - CONST134 * VAR05 * VAR26 * y
            + CONST262 * VAR07 * VAR24 * y
            + CONST280 * VAR03 * y
        )
        + g_3
        * (
            CONST002
            * VAR17
            * (CONST103 * VAR23 * x + CONST138 * VAR07 * VAR25 - CONST205 * VAR05 * z)
            - CONST237 * VAR05 * VAR25
            - CONST237 * VAR07 * VAR23
            + CONST272 * VAR03 * z
            + CONST272 * VAR21 * x
        )
        + g_4
        * (
            CONST001
            * y
            * (
                CONST110 * VAR05 * VAR26
                - CONST224 * VAR07 * VAR24
                + CONST224 * VAR22 * x
                + CONST287 * VAR03
            )
            + CONST003
            * VAR16
            * (CONST114 * VAR24 * x + CONST159 * VAR07 * VAR26 - CONST269 * VAR05)
        )
        + g_5
        * (
            CONST002 * VAR17 * (CONST112 * VAR05 * z + CONST195 * VAR23 * x)
            + CONST004 * VAR15 * (CONST195 * VAR07 * z - CONST195 * VAR25 * x)
            + CONST037 * VAR07 * VAR23
            + CONST284 * VAR05 * VAR25
            - CONST284 * VAR21 * x
            + CONST285 * VAR03 * z
        )
        + g_6
        * (
            CONST258 * VAR03 * y
            + VAR05 * (CONST057 * VAR26 * y - CONST171 * VAR16)
            + VAR07
            * (CONST113 * VAR16 * VAR26 + CONST185 * VAR14 - CONST187 * VAR24 * y)
            + x
            * (
                -CONST066 * VAR14 * VAR26
                - CONST206 * VAR22 * y
                + CONST217 * VAR16 * VAR24
            )
        )
        + g_7
        * (
            CONST292 * VAR03 * z
            + VAR05 * (-CONST165 * VAR17 * z + CONST270 * VAR25)
            + VAR07
            * (CONST207 * VAR15 * z + CONST223 * VAR17 * VAR25 + CONST270 * VAR23)
            + x
            * (
                CONST151 * VAR13 * z
                - CONST165 * VAR17 * VAR23
                + CONST207 * VAR15 * VAR25
                + CONST292 * VAR21
            )
        )
        + g_8
        * (
            CONST253 * VAR03 * y
            + VAR05 * (CONST156 * VAR16 + CONST202 * VAR26 * y)
            + VAR07
            * (CONST139 * VAR14 + CONST202 * VAR24 * y + CONST303 * VAR16 * VAR26)
            + x
            * (
                CONST081 * VAR12
                + CONST140 * VAR14 * VAR26
                + CONST156 * VAR16 * VAR24
                + CONST253 * VAR22 * y
            )
        )
        + g_9
        * (
            CONST002
            * VAR17
            * (
                CONST211 * VAR06 * VAR26
                + CONST211 * VAR08 * VAR24
                + CONST263 * VAR04
                + CONST263 * VAR22
            )
            + CONST004
            * VAR15
            * (CONST076 * VAR06 + CONST076 * VAR24 + CONST106 * VAR08 * VAR26)
            + CONST006 * VAR13 * (CONST273 * VAR26 + CONST274 * VAR08)
            + CONST031 * VAR11
            + CONST032 * VAR04 * VAR26
            + CONST032 * VAR08 * VAR22
            + CONST033 * VAR20
            + CONST040 * VAR06 * VAR24
            + CONST070 * VAR02
        )
    )
    g_z = (
        g_0
        * (
            CONST132 * VAR07 * VAR23
            + CONST175 * VAR05 * VAR25
            + CONST234 * VAR03 * z
            - CONST234 * VAR21 * x
        )
        + g_1
        * y
        * (
            -CONST065 * VAR05 * VAR26
            + CONST149 * VAR07 * VAR24
            - CONST183 * VAR22 * x
            + CONST271 * VAR03
        )
        + g_10
        * (
            CONST000 * VAR02
            + CONST002
            * VAR26
            * (
                CONST100 * VAR04
                + CONST115 * VAR08 * VAR15
                + CONST231 * VAR13
                + CONST235 * VAR06 * VAR17
            )
            + CONST004
            * VAR24
            * (CONST008 * VAR06 + CONST086 * VAR15 + CONST236 * VAR08 * VAR17)
            + CONST006 * VAR22 * (CONST005 * VAR08 + CONST282 * VAR17)
            + CONST015 * VAR20
            + CONST027 * VAR11
            + CONST086 * VAR06 * VAR15
            + CONST232 * VAR08 * VAR13
            + CONST282 * VAR04 * VAR17
        )
        + g_11
        * (
            CONST161 * VAR14 * VAR25
            - CONST250 * VAR12 * z
            + VAR16
            * (CONST123 * VAR08 * VAR25 - CONST165 * VAR23 + CONST218 * VAR06 * z)
            + y * (CONST038 * VAR04 * z + CONST238 * VAR08 * VAR23 + CONST255 * VAR21)
        )
        + g_12
        * (
            CONST002
            * VAR26
            * (
                CONST097 * VAR04
                - CONST201 * VAR08 * VAR15
                + CONST219 * VAR06 * VAR17
                - CONST267 * VAR13
            )
            + CONST004
            * VAR24
            * (CONST233 * VAR15 + CONST283 * VAR08 * VAR17 - CONST301 * VAR06)
            + CONST107 * VAR17 * VAR22
            - CONST187 * VAR06 * VAR15
            + CONST216 * VAR08 * VAR13
            + CONST239 * VAR04 * VAR17
            + CONST297 * VAR20
            - CONST302 * VAR02
        )
        + g_13
        * (
            VAR14 * (CONST129 * VAR08 * z - CONST195 * VAR25)
            + VAR16
            * (CONST166 * VAR06 * z + CONST177 * VAR23 - CONST222 * VAR08 * VAR25)
            + y
            * (
                CONST188 * VAR06 * VAR25
                + CONST210 * VAR08 * VAR23
                + CONST260 * VAR04 * z
                - CONST260 * VAR21
            )
        )
        + g_14
        * (
            CONST007 * VAR02
            + CONST189 * VAR20
            + CONST256 * VAR06 * VAR24
            + CONST269 * VAR08 * VAR22
            + VAR15 * (CONST114 * VAR06 + CONST114 * VAR24 + CONST168 * VAR08 * VAR26)
            + VAR17
            * (
                CONST061 * VAR08 * VAR24
                + CONST133 * VAR06 * VAR26
                + CONST213 * VAR22
                + CONST226 * VAR04
            )
        )
        + g_15
        * (
            VAR16
            * (-CONST068 * VAR06 * z + CONST099 * VAR08 * VAR25 + CONST103 * VAR23)
            + y * (-CONST147 * VAR08 * VAR23 + CONST205 * VAR04 * z + CONST265 * VAR21)
        )
        + g_16
        * (
            CONST074 * VAR02
            + CONST090 * VAR08 * VAR22
            + CONST244 * VAR04 * VAR26
            + CONST251 * VAR06 * VAR24
            + CONST295 * VAR20
            + VAR17
            * (
                CONST078 * VAR22
                - CONST142 * VAR06 * VAR26
                + CONST142 * VAR08 * VAR24
                + CONST228 * VAR04
            )
        )
        + g_17
        * y
        * (
            CONST065 * VAR08 * VAR23
            - CONST149 * VAR06 * VAR25
            + CONST183 * VAR04 * z
            - CONST271 * VAR21
        )
        + g_18
        * (
            CONST021 * VAR02
            + CONST022 * VAR20
            + CONST179 * VAR08 * VAR22
            + CONST180 * VAR04 * VAR26
            + CONST204 * VAR06 * VAR24
        )
        + g_2
        * (
            CONST275 * VAR03 * z
            + VAR05 * (CONST052 * VAR25 - CONST134 * VAR17 * z)
            + VAR07 * (-CONST214 * VAR23 + CONST227 * VAR17 * VAR25)
            + x * (-CONST134 * VAR17 * VAR23 + CONST266 * VAR21)
        )
        + g_3
        * (
            VAR16 * (CONST099 * VAR07 * VAR26 - CONST205 * VAR05 + CONST241 * VAR24 * x)
            + y
            * (
                CONST116 * VAR05 * VAR26
                - CONST163 * VAR07 * VAR24
                + CONST190 * VAR22 * x
                + CONST272 * VAR03
            )
        )
        + g_4
        * (
            CONST042 * VAR21 * x
            + CONST269 * VAR05 * VAR25
            + CONST294 * VAR03 * z
            + VAR15 * (CONST053 * VAR07 * z + CONST261 * VAR25 * x)
            + VAR17
            * (CONST121 * VAR23 * x + CONST145 * VAR05 * z + CONST154 * VAR07 * VAR25)
        )
        + g_5
        * (
            VAR14 * (-CONST130 * VAR26 * x + CONST195 * VAR07)
            + VAR16 * (CONST112 * VAR05 + CONST222 * VAR24 * x)
            + y
            * (
                CONST091 * VAR07 * VAR24
                + CONST105 * VAR22 * x
                + CONST242 * VAR05 * VAR26
                + CONST285 * VAR03
            )
        )
        + g_6
        * (
            VAR05 * (CONST057 * VAR17 * z + CONST290 * VAR25)
            + VAR07
            * (-CONST143 * VAR17 * VAR25 + CONST172 * VAR15 * z + CONST276 * VAR23)
            + x
            * (
                -CONST155 * VAR17 * VAR23
                - CONST184 * VAR13 * z
                + CONST217 * VAR15 * VAR25
                + CONST288 * VAR21
            )
        )
        + g_7
        * (
            CONST292 * VAR03 * y
            + VAR05 * (-CONST218 * VAR16 + CONST221 * VAR26 * y)
            + VAR07
            * (CONST192 * VAR14 + CONST196 * VAR24 * y + CONST223 * VAR16 * VAR26)
            + x
            * (
                CONST124 * VAR14 * VAR26
                + CONST191 * VAR16 * VAR24
                + CONST229 * VAR22 * y
                - CONST250 * VAR12
            )
        )
        + g_8
        * (
            CONST011 * VAR03 * z
            + VAR05 * (CONST028 * VAR25 + CONST202 * VAR17 * z)
            + VAR07
            * (CONST028 * VAR23 + CONST157 * VAR15 * z + CONST173 * VAR17 * VAR25)
            + x
            * (
                CONST011 * VAR21
                + CONST156 * VAR15 * VAR25
                + CONST199 * VAR13 * z
                + CONST202 * VAR17 * VAR23
            )
        )
        + g_9
        * (
            CONST246 * VAR12 * z
            + VAR14 * (CONST141 * VAR08 * z + CONST141 * VAR25)
            + VAR16
            * (CONST131 * VAR08 * VAR25 + CONST178 * VAR06 * z + CONST178 * VAR23)
            + y
            * (
                CONST046 * VAR04 * z
                + CONST046 * VAR21
                + CONST087 * VAR08 * VAR23
                + CONST088 * VAR06 * VAR25
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
