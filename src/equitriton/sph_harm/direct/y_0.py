import triton
import torch
from triton import language as tl

__all__ = ["ZerothOrderSphericalHarmonic"]


class ZerothOrderSphericalHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
        block_size: int = 64,
    ):
        output_tensor = torch.ones(
            (*coords.shape[:-1], 1), dtype=coords.dtype, device=coords.device
        )
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(
        ctx, sph_grad_tensor: torch.Tensor, block_size: int = 64
    ) -> torch.Tensor:
        (coords,) = ctx.saved_tensors
        return torch.zeros_like(coords)


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
    output = torch.ones_like(x)
    return output


@triton.jit
def zeroth_order_fwd(
    output_ptr: tl.tensor,
    block_size: tl.constexpr,
    coord_numel: tl.constexpr,
    output_numel: tl.constexpr,
):
    # work out the row offsets
    block_id = tl.program_id(0)
    output_stride = 1  # [2l + 1]
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = output_striding + (block_size * output_stride * block_id)
    tl.store(output_ptr + output_row_offset, 1.0, mask=output_row_offset < output_numel)
