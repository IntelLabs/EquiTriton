import pytest
import torch

from equitriton import __HAS_XPU__, __HAS_CUDA__
from equitriton.sph_harm.direct import y_2


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(not __HAS_XPU__, reason="No XPUs available."),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not __HAS_CUDA__, reason="No CUDA GPUs available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
def test_forward_equivalence(device, tensor_shape):
    coords = torch.rand(tensor_shape, device=device)
    triton_out = y_2.SecondOrderSphericalHarmonic.apply(coords)
    torch_out = y_2.torch_second_order_fwd(coords)
    assert torch.allclose(triton_out, torch_out, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "xpu",
            marks=pytest.mark.skipif(not __HAS_XPU__, reason="No XPUs available."),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not __HAS_CUDA__, reason="No CUDA GPUs available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("tensor_shape", [(512, 3), (128, 16, 3), (256, 8, 8, 3)])
def test_backward_equivalence(device, tensor_shape):
    coords = torch.rand(tensor_shape, device=device, requires_grad=True)
    # run with autograd first
    torch_out = y_2.torch_second_order_fwd(coords)
    torch_out.backward(gradient=torch.ones_like(torch_out))
    torch_grad = coords.grad.clone().detach()
    coords.grad.zero_()
    # now run the triton result
    triton_out = y_2.SecondOrderSphericalHarmonic.apply(coords)
    triton_out.backward(gradient=torch.ones_like(triton_out))
    triton_grad = coords.grad.clone().detach()
    assert torch.allclose(triton_grad, torch_grad, atol=1e-6, rtol=1e-4)