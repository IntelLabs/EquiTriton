# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from argparse import ArgumentParser

import torch
import numpy as np
import e3nn
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from equitriton.sph_harm.bindings import *

"""
This script is used to measure the numerical error between e3nn
and Triton implementations.
"""

logger = getLogger("equitriton.benchmark")
logger.setLevel("INFO")

triton_bindings = [
    None,
    FirstOrderSphericalHarmonics,
    SecondOrderSphericalHarmonics,
    ThirdOrderSphericalHarmonics,
    FourthOrderSphericalHarmonics,
]

parser = ArgumentParser()
parser.add_argument(
    "l", type=int, choices=[1, 2, 3, 4], help="Maximum number of terms to test."
)
parser.add_argument(
    "device", type=str, choices=["xpu", "cuda"], help="Device to profile on."
)
parser.add_argument("l_max", type=int, help="Maximum angular momentum to consider.")
parser.add_argument(
    "-n",
    "--num_iter",
    type=int,
    default=1000,
    help="Total number of iterations to sample over.",
)
parser.add_argument(
    "-i",
    "--num_feats",
    type=int,
    default=5000,
    help="Number of nodes/features to compute over.",
)
parser.add_argument(
    "--relative",
    action="store_true",
    help="Flag to calculate relative percentage errors instead of absolute errors.",
)
parser.add_argument(
    "-d",
    "--dtype",
    choices=["float", "float32", "float64"],
    help="Precision to perform the tests with.",
)

args = parser.parse_args()


def compare_e3nn_triton(
    joint_tensor: torch.Tensor, l_max: int, relative: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    # clear gradients just in case
    joint_tensor.grad = None
    x, y, z = (
        joint_tensor[..., 0].contiguous(),
        joint_tensor[..., 1].contiguous(),
        joint_tensor[..., 2].contiguous(),
    )
    e3nn.set_optimization_defaults(jit_script_fx=False)
    e3nn_output = torch.compile(_spherical_harmonics, fullgraph=True, mode="max-autotune")(l_max, x, y, z)
    e3nn_output.backward(gradient=torch.ones_like(e3nn_output))
    e3nn_grad = joint_tensor.grad.detach().clone()
    joint_tensor.grad = None
    # now do the same with the Triton version
    kernel = triton_bindings[l_max]
    triton_output = kernel.apply(x, y, z)
    triton_output.backward(gradient=torch.ones_like(triton_output))
    triton_grad = joint_tensor.grad.detach().clone()
    # overzealous with the detachs honestly :P
    signed_fwd_error = (e3nn_output - triton_output).detach().cpu().numpy()
    if relative:
        # compute relative percentage error
        signed_fwd_error /= e3nn_output.detach().cpu().numpy()
        signed_fwd_error *= 100.0
    signed_bwd_error = (e3nn_grad - triton_grad).detach().cpu().numpy()
    if relative:
        signed_bwd_error /= e3nn_grad.detach().cpu().numpy()
        signed_bwd_error *= 100.0
    # delete intermediate tensors to make sure we don't leak
    del e3nn_output
    del triton_output
    e3nn.set_optimization_defaults(jit_script_fx=True) # Turn it back on to avoid any issues 
    return (signed_fwd_error, signed_bwd_error)


def run_test(
    num_iter: int,
    num_feats: int,
    device: str | torch.device,
    l_max: int,
    percentiles: list[float] | np.ndarray = [0.02, 0.5, 0.98],
    relative: bool = True,
    dtype: torch.dtype = torch.float,
):
    """
    Run a set of numerical error tests comparing the e3nn and Triton forward
    and backward results. This is used to quantify, for a given precision,
    how far off the Triton result might be from e3nn.

    It is recommended that this is run and understood before replacing
    the e3nn kernels with the _EquiTriton_ ones.

    Parameters
    ----------
    num_iter : int
        Number of iterations to test. This is basically how many
        random tensors are going to be initialized.
    num_feats
        Number of nodes/features per iterations.
    device : str | torch.deviec
        Device to execute on.
    l_max : int
        Maximum number of terms to consider.
    percentiles : list[float] | np.ndarray
        Percentiles to compute statistics with. The default values should
        be reasonably descriptive. Keep in mind that, if ``relative`` is
        True, then the values reported are in percentage error, as opposed
        to absolute error.
    relative : bool, default True
        If True, computes the relative percentage error by dividing
        by the e3nn result.
    dtype : torch.dtype, default torch.float
        Data type to compute with.
    """
    fwd_errors = []
    bwd_errors = []
    for _ in range(num_iter):
        joint_tensor = torch.rand(
            [num_feats, 3], device=device, requires_grad=True, dtype=dtype
        )
        fwd_error, bwd_error = compare_e3nn_triton(joint_tensor, l_max, relative)
        fwd_errors.append(fwd_error)
        bwd_errors.append(bwd_error)
    # get back shape of [num_feats, 3] for binning
    fwd_errors = np.vstack(fwd_errors)
    bwd_errors = np.vstack(bwd_errors)
    # calculate error percentiles along samples dimension;
    # output array is [percentiles, xyz]
    fwd_percentiles = np.percentile(fwd_errors, percentiles, axis=0)
    bwd_percentiles = np.percentile(bwd_errors, percentiles, axis=0)
    logger.info(
        f"Numerical error analysis for l_max=${l_max} on {device}. {num_iter} iterations using {num_feats} random nodes."
    )
    for index, axis in enumerate(["x", "y", "z"]):
        logger.info(f"---------- Result for axis: {axis} ----------")
        logger.info(
            f"Forward signed percentile ({percentiles}) errors: {fwd_percentiles[:,index]}"
        )
        logger.info(
            f"Backward signed percentile ({percentiles}) errors: {bwd_percentiles[:,index]}"
        )


dtype = getattr(torch, args.dtype)

run_test(
    args.num_iter,
    args.num_feats,
    args.device,
    args.l,
    relative=args.relative,
    dtype=dtype,
)
