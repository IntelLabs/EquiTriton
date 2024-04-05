# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from argparse import ArgumentParser
from logging import getLogger

import torch
from torch.profiler import record_function
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from equitriton.sph_harm.bindings import *
from equitriton.benchmark import profile

"""
Runs the PyTorch profiler on either the Triton or ``e3nn`` kernels.

This provides a more in-depth analysis into the relative performance
of the kernels, as it produces a timeline for operations performed.
"""

logger = getLogger("equitriton.benchmark").setLevel("INFO")

triton_bindings = [
    None,
    FirstOrderSphericalHarmonics,
    SecondOrderSphericalHarmonics,
    ThirdOrderSphericalHarmonics,
    FourthOrderSphericalHarmonics,
]

parser = ArgumentParser()
parser.add_argument(
    "device", type=str, choices=["xpu", "cuda"], help="Device to profile on."
)
parser.add_argument("l_max", type=int, help="Maximum angular momentum to consider.")
parser.add_argument(
    "-n",
    "--num_steps",
    type=int,
    default=100,
    help="Total number of steps to profile over.",
)
parser.add_argument(
    "-w",
    "--warmup_fraction",
    type=float,
    default=0.1,
    help="Fraction of `num_steps` to use as warmup.",
)
parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    default="",
    help="Prefix to use for naming this experiment.",
)
parser.add_argument(
    "-s", "--size", type=int, default=10_000_000, help="Number of nodes to use."
)

args = parser.parse_args()


@profile(
    experiment_name=f"{args.prefix}e3nn_{args.device}",
    num_steps=args.num_steps,
    warmup_fraction=args.warmup_fraction,
)
def e3nn_benchmark(tensor_shape: list[int], device: str | torch.device, l_max: int):
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    x, y, z = (
        joint_tensor[..., 0].contiguous(),
        joint_tensor[..., 1].contiguous(),
        joint_tensor[..., 2].contiguous(),
    )
    with record_function("forward"):
        output = _spherical_harmonics(l_max, x, y, z)
    with record_function("backward"):
        output.backward(gradient=torch.ones_like(output))
    # delete references to ensure memory gets cleared
    del output
    del joint_tensor


@profile(
    experiment_name=f"{args.prefix}triton_{args.device}",
    num_steps=args.num_steps,
    warmup_fraction=args.warmup_fraction,
)
def triton_benchmark(tensor_shape: list[int], device: str | torch.device, l_max: int):
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    x, y, z = (
        joint_tensor[..., 0].contiguous(),
        joint_tensor[..., 1].contiguous(),
        joint_tensor[..., 2].contiguous(),
    )
    kernel = triton_bindings[l_max]
    with record_function("forward"):
        output = kernel.apply(x, y, z)
    with record_function("backward"):
        output.backward(gradient=torch.ones_like(output))
    # delete references to ensure memory gets cleared
    del output
    del joint_tensor


e3nn_benchmark(tensor_shape=(args.size, 3), device=args.device, l_max=args.l_max)
triton_benchmark(tensor_shape=(args.size, 3), device=args.device, l_max=args.l_max)
