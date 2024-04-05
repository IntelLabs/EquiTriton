# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from argparse import ArgumentParser
from logging import getLogger

import torch
import numpy as np
import pandas as pd
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from equitriton.sph_harm.bindings import *
from equitriton.benchmark import benchmark

"""
This script is used to benchmark the performance of the Triton spherical
harmonics against the original e3nn implementation.

The script runs kernels a specified number warm up and recorded steps,
and uses them to calculate percentiles for the combined forward and
backward passes. The end result is a CSV file containing these statistics
as a function of the number of nodes.
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
    "-j",
    "--min_log_size",
    type=float,
    default=2.0,
    help="Minimum (log10) number of nodes.",
)
parser.add_argument(
    "-k",
    "--max_log_size",
    type=float,
    default=9.0,
    help="Maximum (log10) number of nodes.",
)
parser.add_argument(
    "-i", "--matrix_samples", type=int, default=20, help="Number of experiments to run."
)

args = parser.parse_args()


@benchmark(num_steps=args.num_steps, warmup_fraction=args.warmup_fraction)
def e3nn_benchmark(tensor_shape: list[int], device: str | torch.device, l_max: int):
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    x, y, z = (
        joint_tensor[..., 0].contiguous(),
        joint_tensor[..., 1].contiguous(),
        joint_tensor[..., 2].contiguous(),
    )
    output = _spherical_harmonics(l_max, x, y, z)
    output.backward(gradient=torch.ones_like(output))
    # delete references to ensure memory gets cleared
    del output
    del joint_tensor


@benchmark(num_steps=args.num_steps, warmup_fraction=args.warmup_fraction)
def triton_benchmark(tensor_shape: list[int], device: str | torch.device, l_max: int):
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    x, y, z = (
        joint_tensor[..., 0].contiguous(),
        joint_tensor[..., 1].contiguous(),
        joint_tensor[..., 2].contiguous(),
    )
    kernel = triton_bindings[l_max]
    output = kernel.apply(x, y, z)
    output.backward(gradient=torch.ones_like(output))
    # delete references to ensure memory gets cleared
    del output
    del joint_tensor


n_values = np.linspace(args.min_log_size, args.max_log_size, args.matrix_samples)

all_data = []
for N in n_values:
    joint_results = {"N": N}
    try:
        e3nn_prof = e3nn_benchmark(
            (int(10**N), 3), device=args.device, l_max=args.l_max
        )
        e3nn_stats = np.percentile(np.array(e3nn_prof), [0.05, 0.5, 0.95])
        for key, value in zip(["e3nn 5%", "e3nn 50%", "e3nn 95%"], e3nn_stats):
            joint_results[key] = value
    except Exception as e:
        logger.warn(f"e3nn benchmark failed for 10**{N} shape due to {e}")
    try:
        triton_prof = triton_benchmark(
            (int(10**N), 3), device=args.device, l_max=args.l_max
        )
        triton_stats = np.percentile(np.array(triton_prof), [0.05, 0.5, 0.95])
        for key, value in zip(["triton 5%", "triton 50%", "triton 95%"], triton_stats):
            joint_results[key] = value
    except Exception as e:
        logger.warn(f"Triton benchmark failed for 10**{N} shape due to {e}")
    all_data.append(joint_results)

df = pd.DataFrame(all_data)
df.to_csv(f"{args.device}_lmax{args.l_max}_results.csv", index=False)
