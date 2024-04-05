# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from argparse import ArgumentParser
from logging import getLogger
from time import time_ns

import torch
import numpy as np
import pandas as pd
import triton

from equitriton.sph_harm import SphericalHarmonics

SEED = 215616
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

"""
This script is used to benchmark the performance of the Triton spherical
harmonics over a uniform random number of nodes. The idea behind this script
is to look at the overhead associated with kernel compilation, which can
impact training/inference performance if input shapes change wildly.

The minimum and maximum number of nodes may need to be tweaked to match
what you might expect based on the data you work with.
"""

logger = getLogger("equitriton.benchmark")
logger.setLevel("INFO")


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
parser.add_argument("-p", "--pad", action="store_true", help="Enable tensor padding.")

args = parser.parse_args()


def triton_benchmark(
    tensor_shape: list[int], device: str | torch.device, l_max: int, pad_tensors: bool
):
    joint_tensor = torch.rand(tensor_shape, device=device, requires_grad=True)
    sph_harm = SphericalHarmonics(l_max, pad_tensors)
    output = sph_harm(joint_tensor)
    output.backward(gradient=torch.ones_like(output))
    # delete references to ensure memory gets cleared
    del output
    del joint_tensor


all_data = []
start_time = time_ns()
last_time = start_time
for _ in range(args.num_steps):
    num_nodes = int(10 ** rng.uniform(args.min_log_size, args.max_log_size))
    expect_pad = triton.next_power_of_2(num_nodes)
    joint_results = {"N": num_nodes, "pad_size": expect_pad}
    try:
        triton_benchmark(
            (num_nodes, 3), device=args.device, l_max=args.l_max, pad_tensors=args.pad
        )
    except Exception as e:
        logger.warning(f"Triton benchmark failed for {num_nodes} nodes due to {e}")
    end_time = time_ns()
    timedelta = (end_time - last_time) * 1e-9
    last_time = end_time
    joint_results["timedelta"] = timedelta
    all_data.append(joint_results)
logger.info(f"All tests finished in {(last_time - start_time) * 1e-9} seconds.")

df = pd.DataFrame(all_data)
df.to_csv(f"{args.device}_lmax{args.l_max}_jit_results.csv", index=False)
