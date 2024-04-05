# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, schedule
from torch.profiler import profile as torch_profile
from typing import Callable
from functools import wraps
from time import perf_counter_ns
from logging import getLogger, INFO, basicConfig

from tqdm import tqdm
import numpy as np

basicConfig()

__all__ = ["benchmark"]


def benchmark(
    num_steps: int = 100,
    warmup_fraction: float = 0.05,
    percentiles: list[float] = [0.05, 0.1, 0.5, 0.9, 0.95],
):
    def decorator(func: Callable):
        @wraps(func)
        def benchmark_func(*args, **kwargs):
            logger = getLogger("equitriton.benchmark")
            logger.setLevel(INFO)
            times = []
            assert (
                warmup_fraction < 1.0
            ), f"Invalid warm up fraction: got {warmup_fraction}"
            warmup_steps = int(warmup_fraction * num_steps)
            # try and determine the device from kwargs
            if "device" in kwargs:
                device = kwargs["device"]
                if isinstance(device, str) and "xpu" in device:
                    sync_func = torch.xpu.synchronize
                elif isinstance(device, str) and "cuda" in device:
                    sync_func = torch.cuda.synchronize
                elif isinstance(device, torch.device):
                    device_type = device.type
                    submodule = import_module(f"torch.{device_type}")
                    sync_func = getattr(submodule, "synchronize", None)
                    if not sync_func:
                        raise NotImplementedError(
                            f"Device {device} does not have a synchronize function in torch."
                        )
            else:
                device = "unknown device"
                sync_func = None
            logger.info(
                f"Benchmarking {func} on {device} with {num_steps} steps ({warmup_steps} warm up)."
            )
            # clear cache
            if sync_func:
                cache = torch.empty(256_000_000, dtype=torch.int8, device=device)
                sync_func()
            total_start = perf_counter_ns()
            for i in tqdm(range(num_steps), desc=f"{func} on {device}"):
                if sync_func:
                    cache.zero_()
                    sync_func()
                if i > warmup_steps:
                    start_time = perf_counter_ns()
                _ = func(*args, **kwargs)
                if sync_func:
                    sync_func()
                if i > warmup_steps:
                    end_time = perf_counter_ns()
                    times.append(end_time - start_time)
            total_end = perf_counter_ns()
            times = np.array(times) / 1e6  # convert to milliseconds
            benchmark_percentiles = np.percentile(times, q=percentiles)
            end_to_end = (total_end - total_start) / 1e6
            logger.info(
                f"{num_steps} took {end_to_end} milliseconds to complete (including warm up!)."
            )
            logger.info("Reporting percentiles.")
            for per, value in zip(percentiles, benchmark_percentiles):
                logger.info(f"{per * 100} percentile - {value} milliseconds")
            return times

        return benchmark_func

    return decorator


def profile(
    experiment_name: str,
    num_steps: int = 100,
    warmup_fraction: float = 0.05,
    repeat: int = 1,
    **profile_kwargs,
):
    profile_kwargs.setdefault("profile_memory", True)
    profile_kwargs.setdefault("with_stack", True)
    profile_kwargs.setdefault("record_shapes", True)

    def decorator(func: Callable):
        @wraps(func)
        def benchmark_func(*args, **kwargs):
            logger = getLogger("equitriton.benchmark")
            logger.setLevel(INFO)
            activities = [ProfilerActivity.CPU]
            if "device" in kwargs:
                device = kwargs["device"]
                if "cuda" in device:
                    activities.append(ProfilerActivity.CUDA)
                    profile_kwargs.setdefault("use_cuda", True)
                if "xpu" in device:
                    activities.append(ProfilerActivity.XPU)
            sch = schedule(
                active=num_steps,
                warmup=int(num_steps * warmup_fraction),
                wait=0,
                repeat=repeat,
            )
            logger.info(
                f"Profiling {activities} for {num_steps} steps ({int(num_steps * warmup_fraction)} warmup)."
            )
            with torch_profile(
                activities=activities, schedule=sch, **profile_kwargs
            ) as prof_obj:
                for _ in tqdm(range(num_steps)):
                    _ = func(*args, **kwargs)
            print(prof_obj.key_averages().table(row_limit=10))
            try:
                prof_obj.export_chrome_trace(f"{experiment_name}_trace.json")
            except Exception as e:
                logger.warn(f"Unable to export trace due to {e}.")
            try:
                prof_obj.export_memory_timeline(f"{experiment_name}_memory.json")
            except Exception as e:
                logger.warn(f"Unable to export memory profile due to {e}.")
            return prof_obj

        return benchmark_func

    return decorator
