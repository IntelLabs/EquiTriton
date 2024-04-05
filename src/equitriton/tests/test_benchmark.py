# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from time import sleep
from random import random

from equitriton.benchmark import benchmark


def test_benchmark_decorator():
    @benchmark(num_steps=50)
    def dummy_func():
        sleep(random()) # nosec

    dummy_func()
