# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from os import environ
from warnings import warn
from importlib.util import find_spec

import torch

__HAS_IPEX__ = True if find_spec("intel_extension_for_pytorch") else False
__HAS_CUDA__ = torch.cuda.is_available()
__HAS_XPU__ = False

if __HAS_IPEX__:
    try:
        import intel_extension_for_pytorch  # noqa: F401

        __HAS_XPU__ = torch.xpu.device_count() != 0
    except ImportError as e:
        warn(f"Unable to load IPEX due to {e}; XPU may not function.")

if "PATCH_E3NN" in environ:
    _will_patch = bool(environ.get("PATCH_E3NN", False))

    if _will_patch:
        from equitriton import patch  # noqa: F401

__version__ = "0.2.0"
