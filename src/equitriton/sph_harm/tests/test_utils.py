# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
import pytest
import torch

from equitriton.sph_harm.bindings import split_tensor_by_l, total_projections


@pytest.mark.parametrize("l_max", [1, 2, 3])
def test_split_tensor(l_max):
    feat_dim = total_projections(l_max)
    # this is equivalent to 128 nodes
    expected_output = torch.rand(128, feat_dim)
    split_tensors = split_tensor_by_l(expected_output, l_max)
    # should be a tensor for each component
    assert len(split_tensors) == l_max + 1
