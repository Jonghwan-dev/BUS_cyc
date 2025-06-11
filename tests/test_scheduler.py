import os
import sys
import types

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import networks


def test_get_scheduler_invalid_policy_raises():
    optimizer = torch.optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=0.1)
    opt = types.SimpleNamespace(lr_policy="invalid")
    with pytest.raises(NotImplementedError):
        networks.get_scheduler(optimizer, opt)
