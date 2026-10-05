"""UMA's Hessian of the ML region comes back on the ML device.

The batch is built on the CPU (test_uma_batch_on_cpu.py), so autograd returns
the Hessian on the CPU. The rest of the ML/MM Hessian is assembled on the ML
device, and adding the two failed: "Expected all tensors to be on the same
device, but found at least two devices, cuda:0 and cpu" (Picard, 2026-10-05,
ML/MM GPU reproducibility check). The Mac's MPS stands in for the GPU here, and
a stand-in predictor with the energy sum(pos^2) has the Hessian 2 * identity.
"""
import types

import pytest
import torch

from test_charge_and_spin_reach_uma import core_without_models

other_devices = [name for name, available in [
    ("cuda", torch.cuda.is_available()),
    ("mps", torch.backends.mps.is_available()),
] if available]


class StandInPredictor:
    """Predicts the energy sum(pos^2) of the batch, as fairchem's predictor returns it."""

    def __init__(self):
        self.model = types.SimpleNamespace(train=lambda: None, eval=lambda: None)

    def predict(self, batch):
        return {"energy": (batch.pos ** 2).sum()}


def hessian_of_stand_in(ml_device):
    core = core_without_models(model_charge=0, model_mult=6)
    core.ml_device = torch.device(ml_device)
    core.H_dtype = torch.float32
    core.predictor = StandInPredictor()

    n_ml = 2
    batch = types.SimpleNamespace(pos=torch.zeros(n_ml, 3))
    pos = batch.pos.clone().requires_grad_(True)
    return core._uma_hessian(batch, pos, n_ml)


def test_the_hessian_is_the_second_derivative_of_the_energy():
    hessian = hessian_of_stand_in("cpu").reshape(6, 6)

    assert torch.allclose(hessian, 2 * torch.eye(6))


@pytest.mark.skipif(not other_devices, reason="no GPU or MPS on this machine")
def test_the_hessian_is_on_the_ml_device_when_the_batch_is_on_the_cpu():
    hessian = hessian_of_stand_in(other_devices[0])

    assert hessian.device.type == other_devices[0]
