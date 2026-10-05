"""UMA's input batch is built on the CPU, whatever the ML device.

fairchem's predictor moves the batch to its own device. On its first call it
mixes UMA's experts (merge_mole) before moving the model off the CPU, so a
batch already on the GPU fails there: "Expected all tensors to be on the same
device" (Picard, 2026-10-05). The Mac's MPS stands in for the GPU here.
"""
import pytest
import torch

from test_charge_and_spin_reach_uma import core_without_models, water

other_devices = [name for name, available in [
    ("cuda", torch.cuda.is_available()),
    ("mps", torch.backends.mps.is_available()),
] if available]


@pytest.mark.skipif(not other_devices, reason="no GPU or MPS on this machine")
def test_batch_is_on_the_cpu_when_the_ml_device_is_not():
    core = core_without_models(model_charge=0, model_mult=6)
    core.ml_device = torch.device(other_devices[0])

    batch = core._ase_to_batch(water())

    assert batch.pos.device.type == "cpu"
    assert batch.charge.device.type == "cpu"
