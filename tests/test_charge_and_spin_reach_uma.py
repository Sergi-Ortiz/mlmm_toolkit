"""The ML region's charge and spin multiplicity reach UMA's input.

fairchem's AtomicData.from_ase reads `charge` and `spin` from atoms.info only
when they are listed in r_data_keys; without it, every structure gets charge 0
and spin 0. These tests build UMA's input batch through MLMMCore._ase_to_batch
with real fairchem, on a core that loads no model and no MM system.
"""
from types import SimpleNamespace

import pytest
from ase import Atoms

from mlmm.mlmm_calc import MLMMCore


def core_without_models(model_charge, model_mult):
    """An MLMMCore with only what _ase_to_batch reads."""
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData

    core = object.__new__(MLMMCore)
    core.model_charge = model_charge
    core.model_mult = model_mult
    core.uma_task_name = "omol"
    core.ml_device = "cpu"
    core._AtomicData = AtomicData
    core._data_list_collater = data_list_collater
    backbone = SimpleNamespace(max_neighbors=30, cutoff=6.0)
    core.predictor = SimpleNamespace(model=SimpleNamespace(module=SimpleNamespace(backbone=backbone)))
    return core


def water():
    return Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.9578, 0.0, 0.0], [-0.2399, 0.9273, 0.0]])


@pytest.mark.parametrize("model_charge, model_mult", [(0, 1), (-1, 2), (1, 6)])
def test_charge_and_multiplicity_are_in_the_batch(model_charge, model_mult):
    batch = core_without_models(model_charge, model_mult)._ase_to_batch(water())

    assert batch.charge.tolist() == [model_charge]
    assert batch.spin.tolist() == [model_mult]
