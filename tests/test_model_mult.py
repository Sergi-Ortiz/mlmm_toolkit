"""The ML region's spin multiplicity reaches UMA from every entry point.

Every entry point (mlmm via pysisyphus, ts_search, bond_scan,
energy_summary) builds the pysisyphus calculator `mlmm`, so its
multiplicity must reach both MLMMCore (which sends it to UMA as
atoms.info["spin"]) and pysisyphus's own Calculator.
"""
import mlmm.mlmm_pysis as mlmm_pysis


class RecordingCore:
    """Stands in for MLMMCore: records its arguments, loads nothing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.freeze_atoms = []


def test_model_mult_reaches_the_ml_core_and_pysisyphus(monkeypatch):
    monkeypatch.setattr(mlmm_pysis, "MLMMCore", RecordingCore)

    calc = mlmm_pysis.mlmm(model_charge=-1, model_mult=2)

    assert calc.core.kwargs["model_mult"] == 2
    assert calc.mult == 2


def test_default_multiplicity_is_a_singlet(monkeypatch):
    monkeypatch.setattr(mlmm_pysis, "MLMMCore", RecordingCore)

    calc = mlmm_pysis.mlmm(model_charge=0)

    assert calc.core.kwargs["model_mult"] == 1
    assert calc.mult == 1
