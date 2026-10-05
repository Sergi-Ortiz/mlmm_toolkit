import pytest

import mlmm.mlmm_pysis as mlmm_pysis


class RecordingCore:
    """Stands in for MLMMCore: records its arguments, loads nothing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.freeze_atoms = []


@pytest.fixture
def recording_core(monkeypatch):
    """The pysisyphus calculator builds a RecordingCore instead of MLMMCore."""
    monkeypatch.setattr(mlmm_pysis, "MLMMCore", RecordingCore)
