"""ts_search sets deterministic mode before its own CUDA work.

With max_cycles_preopt: 0, ts_search's first CUDA work is a torch.cdist in
_compute_dynamic_freeze, before any calculator exists. cuBLAS reads
CUBLAS_WORKSPACE_CONFIG only once, at its first call, so deterministic mode
must be set when ts_search starts, not when its first calculator is built.

These tests stop the constructor at its `model_pdb` check, which comes just
after deterministic mode is set, so no input files are needed.
"""
import pytest

import mlmm.partial_hessian_dimer as partial_hessian_dimer


@pytest.fixture
def deterministic_calls(monkeypatch):
    """Record calls to use_deterministic_torch instead of making them."""
    calls = []
    monkeypatch.setattr(
        partial_hessian_dimer, "use_deterministic_torch", lambda: calls.append(True)
    )
    return calls


def start_ts_search_without_model_pdb(tmp_path, mlmm_kwargs):
    with pytest.raises(ValueError, match="model_pdb"):
        partial_hessian_dimer.PartialHessianDimer(
            out_dir=str(tmp_path / "dimer"),
            vib_dir=str(tmp_path / "vib"),
            mlmm_kwargs=mlmm_kwargs,
        )


def test_deterministic_mode_is_set_by_default(tmp_path, deterministic_calls):
    start_ts_search_without_model_pdb(tmp_path, mlmm_kwargs={})

    assert len(deterministic_calls) == 1


def test_deterministic_mode_is_set_when_asked(tmp_path, deterministic_calls):
    start_ts_search_without_model_pdb(tmp_path, mlmm_kwargs={"deterministic": True})

    assert len(deterministic_calls) == 1


def test_deterministic_false_leaves_torch_alone(tmp_path, deterministic_calls):
    start_ts_search_without_model_pdb(tmp_path, mlmm_kwargs={"deterministic": False})

    assert len(deterministic_calls) == 0
