"""UMA is loaded with the settings ML/MM needs.

The ML region's composition, charge and spin are fixed for a run, so the
experts can be merged once (merge_mole). torch.compile stays off: it crashes
on macOS CPU, and the Hessian needs a double backward through the model.

uma_settings: legacy loads UMA as the upstream toolkit did on its older
fairchem: experts mixed on every call, activation checkpointing on.
"""
import pytest

import mlmm.mlmm_pysis as mlmm_pysis
from mlmm.mlmm_calc import MLMMCore, uma_inference_settings


def test_experts_are_merged_once():
    assert uma_inference_settings().merge_mole is True


def test_model_is_not_compiled():
    assert uma_inference_settings().compile is False


def test_the_model_builds_its_own_graph():
    # The toolkit gives UMA no edges (otf_graph=True): the model must build them.
    # Older fairchem leaves this unset unless the settings start from its preset.
    assert uma_inference_settings().external_graph_gen is False


def test_default_keeps_to_the_general_backend():
    # umas_fast_gpu's Triton kernels cannot be differentiated twice: its Hessians are wrong
    assert uma_inference_settings().execution_mode == "general"


def test_default_is_the_same_when_named():
    assert uma_inference_settings("default") == uma_inference_settings()


def test_legacy_mixes_the_experts_on_every_call():
    assert uma_inference_settings("legacy").merge_mole is False


def test_legacy_checkpoints_activations_without_tf32():
    settings = uma_inference_settings("legacy")
    assert settings.activation_checkpointing is True
    assert settings.tf32 is False


def test_legacy_is_not_compiled_and_builds_its_own_graph():
    settings = uma_inference_settings("legacy")
    assert settings.compile is False
    assert settings.external_graph_gen is False


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="uma_settings"):
        uma_inference_settings("fast")


def test_the_core_refuses_an_unknown_name_before_reading_any_file():
    # the input files do not exist: the name is checked before they are copied
    with pytest.raises(ValueError, match="uma_settings"):
        MLMMCore(real_pdb="missing.pdb", real_parm7="missing.parm7", real_rst7="missing.rst7",
                 model_pdb="missing_ml.pdb", uma_settings="fast")


def test_the_pysisyphus_calculator_passes_uma_settings_to_the_ml_core(recording_core):
    calc = mlmm_pysis.mlmm(model_charge=0, uma_settings="legacy")

    assert calc.core.kwargs["uma_settings"] == "legacy"


def test_the_pysisyphus_calculator_uses_default_settings_unless_asked(recording_core):
    calc = mlmm_pysis.mlmm(model_charge=0)

    assert calc.core.kwargs["uma_settings"] == "default"
