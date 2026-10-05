"""UMA is loaded with the settings ML/MM needs.

The ML region's composition, charge and spin are fixed for a run, so the
experts can be merged once (merge_mole). torch.compile stays off: it crashes
on macOS CPU, and the Hessian needs a double backward through the model.
"""
from mlmm.mlmm_calc import uma_inference_settings


def test_experts_are_merged_once():
    assert uma_inference_settings().merge_mole is True


def test_model_is_not_compiled():
    assert uma_inference_settings().compile is False
