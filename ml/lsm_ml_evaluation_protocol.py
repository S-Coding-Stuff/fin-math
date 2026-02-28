"""Backward-compatible façade for the LSM-ML evaluation protocol.

The implementation now lives in ``ml.lsm_protocol``.
This module is kept so existing imports continue to work:

``from ml.lsm_ml_evaluation_protocol import run_evaluation_protocol``
"""

from ml.lsm_protocol import (
    LSMPolicy,
    OptionScenario,
    StepModel,
    evaluate_lsm_exercise_policy,
    evaluate_policy,
    generate_academic_error_surface_from_csv,
    train_lsm_exercise_policy,
    run_evaluation_protocol,
    train_lsm_policy,
)

__all__ = [
    "LSMPolicy",
    "OptionScenario",
    "StepModel",
    "evaluate_lsm_exercise_policy",
    "evaluate_policy",
    "generate_academic_error_surface_from_csv",
    "run_evaluation_protocol",
    "train_lsm_exercise_policy",
    "train_lsm_policy",
]
