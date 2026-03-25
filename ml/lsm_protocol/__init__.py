"""LSM-ML evaluation protocol package.

This package splits the former monolithic script into focused modules:
- ``models``: data structures
- ``core``: numerical training/evaluation logic
- ``experiments``: scenario sweeps and aggregation
- ``plots``: figure generation
- ``api``: public protocol runner
"""

import sys
from pathlib import Path

# Ensure repository root is importable when called from notebooks/scripts.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .api import run_evaluation_protocol
from .core import (
    evaluate_lsm_exercise_policy,
    evaluate_policy,
    train_lsm_exercise_policy,
    train_lsm_policy,
)
from .models import LSMPolicy, OptionScenario, StepModel
from .plots import generate_academic_error_surface_from_csv, generate_protocol_plots

__all__ = [
    "LSMPolicy",
    "OptionScenario",
    "StepModel",
    "evaluate_lsm_exercise_policy",
    "evaluate_policy",
    "generate_academic_error_surface_from_csv",
    "generate_protocol_plots",
    "run_evaluation_protocol",
    "train_lsm_exercise_policy",
    "train_lsm_policy",
]
