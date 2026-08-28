"""The baseline ladder, rungs 0-4, plus artefacts and the experiment runner.

Depends on: configuration, contracts, features, decision (for the cost-derived
threshold), eval. Knows nothing about the API, the database or the agent layer.
A model produces a probability and stops there.

EVERY RUNG IN PHASE 4 IS UNCALIBRATED. ``ModelCard.calibration_method`` is None,
and the decision engine refuses a score whose calibration method is None - so an
uncalibrated model physically cannot reach a decision. Isotonic calibration on the
validation fold is Phase 5.
"""

from rto_sentinel.models.artifacts import (
    ArtifactError,
    artifact_dir,
    list_artifacts,
    load_artifact,
    read_card,
    save_artifact,
    verify_provenance,
)
from rto_sentinel.models.base import HeuristicModel, NotFittedError, RiskModel
from rto_sentinel.models.calibration import (
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
)
from rto_sentinel.models.experiment import (
    EXPERIMENT_VERSION,
    TrainedRung,
    run_ladder,
    save_results,
    scores_frame,
    train_rung,
)
from rto_sentinel.models.registry import RUNG_REGISTRY, UnknownRungError, resolve_rung
from rto_sentinel.models.rung0_do_nothing import DoNothingModel
from rto_sentinel.models.rung1_blanket_block import BlanketCodBlockModel
from rto_sentinel.models.rung2_pincode_blocklist import PincodeBlocklistModel
from rto_sentinel.models.rung3_logistic import LogisticRegressionModel
from rto_sentinel.models.rung4_lightgbm import LightGbmModel

__all__ = [
    "EXPERIMENT_VERSION",
    "RUNG_REGISTRY",
    "ArtifactError",
    "BlanketCodBlockModel",
    "Calibrator",
    "DoNothingModel",
    "HeuristicModel",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "LightGbmModel",
    "LogisticRegressionModel",
    "NotFittedError",
    "PincodeBlocklistModel",
    "RiskModel",
    "TrainedRung",
    "UnknownRungError",
    "artifact_dir",
    "list_artifacts",
    "load_artifact",
    "read_card",
    "resolve_rung",
    "run_ladder",
    "save_artifact",
    "save_results",
    "scores_frame",
    "train_rung",
    "verify_provenance",
]
