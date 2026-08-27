"""Probability calibration - the component the decision layer depends on.

SPEC section 05: "The entire decision layer depends on the score being an honest
probability. If the model says 0.30 and the true rate for that bucket is 0.55,
the expected-value threshold is wrong and the rupee numbers are fiction."

That sentence is the reason this module exists as a first-class component rather
than a two-line postprocessing step. Three rules:

1. **Fitted on validation, never on train.** A calibrator fitted on the training
   fold is calibrated to the model's own overfitting and reports a flattering
   ECE that will not survive contact with new data.
2. **Never fitted on test.** Fitting anything on the sealed set is the exact
   failure mode SPEC section 03 was written to prevent.
3. **Declared on the output.** A :class:`~rto_sentinel.contracts.risk.RiskScore`
   records which method calibrated it, and the decision engine refuses to act on
   a score whose ``calibration_method`` is None. An uncalibrated probability
   cannot reach a threshold comparison by accident.

STATUS: Phase 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

CalibrationMethod = Literal["isotonic", "platt", "none"]


class Calibrator(ABC):
    """Maps raw model scores to honest probabilities."""

    method: CalibrationMethod

    @abstractmethod
    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        """Fit on the VALIDATION fold's scores and labels. Never train, never test."""

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``."""


class IsotonicCalibrator(Calibrator):
    """Isotonic regression on the validation fold. The default for rung 4.

    Chosen over Platt scaling because boosting distortion is not reliably
    sigmoid-shaped, and isotonic makes no parametric assumption. The cost is that
    it needs enough validation rows to be stable - with 21 days of validation
    data that is comfortably satisfied here, and the reliability diagram in the
    report is the check that it was.
    """

    method: CalibrationMethod = "isotonic"

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        raise NotImplementedError("Isotonic calibration lands in Phase 3.")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Isotonic calibration lands in Phase 3.")


class IdentityCalibrator(Calibrator):
    """Passes scores through unchanged, for rungs already calibrated by construction.

    Logistic regression on the right link function is calibrated by
    construction, and the do-nothing baseline emits a base rate. Both still route
    through a calibrator so that the pipeline has exactly one code path.
    """

    method: CalibrationMethod = "none"

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        return None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return scores
