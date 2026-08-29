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

HOW THE METHOD IS CHOSEN
========================
Not by fitting each candidate on validation and reading its error off the same
rows - that scores every method on data it has already seen, and isotonic
regression in particular can drive that number to near zero while generalising
worse than doing nothing. :func:`compare_calibrators` instead uses **K-fold
cross-validation inside the validation split**: each candidate is fitted on K-1
folds and scored on the fold it did not see, and the out-of-fold predictions are
assembled into one honest set of calibrated scores.

The chosen method is then refitted on the whole validation split for the shipped
artefact - standard practice, and the reason the cross-validated number is the
one reported rather than the refit's own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from rto_sentinel.eval.metrics import calibration_metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from rto_sentinel.contracts.evaluation import CalibrationMetrics

CalibrationMethod = Literal["isotonic", "platt", "none"]

#: Scores are clipped away from the open ends before any logit transform. A raw
#: 0.0 or 1.0 maps to an infinite logit, which no linear model can consume.
_EPSILON = 1e-6


class CalibrationError(RuntimeError):
    """Raised when a calibrator is used in a way that would produce a false claim."""


def _as_float_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype="float64").ravel()
    if array.size and not np.all(np.isfinite(array)):
        msg = "calibration received non-finite scores"
        raise CalibrationError(msg)
    return array


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, _EPSILON, 1.0 - _EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype="float64")


class Calibrator(ABC):
    """Maps raw model scores to honest probabilities."""

    method: CalibrationMethod

    def __init__(self) -> None:
        self._fitted = False
        self.n_fit_rows_ = 0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def _fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        """Method-specific fitting."""

    @abstractmethod
    def _transform(self, scores: np.ndarray) -> np.ndarray:
        """Method-specific mapping."""

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        """Fit on the VALIDATION fold's scores and labels. Never train, never test.

        This class cannot enforce *which* split it is handed - that is the
        caller's contract, asserted in ``tests/unit/test_calibration.py`` and
        recorded on the model card as ``calibration_fitted_on``.
        """
        x = _as_float_array(scores)
        y = np.asarray(y_true).astype(int).ravel()
        if x.shape != y.shape:
            msg = f"scores and labels disagree in length: {x.shape} vs {y.shape}"
            raise CalibrationError(msg)
        if x.size == 0:
            msg = "refusing to fit a calibrator on zero rows"
            raise CalibrationError(msg)

        self._fit(x, y)
        self.n_fit_rows_ = int(x.size)
        self._fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``."""
        if not self._fitted:
            msg = f"{self.method} calibrator has not been fitted"
            raise CalibrationError(msg)
        out = self._transform(_as_float_array(scores))
        # Clipping rather than asserting: isotonic's interpolation can land a
        # hair outside the range through floating point. A genuinely wrong value
        # is caught by the range test, which checks the shape of the mapping.
        return np.asarray(np.clip(np.asarray(out, dtype="float64"), 0.0, 1.0), dtype="float64")

    def state(self) -> dict[str, Any]:
        """Everything needed to reconstruct this calibrator, for the artefact."""
        return {"method": self.method, "fitted": self._fitted, "n_fit_rows": self.n_fit_rows_}

    def restore(self, state: dict[str, Any]) -> None:
        self._fitted = bool(state["fitted"])
        self.n_fit_rows_ = int(state.get("n_fit_rows", 0))


class IsotonicCalibrator(Calibrator):
    """Isotonic regression on the validation fold. The default for rung 4.

    Chosen as the first candidate because boosting distortion is not reliably
    sigmoid-shaped, and isotonic makes no parametric assumption. The cost is that
    it needs enough validation rows to be stable, and that it can only produce as
    many distinct output values as it found steps - which is why it is a
    *candidate* here rather than a foregone conclusion, and why the reliability
    diagram is reported next to the number.
    """

    method: CalibrationMethod = "isotonic"

    def __init__(self) -> None:
        super().__init__()
        self.model_: IsotonicRegression | None = None

    def _fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        # out_of_bounds="clip" so a test score beyond the validation range maps to
        # the nearest fitted value instead of raising at serving time.
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        model.fit(scores, y_true.astype("float64"))
        self.model_ = model

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        if self.model_ is None:  # pragma: no cover - guarded by the base class
            msg = "isotonic calibrator is missing its fitted model"
            raise CalibrationError(msg)
        return np.asarray(self.model_.predict(scores), dtype="float64")

    def state(self) -> dict[str, Any]:
        return {**super().state(), "model": self.model_}

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.model_ = state["model"]


class PlattCalibrator(Calibrator):
    """Platt scaling: a one-parameter sigmoid fitted on the score's logit.

    Classic Platt scaling fits ``sigmoid(a·s + b)`` to an SVM decision value. The
    input here is already a probability, so the transform is applied to
    ``logit(s)`` instead of ``s``. That choice matters: with ``a=1, b=0`` the
    mapping is the identity, so "already calibrated" is inside the hypothesis
    space and a well-calibrated model passes through roughly unchanged. Fitting a
    sigmoid to the raw probability cannot represent the identity, and would
    distort a model that needed no correction.

    Two parameters rather than isotonic's step function, so it is far steadier on
    a small validation split - at the cost of only being able to fix distortions
    that are sigmoid-shaped.
    """

    method: CalibrationMethod = "platt"

    def __init__(self) -> None:
        super().__init__()
        self.model_: LogisticRegression | None = None

    def _fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        if len(np.unique(y_true)) < 2:
            msg = "Platt scaling needs both classes present in the calibration fold"
            raise CalibrationError(msg)
        # C is large, i.e. almost unregularised: this is a 2-parameter fit on
        # thousands of rows, so shrinkage would bias the mapping for no variance
        # benefit worth having.
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(_logit(scores).reshape(-1, 1), y_true)
        self.model_ = model

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        if self.model_ is None:  # pragma: no cover - guarded by the base class
            msg = "platt calibrator is missing its fitted model"
            raise CalibrationError(msg)
        proba = self.model_.predict_proba(_logit(scores).reshape(-1, 1))
        return np.asarray(np.asarray(proba)[:, 1], dtype="float64")

    def state(self) -> dict[str, Any]:
        return {**super().state(), "model": self.model_}

    def restore(self, state: dict[str, Any]) -> None:
        super().restore(state)
        self.model_ = state["model"]


class IdentityCalibrator(Calibrator):
    """Passes scores through unchanged, for rungs already calibrated by construction.

    Logistic regression on the right link function is calibrated by
    construction, and the do-nothing baseline emits a base rate. Both still route
    through a calibrator so that the pipeline has exactly one code path.

    It is also the honest *candidate* to beat: if neither isotonic nor Platt
    improves on leaving the scores alone, leaving them alone is the answer.
    """

    method: CalibrationMethod = "none"

    def _fit(self, scores: np.ndarray, y_true: np.ndarray) -> None:
        return None

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        return scores


#: Every calibration method this project can fit or load.
CALIBRATORS: dict[str, type[Calibrator]] = {
    "none": IdentityCalibrator,
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
}


def build_calibrator(method: str) -> Calibrator:
    """Construct a calibrator by name, refusing an unknown one.

    Refusing rather than defaulting: a typo silently becoming "none" would put an
    uncalibrated model into production wearing a card that says otherwise.
    """
    if method not in CALIBRATORS:
        msg = f"unknown calibration method {method!r}; expected one of {sorted(CALIBRATORS)}"
        raise CalibrationError(msg)
    return CALIBRATORS[method]()


def restore_calibrator(state: dict[str, Any]) -> Calibrator:
    """Rebuild a calibrator from the state stored in an artefact."""
    calibrator = build_calibrator(str(state["method"]))
    calibrator.restore(state)
    return calibrator


def cross_validated_scores(
    method: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold calibrated scores for one method, over the validation split.

    Every returned value was produced by a calibrator that had not seen that row.
    Fitting and scoring on the same rows is what makes isotonic regression look
    perfect on the data it memorised; this is the whole reason the comparison is
    trustworthy.
    """
    x = _as_float_array(scores)
    y = np.asarray(y_true).astype(int).ravel()
    if x.shape != y.shape:
        msg = f"scores and labels disagree in length: {x.shape} vs {y.shape}"
        raise CalibrationError(msg)

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = np.empty_like(x)
    for fit_index, score_index in folds.split(x.reshape(-1, 1), y):
        calibrator = build_calibrator(method)
        calibrator.fit(x[fit_index], y[fit_index])
        out[score_index] = calibrator.transform(x[score_index])
    return out


def compare_calibrators(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    methods: Sequence[str],
    n_folds: int = 5,
    seed: int = 0,
    n_bins: int = 10,
) -> dict[str, CalibrationMetrics]:
    """Cross-validated calibration quality for each candidate method.

    Returns ECE, Brier and the reliability bins per method, all computed from
    out-of-fold predictions. ``"none"`` is included by the caller as the
    do-nothing candidate, and it wins whenever correcting the scores makes them
    worse.
    """
    return {
        method: calibration_metrics(
            np.asarray(y_true).astype(bool),
            cross_validated_scores(method, scores, y_true, n_folds=n_folds, seed=seed),
            n_bins=n_bins,
        )
        for method in methods
    }
