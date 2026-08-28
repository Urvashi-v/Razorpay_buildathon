"""Rung 4: LightGBM.

**The question this answers: what does the stronger nonlinear model achieve?**

The expected winner on ranking. It handles interactions and missingness natively,
which matters here because "no customer history" is a real state rather than a
zero, and forcing it to a number would teach the model something false.

THIS RUNG IS NOT CALIBRATED, AND THAT IS THE POINT OF PHASE 5
=============================================================
``predict_proba`` returns raw boosting output. It ranks well and it is **not an
honest probability** - gradient boosting systematically pushes scores toward the
extremes.

That matters more here than in most projects, because the entire decision layer
is an expected-value comparison against a derived threshold. If the model says
0.30 and the true rate for that bucket is 0.55, the threshold is wrong and every
rupee figure downstream is fiction.

So Phase 4 *measures* the calibration error and reports it; Phase 5 fits isotonic
regression on the validation fold to fix it. The model card's
``calibration_method`` stays ``None`` until then, and the decision engine refuses
a score whose calibration method is None. The uncalibrated model physically
cannot reach a decision.

CLASS IMBALANCE
===============
RTO on COD runs around one in four - mild by fraud standards. SPEC section 05 is
explicit: no SMOTE, no aggressive resampling, ``scale_pos_weight`` if anything.
Synthetic minority oversampling on tabular risk data manufactures optimism, and
the config validator refuses ``smote: true`` outright.

DETERMINISM
===========
``random_state`` is fixed and ``deterministic=True`` with ``num_threads=1`` is
set, because LightGBM's default multithreaded histogram construction is not
bit-reproducible. That costs wall-clock time on a large fit and buys the property
that two runs with the same seed produce the same model - which the reproducibility
test asserts, and without which "same seed, same result" would be a claim rather
than a fact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from rto_sentinel.models.base import RiskModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.risk import FeatureContribution

DEFAULTS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0,
}


class LightGbmModel(RiskModel):
    """Gradient-boosted trees over the design matrix, categoricals included natively."""

    rung_id = 4
    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.booster_: lgb.LGBMClassifier | None = None
        self.categorical_features_: tuple[str, ...] = ()

    def _hyperparameters(self) -> dict[str, Any]:
        settings = {**DEFAULTS, **{k: v for k, v in self.params.items() if k in DEFAULTS}}
        settings.update(
            {
                "random_state": int(self.params.get("random_state", 0)),
                # Determinism over speed. See the module docstring.
                "deterministic": True,
                "force_row_wise": True,
                "num_threads": 1,
                "verbose": -1,
                # subsample only takes effect with a frequency set; without this
                # the parameter is silently ignored and the run is not what the
                # recorded hyperparameters claim.
                "subsample_freq": 1,
            }
        )
        return settings

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        prepared = self._prepare(x, fitting=True)
        self.booster_ = lgb.LGBMClassifier(**self._hyperparameters())
        self.booster_.fit(
            prepared,
            y.astype(int),
            categorical_feature=list(self.categorical_features_) or "auto",
        )

    def _prepare(self, x: pd.DataFrame, *, fitting: bool = False) -> pd.DataFrame:
        """Cast categoricals so LightGBM handles them natively.

        No imputation: NaN is passed straight through, because LightGBM learns a
        default direction for missing values at each split. That is the whole
        reason "no history" is encoded as NaN rather than 0 upstream.
        """
        prepared = x.copy()
        if fitting:
            self.categorical_features_ = tuple(
                column
                for column in x.columns
                if isinstance(x[column].dtype, pd.CategoricalDtype) or x[column].dtype == object
            )
        for column in self.categorical_features_:
            prepared[column] = prepared[column].astype("category")
        return prepared

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        if self.booster_ is None:  # pragma: no cover - guarded by base class
            msg = "LightGBM booster is missing"
            raise RuntimeError(msg)
        proba = np.asarray(self.booster_.predict_proba(self._prepare(x)))
        return np.asarray(proba[:, 1], dtype="float64")

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, largest first.

        Gain rather than split count: split count rewards high-cardinality
        features for being splittable, which says more about the column's
        cardinality than its usefulness.
        """
        if self.booster_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        importance = self.booster_.booster_.feature_importance(importance_type="gain")
        return pd.Series(importance, index=self.feature_names_).sort_values(ascending=False)

    def explain(self, x: pd.DataFrame, top_k: int = 5) -> list[list[FeatureContribution]]:
        """Per-row SHAP contributions.

        Not used in Phase 4 - reason codes are Phase 6 - but implemented here
        because it is a property of this rung rather than of the decision layer,
        and because the harness must be able to call it uniformly.
        """
        from rto_sentinel.contracts.risk import FeatureContribution

        if self.booster_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)

        prepared = self._prepare(x)
        raw = self.booster_.booster_.predict(prepared, pred_contrib=True)
        contributions = np.asarray(raw)[:, :-1]  # final column is the base value

        rows: list[list[FeatureContribution]] = []
        for index in range(len(x)):
            order = np.argsort(-np.abs(contributions[index]))[:top_k]
            rows.append(
                [
                    FeatureContribution(
                        feature=self.feature_names_[position],
                        family=self.feature_names_[position].split("_")[0],
                        value=_scalar(x.iloc[index, position]),
                        contribution=float(contributions[index][position]),
                    )
                    for position in order
                ]
            )
        return rows

    def _state(self) -> dict[str, Any]:
        return {
            **super()._state(),
            "booster": self.booster_,
            "categorical_features": self.categorical_features_,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        self.booster_ = state["booster"]
        self.categorical_features_ = tuple(state["categorical_features"])


def _scalar(value: Any) -> float | str | bool | None:
    """Coerce a cell into something the contract accepts."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)
