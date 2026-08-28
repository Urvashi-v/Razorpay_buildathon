"""Rung 3: logistic regression.

**The question this answers: what does a simple, interpretable model achieve?**

Interpretable, fast, and calibrated by construction on the right link function -
a genuine shipping candidate rather than a strawman. If it wins on net rupees, it
ships, and that is the ladder working rather than failing.

WHAT THE PREPROCESSING DOES, AND WHY EACH PIECE IS NEEDED
=========================================================
Unlike LightGBM, a linear model cannot handle missing values or categoricals, and
it is sensitive to scale. So:

**Missing values → median, plus an explicit indicator.** Several features here are
NaN precisely because *nothing is known* - a first-time customer has no return
rate. Imputing the median silently converts "unknown" into "typical", which is a
substantive claim about a cohort the model understands least. The added
``missing`` indicator column lets the model learn that absence separately, which
recovers most of what LightGBM gets natively.

**Categoricals → one-hot.** Only three low-cardinality columns (category, device,
tier, courier), so the width cost is small. ``handle_unknown="ignore"`` means an
unseen level at inference produces all-zeros rather than an exception.

**Numerics → standardised.** Required for L2 regularisation to penalise
coefficients comparably rather than punishing whatever happens to be measured in
rupees.

NO MONOTONIC CONSTRAINTS HERE
=============================
``config/models/ladder.yaml`` lists monotonic features for this rung.
scikit-learn's ``LogisticRegression`` has no monotonicity support, so they are
not applied, and pretending otherwise would be worse than saying so. The
constraint is recorded on the feature specs and will be applied to LightGBM in a
later phase where the library supports it. Reported as a known gap rather than
silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rto_sentinel.models.base import RiskModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULTS: dict[str, Any] = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": None,
    "max_iter": 2000,
}

#: Penalties this rung will accept. An unrecognised one is refused rather than
#: silently falling back to L2, because that would make the hyperparameters
#: recorded on the experiment record a lie about what was actually fitted.
#:
#: `l1` and `elasticnet` need a solver that supports them, so they are not in
#: the accepted set until the solver is made configurable alongside them. The
#: ladder asks for `l2`; anything else is a deliberate change, not a default.
SUPPORTED_PENALTIES = ("l2",)

#: The same penalty expressed as an elastic-net mixing ratio: 0 is pure L2.
_PENALTY_AS_L1_RATIO = {"l2": 0.0}

#: scikit-learn 1.8 deprecated ``penalty=`` in favour of ``l1_ratio=``. Before
#: 1.8, passing ``l1_ratio`` alongside a non-elasticnet penalty warns and is
#: ignored; from 1.8, passing ``penalty`` warns instead. The two spellings mean
#: the same fitted model, so this picks whichever the installed version wants
#: rather than emitting a deprecation warning on every training run.
_L1_RATIO_REPLACES_PENALTY = tuple(int(part) for part in sklearn_version.split(".")[:2]) >= (1, 8)


def _penalty_kwargs(penalty: str) -> dict[str, Any]:
    """Spell the penalty the way the installed scikit-learn expects."""
    if _L1_RATIO_REPLACES_PENALTY:
        return {"l1_ratio": _PENALTY_AS_L1_RATIO[penalty]}
    return {"penalty": penalty}


class LogisticRegressionModel(RiskModel):
    """L2 logistic regression over an imputed, scaled, one-hot design matrix."""

    rung_id = 3
    name = "logistic_regression"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.pipeline_: Pipeline | None = None

    def _build_pipeline(self, x: pd.DataFrame) -> Pipeline:
        categorical = [
            column
            for column in x.columns
            if isinstance(x[column].dtype, pd.CategoricalDtype) or x[column].dtype == object
        ]
        numeric = [column for column in x.columns if column not in categorical]

        settings = {**DEFAULTS, **{k: v for k, v in self.params.items() if k in DEFAULTS}}
        if settings["penalty"] not in SUPPORTED_PENALTIES:
            msg = (
                f"unsupported penalty {settings['penalty']!r}; this rung supports "
                f"{list(SUPPORTED_PENALTIES)}. Adding another requires choosing a "
                "solver that supports it."
            )
            raise ValueError(msg)

        return Pipeline(
            [
                (
                    "prepare",
                    ColumnTransformer(
                        [
                            (
                                "numeric",
                                Pipeline(
                                    [
                                        # add_indicator: "unknown" must stay
                                        # distinguishable from "typical".
                                        (
                                            "impute",
                                            SimpleImputer(strategy="median", add_indicator=True),
                                        ),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                numeric,
                            ),
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01),
                                categorical,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        **_penalty_kwargs(settings["penalty"]),
                        C=settings["C"],
                        class_weight=settings["class_weight"],
                        max_iter=settings["max_iter"],
                        random_state=int(self.params.get("random_state", 0)),
                    ),
                ),
            ]
        )

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        self.pipeline_ = self._build_pipeline(x)
        self.pipeline_.fit(x, y.astype(int))

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        if self.pipeline_ is None:  # pragma: no cover - guarded by base class
            msg = "logistic regression pipeline is missing"
            raise RuntimeError(msg)
        return np.asarray(self.pipeline_.predict_proba(x)[:, 1], dtype="float64")

    def coefficients(self) -> pd.Series:
        """Fitted coefficients by expanded feature name, largest magnitude first.

        The reason this rung is worth having: you can read why it decided what it
        decided without a SHAP library.
        """
        if self.pipeline_ is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        names = self.pipeline_.named_steps["prepare"].get_feature_names_out()
        weights = self.pipeline_.named_steps["model"].coef_[0]
        series = pd.Series(weights, index=names)
        return series.reindex(series.abs().sort_values(ascending=False).index)

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "pipeline": self.pipeline_}

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        self.pipeline_ = state["pipeline"]
