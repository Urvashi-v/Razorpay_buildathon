"""Assembles feature families into a design matrix, and polices what gets in.

This module is the single choke point between raw orders and any model. Five
guarantees are enforced here, once, rather than trusted to six separate family
implementations.

**1. Declared availability.** Every emitted feature must declare
``Availability.AT_ORDER_TIME``. A spec that says otherwise is rejected before any
computation happens. This is the cheapest possible leak check and it runs first.

**2. No target leakage.** Nothing in ``data.schema.FORBIDDEN_IN_FEATURES``
reaches the matrix - not the label, not the resolution timestamp, not an identity
column, not the simulator's latent variables.

**3. No refused features.** Nothing matching a pattern in the ``refused`` block of
``config/features.yaml``. Name-derived features, raw pincode categoricals,
protected attributes and cross-merchant signals are rejected by name, with the
configured reason quoted in the error - so whoever hits it reads *why*, not
just *no*.

**4. Emit exactly what was declared.** A family returning a column it did not
declare fails. That is what stops a debugging column nobody audits from becoming
a production feature.

**5. Consistent column order.** Training and inference build the matrix through
the same object, so a reordered column cannot silently change what the model reads
at serving time.

WHY THE PIPELINE SEES EVERY ROW, INCLUDING TEST ROWS
====================================================
``build`` is handed the whole order table, not one split. That looks alarming and
is in fact the correct and safer choice.

Features are computed **as-of each row's own order time**. A training row from day
40 sees only outcomes resolved before day 40, regardless of what else is in the
frame - the as-of machinery does not care whether a later row is labelled "test".

Filtering to a split *before* computing features would be actively worse. A
validation-window order would lose the customer history it genuinely had, and the
model would be trained on a customer who looks like a first-time buyer when the
merchant knew perfectly well they were not. That is a distribution shift
introduced in the name of safety.

``tests/leakage/test_test_set_isolation.py`` verifies the claim rather than
asserting it: it computes features on the full frame and again on a
training-window-only frame, and checks the training rows are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.address import AddressQualityFamily
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.customer_history import CustomerHistoryFamily
from rto_sentinel.features.geography import GeographyRouteFamily
from rto_sentinel.features.order_shape import OrderShapeFamily
from rto_sentinel.features.session_intent import SessionIntentFamily
from rto_sentinel.features.spec import FeatureSet
from rto_sentinel.features.temporal import TemporalFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import FeaturesConfig, GeneratorConfig

#: Behavioural version of the feature engineering itself. Bump when a feature's
#: *definition* changes in a way that makes old and new datasets incomparable.
#: The fingerprint on ``FeatureSet`` catches additions and removals automatically;
#: this is for changes a fingerprint cannot see, such as a bug fix inside a
#: transform that leaves the declaration identical.
FEATURE_VERSION = "1.0.0"

#: Families, in the order they are applied. Registered here rather than
#: discovered, so adding one is a deliberate, reviewable act.
FAMILY_REGISTRY: dict[str, type[FeatureFamily]] = {
    "customer_history": CustomerHistoryFamily,
    "temporal": TemporalFamily,
    "order_shape": OrderShapeFamily,
    "address_quality": AddressQualityFamily,
    "session_intent": SessionIntentFamily,
    "geography_route": GeographyRouteFamily,
}


def _tokenise(name: str) -> list[str]:
    """Split a feature name into lowercase tokens."""
    return [token for token in name.lower().split("_") if token]


def _matches_token_pattern(column: str, pattern: str) -> bool:
    """True when ``pattern``'s tokens appear as a consecutive run in ``column``.

    ``customer_name`` matches ``cust_customer_name_hash``; it does not match
    ``customer_history_count``. ``age`` matches a column named exactly ``age``
    or ``customer_age``; it does not match ``page_seconds``, because ``page`` is
    a different token.
    """
    column_tokens = _tokenise(column)
    pattern_tokens = _tokenise(pattern)
    if not pattern_tokens or len(pattern_tokens) > len(column_tokens):
        return False
    span = len(pattern_tokens)
    return any(
        column_tokens[start : start + span] == pattern_tokens
        for start in range(len(column_tokens) - span + 1)
    )


class RefusedFeatureError(ValueError):
    """Raised when a family tries to emit a feature this project refuses to use.

    The message quotes the reason from ``config/features.yaml`` verbatim, so
    whoever hits it reads *why* rather than just *no*.
    """


class TargetLeakageError(ValueError):
    """Raised when a forbidden column or an unavailable feature reaches the matrix."""


class FeatureContractError(ValueError):
    """Raised when a family emits something other than what it declared."""


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """A design matrix plus the metadata a model artefact needs to reproduce it."""

    matrix: pd.DataFrame
    feature_set: FeatureSet
    families_used: tuple[str, ...]
    feature_version: str
    feature_fingerprint: str

    @property
    def n_rows(self) -> int:
        return len(self.matrix)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_set.names


class FeaturePipeline:
    """Builds the design matrix from the enabled families.

    Constructed from config so that an ablation run - the leave-one-family-out
    study in SPEC section 07 - is a matter of flipping ``enabled`` and rebuilding,
    with no code path that differs between the ablated and full runs.
    """

    def __init__(
        self,
        config: FeaturesConfig,
        generator_config: GeneratorConfig | None = None,
        families: list[FeatureFamily] | None = None,
    ) -> None:
        self.config = config
        self.generator_config = generator_config
        self._families = families if families is not None else self._build_families()

    def _build_families(self) -> list[FeatureFamily]:
        built: list[FeatureFamily] = []
        for name, family_class in FAMILY_REGISTRY.items():
            family_config = self.config.families.get(name)
            if family_config is None:
                msg = (
                    f"family {name!r} is registered in code but absent from "
                    "config/features.yaml; every family must be declared"
                )
                raise FeatureContractError(msg)
            if not family_config.enabled:
                continue
            built.append(family_class(family_config, self.generator_config))
        return built

    @property
    def families(self) -> tuple[FeatureFamily, ...]:
        return tuple(self._families)

    @property
    def feature_set(self) -> FeatureSet:
        """Every feature the enabled families declare, without touching data."""
        combined = FeatureSet()
        for family in self._families:
            combined = combined.merge(family.feature_set)
        return combined

    # ------------------------------------------------------------------
    # guards
    # ------------------------------------------------------------------

    def check_declarations(self) -> None:
        """Validate the declared feature set before any computation happens.

        Runs first and touches no data. A leak caught here is a typo; the same
        leak caught after training is a retracted result.
        """
        feature_set = self.feature_set

        unavailable = feature_set.unavailable_at_prediction_time()
        if unavailable:
            names = [spec.name for spec in unavailable]
            msg = (
                f"features declared unavailable at prediction time cannot be emitted: {names}. "
                "A feature that would not exist at checkout must not be in the design matrix."
            )
            raise TargetLeakageError(msg)

        self.check_refused(list(feature_set.names))
        self.check_forbidden(list(feature_set.names))

    def check_refused(self, columns: list[str]) -> None:
        """Reject any column matching a refused pattern from ``features.yaml``.

        Matching is by WHOLE TOKEN, not substring. Feature names are split on
        underscores and a pattern must match a consecutive run of those tokens.

        Substring matching was the first implementation and was unusable: the
        pattern ``age`` matched ``cust_account_age_days`` and
        ``session_product_page_seconds``, neither of which has anything to do
        with a customer's age. A check that cries wolf gets switched off, which
        is worse than no check at all.

        Genuine collisions that survive token matching are listed in
        ``allowed_exceptions`` with a written justification.

        Called on the declarations and again on the assembled matrix. The
        redundancy is deliberate: this is the check that keeps a fairness
        commitment from decaying into a comment.
        """
        exempt = self.config.exempt_features
        for group in self.config.refused:
            for pattern in group.patterns:
                offenders = [
                    column
                    for column in columns
                    if column.lower() not in exempt and _matches_token_pattern(column, pattern)
                ]
                if offenders:
                    msg = (
                        f"refused feature pattern {pattern!r} matched {offenders}.\n"
                        f"Refusal group {group.id!r}: {group.reason.strip()}"
                    )
                    raise RefusedFeatureError(msg)

    @staticmethod
    def check_forbidden(columns: list[str]) -> None:
        """Reject any column in ``data.schema.FORBIDDEN_IN_FEATURES``."""
        offenders = sorted(set(columns) & cols.FORBIDDEN_IN_FEATURES)
        if offenders:
            msg = (
                f"forbidden columns reached the design matrix: {offenders}. "
                "These are the label, its timestamps, identity columns, or simulator latents."
            )
            raise TargetLeakageError(msg)

    # ------------------------------------------------------------------
    # assembly
    # ------------------------------------------------------------------

    def build(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Run every enabled family and assemble the validated matrix."""
        self.check_declarations()

        if not self._families:
            msg = "no feature families are enabled; the design matrix would be empty"
            raise FeatureContractError(msg)

        missing_inputs = self._missing_source_columns(frame)
        if missing_inputs:
            msg = f"the order table is missing columns the features need: {sorted(missing_inputs)}"
            raise FeatureContractError(msg)

        blocks: list[pd.DataFrame] = []
        for family in self._families:
            declared = family.feature_set.names
            produced = family.transform(frame)

            if list(produced.columns) != list(declared):
                msg = (
                    f"family {family.name!r} emitted {list(produced.columns)} but declared "
                    f"{list(declared)}. A family must emit exactly what it declares."
                )
                raise FeatureContractError(msg)
            if not produced.index.equals(frame.index):
                msg = f"family {family.name!r} returned a frame with a different index"
                raise FeatureContractError(msg)

            blocks.append(produced)

        matrix = pd.concat(blocks, axis=1)

        # Re-run the guards on the assembled result. Cheap, and it catches a
        # family that declared safely and emitted otherwise.
        self.check_refused(list(matrix.columns))
        self.check_forbidden(list(matrix.columns))

        feature_set = self.feature_set
        if list(matrix.columns) != list(feature_set.names):
            msg = "assembled matrix column order does not match the declared feature set"
            raise FeatureContractError(msg)

        return FeatureMatrix(
            matrix=matrix,
            feature_set=feature_set,
            families_used=tuple(family.name for family in self._families),
            feature_version=FEATURE_VERSION,
            feature_fingerprint=feature_set.fingerprint(),
        )

    def _missing_source_columns(self, frame: pd.DataFrame) -> set[str]:
        needed: set[str] = set()
        for spec in self.feature_set:
            needed.update(spec.source_columns)
        # ``signup_at`` arrives by a join from the customers table and is optional:
        # account age degrades to NaN without it rather than failing the build.
        optional = {"signup_at"}
        return (needed - set(frame.columns)) - optional
