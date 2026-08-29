"""The baseline ladder: every rung predicts, and predicts something valid.

These tests are deliberately about *contract* rather than *accuracy*. Whether
LightGBM beats logistic regression is a finding to be measured and reported, not
a property to be asserted - a test that pinned it would turn a result into a
requirement and would fail the moment the benchmark honestly changed.

What is asserted here is the machinery: predictions exist, are in range, come
from the training split only, are reproducible under a fixed seed, and survive a
round-trip to disk.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import CostModelConfig, LadderConfig
from rto_sentinel.features.dataset import ModelingDataset
from rto_sentinel.models import (
    RUNG_REGISTRY,
    ArtifactError,
    BlanketCodBlockModel,
    DoNothingModel,
    LightGbmModel,
    LogisticRegressionModel,
    NotFittedError,
    PincodeBlocklistModel,
    resolve_rung,
)
from rto_sentinel.models.base import RiskModel

LEARNED_RUNGS = ("logistic_regression", "lightgbm")
HEURISTIC_RUNGS = ("do_nothing", "blanket_cod_block", "pincode_blocklist")


@pytest.fixture(scope="module")
def fitted(ladder_dataset: ModelingDataset) -> dict[str, RiskModel]:
    """Every rung, fitted once on the training split."""
    train = ladder_dataset.train
    models: dict[str, RiskModel] = {}
    for name, model_class in RUNG_REGISTRY.items():
        model = model_class({"random_state": 11})
        model.fit(train.x, train.y, train.context)
        models[name] = model
    return models


# ---------------------------------------------------------------------------
# every baseline produces predictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RUNG_REGISTRY))
def test_every_baseline_produces_predictions(
    name: str, fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    view = ladder_dataset.validation
    scores = fitted[name].predict_proba(view.x, view.context)

    assert scores.shape == (view.n_rows,)
    assert scores.dtype == np.float64


@pytest.mark.parametrize("name", sorted(RUNG_REGISTRY))
def test_predictions_are_valid_probabilities(
    name: str, fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    view = ladder_dataset.validation
    scores = fitted[name].predict_proba(view.x, view.context)

    assert np.all(np.isfinite(scores)), f"{name} produced non-finite scores"
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


@pytest.mark.parametrize("name", sorted(RUNG_REGISTRY))
def test_predicting_before_fitting_is_refused(name: str, ladder_dataset: ModelingDataset) -> None:
    """A prediction from an unfitted model would be a silent wrong answer."""
    model = RUNG_REGISTRY[name]()
    view = ladder_dataset.validation
    with pytest.raises(NotFittedError):
        model.predict_proba(view.x, view.context)


@pytest.mark.parametrize("name", LEARNED_RUNGS)
def test_a_changed_feature_set_is_refused(
    name: str, fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    """Scoring against different columns than were fitted is caught, not guessed.

    Silently reordering or dropping a column is the classic serving skew: the
    model still returns numbers, and they mean nothing.
    """
    view = ladder_dataset.validation
    mangled = view.x.iloc[:, :-3]
    with pytest.raises(ValueError, match="different feature set"):
        fitted[name].predict_proba(mangled, view.context)


# ---------------------------------------------------------------------------
# each rung behaves the way its docstring claims
# ---------------------------------------------------------------------------


def test_do_nothing_is_constant_at_the_training_base_rate(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    """Rung 0 defines the floor. Its prediction is the base rate, and only that."""
    model = fitted["do_nothing"]
    assert isinstance(model, DoNothingModel)

    view = ladder_dataset.validation
    scores = model.predict_proba(view.x, view.context)

    assert len(np.unique(scores)) == 1
    assert scores[0] == pytest.approx(float(ladder_dataset.train.y.mean()))


def test_blanket_block_flags_exactly_cod_above_the_threshold(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    model = fitted["blanket_cod_block"]
    assert isinstance(model, BlanketCodBlockModel)

    view = ladder_dataset.validation
    scores = model.predict_proba(view.x, view.context)
    expected = (
        view.context["is_cod"].to_numpy(dtype=bool)
        & (view.context["order_value_inr"].to_numpy() > model.value_threshold_inr)
    ).astype(float)

    np.testing.assert_array_equal(scores, expected)
    assert set(np.unique(scores)) <= {0.0, 1.0}


def test_pincode_blocklist_respects_minimum_support(
    ladder_dataset: ModelingDataset,
) -> None:
    """A pincode with too little evidence cannot enter the blocklist.

    Without this a place acquires a reputation from three deliveries, which is
    noise for the merchant and unfairness for the customers who live there.
    """
    train = ladder_dataset.train
    model = PincodeBlocklistModel({"top_decile": 0.10, "min_support": 25})
    model.fit(train.x, train.y, train.context)

    counts = train.context["pincode"].value_counts()
    for pincode in model.blocklist_:
        assert counts.get(pincode, 0) >= 25


def test_pincode_blocklist_is_built_from_training_data_only(
    ladder_dataset: ModelingDataset,
) -> None:
    """A blocklist fitted on everything would be reading the future.

    Refitting on training-plus-validation must change it. If it did not, the
    original fit was seeing more than the training split.
    """
    train, validation = ladder_dataset.train, ladder_dataset.validation

    from_train = PincodeBlocklistModel({"top_decile": 0.10, "min_support": 10})
    from_train.fit(train.x, train.y, train.context)

    combined_x = pd.concat([train.x, validation.x])
    combined_y = pd.concat([train.y, validation.y])
    combined_context = pd.concat([train.context, validation.context])
    from_both = PincodeBlocklistModel({"top_decile": 0.10, "min_support": 10})
    from_both.fit(combined_x, combined_y, combined_context)

    assert from_train.blocklist_ != from_both.blocklist_, (
        "adding validation rows did not change the blocklist, which suggests the "
        "training fit was not confined to the training split"
    )


def test_heuristic_rungs_need_the_operational_context(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    """Rungs 1 and 2 score on raw order data, and say so when it is missing."""
    view = ladder_dataset.validation
    for name in ("blanket_cod_block", "pincode_blocklist"):
        with pytest.raises(ValueError, match="context frame"):
            fitted[name].predict_proba(view.x, None)


def test_learned_rungs_ignore_context(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset
) -> None:
    """The ML rungs must not touch raw pincode, even though it is passed alongside.

    This is the test that makes the ``context`` argument safe: it exists so the
    heuristic baselines can use information the learned models are forbidden, and
    this asserts the learned models genuinely do not read it.
    """
    view = ladder_dataset.validation
    for name in LEARNED_RUNGS:
        with_context = fitted[name].predict_proba(view.x, view.context)
        without_context = fitted[name].predict_proba(view.x, None)
        np.testing.assert_array_equal(with_context, without_context)


def test_no_learned_rung_receives_a_forbidden_column(
    ladder_dataset: ModelingDataset,
) -> None:
    """The design matrix itself must be clean, independently of the context frame."""
    from rto_sentinel.data import schema as cols

    leaked = set(ladder_dataset.features.columns) & cols.FORBIDDEN_IN_FEATURES
    assert not leaked
    assert "pincode" not in ladder_dataset.features.columns


# ---------------------------------------------------------------------------
# train/test separation
# ---------------------------------------------------------------------------


def test_fitting_never_sees_the_evaluation_split(ladder_dataset: ModelingDataset) -> None:
    """A model fitted on train alone must differ from one fitted on train+validation.

    Indirect but strong: if adding validation rows leaves predictions identical,
    either the model ignores its training data or the splits overlap.
    """
    train, validation = ladder_dataset.train, ladder_dataset.validation

    on_train = LogisticRegressionModel({"random_state": 3})
    on_train.fit(train.x, train.y, train.context)

    on_both = LogisticRegressionModel({"random_state": 3})
    on_both.fit(
        pd.concat([train.x, validation.x]),
        pd.concat([train.y, validation.y]),
        pd.concat([train.context, validation.context]),
    )

    assert not np.allclose(
        on_train.predict_proba(validation.x, validation.context),
        on_both.predict_proba(validation.x, validation.context),
    )


def test_the_test_split_stays_sealed_during_training(
    ladder_dataset: ModelingDataset,
) -> None:
    """Training touches train and validation. Nothing reaches for the sealed set."""
    from rto_sentinel.features.dataset import TestSetAccessError

    assert ladder_dataset.test_is_sealed
    with pytest.raises(TestSetAccessError):
        _ = ladder_dataset.test


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LEARNED_RUNGS)
def test_the_same_seed_reproduces_the_same_predictions(
    name: str, ladder_dataset: ModelingDataset
) -> None:
    """Two fits with one seed must agree exactly.

    LightGBM needs ``deterministic=True`` and a single thread for this to hold;
    its default multithreaded histogram build is not bit-reproducible. That
    setting costs wall-clock time and buys this property, and this is the test
    that makes it a fact rather than an intention.
    """
    train, validation = ladder_dataset.train, ladder_dataset.validation
    model_class = RUNG_REGISTRY[name]

    first = model_class({"random_state": 5})
    first.fit(train.x, train.y, train.context)
    second = model_class({"random_state": 5})
    second.fit(train.x, train.y, train.context)

    np.testing.assert_array_equal(
        first.predict_proba(validation.x, validation.context),
        second.predict_proba(validation.x, validation.context),
    )


def test_heuristic_rungs_are_trivially_reproducible(
    ladder_dataset: ModelingDataset,
) -> None:
    train, validation = ladder_dataset.train, ladder_dataset.validation
    for name in HEURISTIC_RUNGS:
        model_class = RUNG_REGISTRY[name]
        first, second = model_class(), model_class()
        first.fit(train.x, train.y, train.context)
        second.fit(train.x, train.y, train.context)
        np.testing.assert_array_equal(
            first.predict_proba(validation.x, validation.context),
            second.predict_proba(validation.x, validation.context),
        )


def test_a_different_seed_changes_the_tree_model(ladder_dataset: ModelingDataset) -> None:
    """Guards against a model that quietly ignores its seed.

    Only asserted for LightGBM: logistic regression with lbfgs is essentially
    deterministic regardless of seed, so requiring it to differ would be
    requiring a bug.
    """
    train, validation = ladder_dataset.train, ladder_dataset.validation
    first = LightGbmModel({"random_state": 5, "subsample": 0.7})
    first.fit(train.x, train.y, train.context)
    second = LightGbmModel({"random_state": 6, "subsample": 0.7})
    second.fit(train.x, train.y, train.context)

    assert not np.array_equal(
        first.predict_proba(validation.x, validation.context),
        second.predict_proba(validation.x, validation.context),
    )


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RUNG_REGISTRY))
def test_a_saved_model_reloads_and_predicts_identically(
    name: str,
    fitted: dict[str, RiskModel],
    ladder_dataset: ModelingDataset,
    tmp_path: Path,
    model_card_factory,
) -> None:
    """The API will load these artefacts. They must survive the round trip exactly."""
    model = fitted[name]
    view = ladder_dataset.validation
    before = model.predict_proba(view.x, view.context)

    directory = tmp_path / f"{name}__v1"
    model.save(directory, model_card_factory(name, model))

    reloaded, card = type(model).load(directory)
    after = reloaded.predict_proba(view.x, view.context)

    np.testing.assert_array_equal(before, after)
    assert card.model_name == name


def test_a_tampered_artefact_is_refused(
    fitted: dict[str, RiskModel], tmp_path: Path, model_card_factory
) -> None:
    """The checksum must actually be checked.

    Loading a joblib file executes pickle. A corrupted or substituted artefact
    producing predictions nobody intended is worse than a hard failure.
    """
    model = fitted["do_nothing"]
    directory = tmp_path / "tampered"
    model.save(directory, model_card_factory("do_nothing", model))

    (directory / "model.joblib").write_bytes(b"not a model")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        DoNothingModel.load(directory)


def test_an_unfitted_model_cannot_be_saved(tmp_path: Path, model_card_factory) -> None:
    model = LogisticRegressionModel()
    with pytest.raises(NotFittedError):
        model.save(tmp_path / "unfitted", model_card_factory("logistic_regression", model))


def test_the_card_records_full_provenance(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset, tmp_path: Path
) -> None:
    """Six versions travel with the artefact, and calibration is declared absent."""
    from rto_sentinel.models.artifacts import read_card

    model = fitted["lightgbm"]
    metadata = ladder_dataset.metadata
    from rto_sentinel.contracts.risk import ModelCard

    card = ModelCard(
        model_name="lightgbm",
        model_version="testv1",
        rung_id=4,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        training_rows=ladder_dataset.train.n_rows,
        feature_names=tuple(ladder_dataset.train.x.columns),
        enabled_families=metadata.families_used,
        random_seed=11,
        config_fingerprint=metadata.config_fingerprint,
        feature_fingerprint=metadata.feature_fingerprint,
        feature_version=metadata.feature_version,
        dataset_run_id=metadata.dataset_run_id,
        generator_version=metadata.generator_version,
    )
    directory = tmp_path / "provenance"
    model.save(directory, card)

    # Readable without unpickling anything.
    loaded = read_card(directory)
    assert loaded.feature_fingerprint == metadata.feature_fingerprint
    assert loaded.dataset_run_id == metadata.dataset_run_id
    assert loaded.generator_version == metadata.generator_version
    assert loaded.calibration_method is None, "Phase 4 models must declare no calibration"


def test_provenance_warnings_flag_a_stale_or_uncalibrated_model(
    fitted: dict[str, RiskModel], ladder_dataset: ModelingDataset, tmp_path: Path
) -> None:
    from rto_sentinel.contracts.risk import ModelCard
    from rto_sentinel.models import verify_provenance

    card = ModelCard(
        model_name="lightgbm",
        model_version="v1",
        rung_id=4,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        training_rows=10,
        feature_names=("a",),
        enabled_families=("x",),
        random_seed=1,
        config_fingerprint="old-fingerprint",
    )
    warnings = verify_provenance(card, config_fingerprint="new-fingerprint")

    assert any("different configuration" in w or "trained under" in w for w in warnings)
    assert any("UNCALIBRATED" in w for w in warnings)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_every_enabled_config_rung_has_an_implementation(
    ladder_config: LadderConfig,
) -> None:
    for rung in ladder_config.rungs:
        if rung.enabled:
            assert resolve_rung(rung.name) is not None


def test_an_unknown_rung_fails_with_a_useful_message() -> None:
    from rto_sentinel.models import UnknownRungError

    with pytest.raises(UnknownRungError, match="known rungs are"):
        resolve_rung("gradient_boosted_wishful_thinking")


def test_rung_ids_match_the_configuration(ladder_config: LadderConfig) -> None:
    """A mismatch would put the wrong row in the comparison table."""
    for rung in ladder_config.rungs:
        if not rung.enabled:
            continue
        assert resolve_rung(rung.name).rung_id == rung.id


def test_the_disabled_text_rung_is_not_registered(ladder_config: LadderConfig) -> None:
    """Rung 5 is unimplemented; registering it would fail confusingly and late."""
    assert "lightgbm_address_text" not in RUNG_REGISTRY
    rung5 = next(r for r in ladder_config.rungs if r.id == 5)
    assert not rung5.enabled


def test_cost_profiles_all_yield_a_usable_threshold(cost_config: CostModelConfig) -> None:
    """Every shipped profile must derive a threshold strictly inside (0, 1)."""
    from rto_sentinel.decision.threshold import derive_threshold
    from rto_sentinel.models.experiment import cost_inputs_from_profile

    for name, profile in cost_config.profiles.items():
        derivation = derive_threshold(cost_inputs_from_profile(profile))
        assert 0.0 < derivation.threshold < 1.0, name
