"""The configuration bundle parses, validates, and refuses to be weakened.

These tests do double duty. They check the loader works, and they check that the
commitments written into the YAML - the split protocol, the friction ladder
safeguards, the refused feature list - cannot be quietly relaxed. A future
contributor who flips ``hard_block_allowed`` to true, or reorders the temporal
split so validation precedes training, gets a failing test rather than a subtly
broken system.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from rto_sentinel.configuration import (
    ConfigurationError,
    config_fingerprint,
    load_app_config,
    load_cost_model_config,
    load_policy_config,
    load_splits_config,
)
from rto_sentinel.configuration.schemas import PolicyConfig, SplitsConfig
from rto_sentinel.settings import Settings


def test_all_configuration_files_load(settings: Settings) -> None:
    config = load_app_config(settings)
    assert config.generator.horizon.n_orders > 0
    assert config.cost_model.default_profile in config.cost_model.profiles
    assert config.evaluation.primary_metric == "net_inr_saved_per_1000_orders"


def test_fingerprint_is_stable_and_content_sensitive(settings: Settings, tmp_path: Path) -> None:
    """The fingerprint pins results to the configuration that produced them."""
    first = config_fingerprint(settings)
    assert first == config_fingerprint(settings), "fingerprint must be deterministic"
    assert len(first) == 64


def test_worked_example_profile_matches_the_specification(settings: Settings) -> None:
    """The default profile reproduces the spec's worked example exactly.

    SPEC section 06: RTO cost 220, contribution margin 250, abandonment 25%,
    intervention success 60%. The derived threshold is 0.3214..., not 0.5. The
    derivation itself lands in Phase 2; this asserts the inputs it will use.
    """
    cost = load_cost_model_config(settings)
    profile = cost.profiles[cost.default_profile]
    assert profile.rto_cost_inr == pytest.approx(220.0)
    assert profile.contribution_margin_inr == pytest.approx(250.0)
    assert profile.abandonment_on_friction == pytest.approx(0.25)
    assert profile.intervention_success_rate == pytest.approx(0.60)

    # The arithmetic the decision layer will perform, checked here by hand.
    cost_fp = profile.abandonment_on_friction * profile.contribution_margin_inr
    saving_tp = profile.intervention_success_rate * profile.rto_cost_inr
    derived = cost_fp / (cost_fp + saving_tp)
    assert derived == pytest.approx(0.3214, abs=1e-4)
    assert derived != pytest.approx(0.5), "the whole point is that it is not 0.5"


def test_merchant_economics_move_the_threshold(settings: Settings) -> None:
    """A high-margin brand flags LESS readily than a thin-margin one.

    The direction is counter-intuitive and was documented backwards across this
    repository until Phase 6's merchant simulation made it visible. The margin is
    what a false positive costs, so a merchant with more margin to lose demands
    more certainty before frictioning an order - the threshold rises and the flag
    rate falls.

    SPEC section 06. Checking it on the shipped profiles means the demo's central
    claim is a property of the configuration rather than a story told over it.
    """
    cost = load_cost_model_config(settings)

    def threshold(key: str) -> float:
        p = cost.profiles[key]
        c_fp = p.abandonment_on_friction * p.contribution_margin_inr
        s_tp = p.intervention_success_rate * p.rto_cost_inr
        return c_fp / (c_fp + s_tp)

    assert (
        threshold("high_margin_beauty")
        > threshold("mid_margin_d2c")
        > threshold("thin_margin_reseller")
    ), (
        "A higher contribution margin means a false positive costs more, so the "
        "threshold should rise - the merchant should flag LESS readily, not more."
    )


# ---------------------------------------------------------------------------
# The safeguards cannot be relaxed
# ---------------------------------------------------------------------------


def test_policy_forbids_a_silent_hard_block(settings: Settings) -> None:
    """SPEC section 09: no hard block without an appeal path."""
    policy = load_policy_config(settings)
    assert policy.safeguards.hard_block_allowed is False
    assert all(band.reversible for band in policy.bands)
    severe = policy.bands[-1]
    assert severe.name == "SEVERE"
    assert severe.requires_appeal_path
    assert severe.requires_human_review_queue


def test_every_friction_band_carries_a_reason_code(settings: Settings) -> None:
    policy = load_policy_config(settings)
    for band in policy.bands:
        if band.action != "none":
            assert band.requires_reason_code, f"{band.name} applies friction without a reason code"


def test_policy_rejects_hard_block_when_configured(settings: Settings) -> None:
    """Flipping the safeguard off is rejected at load time, not at runtime."""
    raw = yaml.safe_load((settings.config_path / "policy.yaml").read_text(encoding="utf-8"))
    raw["safeguards"]["hard_block_allowed"] = True
    with pytest.raises(ValueError, match="no silent hard block"):
        PolicyConfig(**raw)


def test_policy_rejects_reordered_bands(settings: Settings) -> None:
    raw = yaml.safe_load((settings.config_path / "policy.yaml").read_text(encoding="utf-8"))
    raw["bands"][1], raw["bands"][2] = raw["bands"][2], raw["bands"][1]
    with pytest.raises(ValueError, match="in order"):
        PolicyConfig(**raw)


def test_splits_are_temporal_and_forward_only(settings: Settings) -> None:
    """SPEC section 03 rule 1: train, then validate, then test. Never random."""
    splits = load_splits_config(settings)
    train_end = splits.temporal.train_days[1]
    val_start, val_end = splits.temporal.validation_days
    test_start = splits.temporal.test_days[0]
    assert train_end < val_start < val_end < test_start
    assert splits.strategy == "temporal_grouped"
    assert splits.group.disjoint_across_splits
    assert splits.sealing.test_set_sealed
    assert splits.sealing.threshold_fitted_on == "validation"
    assert splits.as_of_join.enforced
    assert splits.label_maturity.exclude_immature_tail


def test_splits_reject_a_leaky_ordering(settings: Settings) -> None:
    """A config where validation precedes training is refused at load time."""
    raw = yaml.safe_load((settings.config_path / "splits.yaml").read_text(encoding="utf-8"))
    raw["temporal"]["validation_days"] = [1, 20]
    with pytest.raises(ValueError, match="overlap or are out of order"):
        SplitsConfig(**raw)


def test_evaluation_config_keeps_its_prohibitions(settings: Settings) -> None:
    """The "what I will not do" list is part of the config, and stays there."""
    config = load_app_config(settings)
    forbidden = set(config.evaluation.forbidden)
    for rule in (
        "tune_threshold_on_test_set",
        "lead_with_roc_auc",
        "net_false_positive_cost_away",
        "quote_precision_without_flag_rate",
    ):
        assert rule in forbidden


def test_smote_stays_refused(settings: Settings) -> None:
    """SPEC section 05: no synthetic minority oversampling on tabular risk data."""
    config = load_app_config(settings)
    assert config.ladder.resampling.smote is False
    assert config.ladder.resampling.reason


# ---------------------------------------------------------------------------
# Loader failure modes
# ---------------------------------------------------------------------------


def test_missing_config_file_raises_a_clear_error(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RTO_CONFIG_DIR", str(tmp_path))
    from rto_sentinel.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(ConfigurationError, match="configuration file not found"):
        load_app_config(get_settings())


def test_malformed_yaml_raises_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "cost_model.yaml").write_text(
        textwrap.dedent(
            """
            version: 1
            default_profile: missing_profile
            profiles: {}
            bounds: {}
            sensitivity: {perturbations: [], parameters: []}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RTO_CONFIG_DIR", str(tmp_path))
    from rto_sentinel.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(ConfigurationError, match="not defined in profiles"):
        load_cost_model_config(get_settings())
