"""The Phase 10 endpoints serve saved artefacts, unchanged.

The 501 path is covered in `test_contract_surface.py`. What is covered here is
the opposite: when an artefact exists, the endpoint returns *that artefact* and
does not quietly normalise, round, or fill in any part of it. A monitoring API
that reshapes what the harness measured is a second implementation of the
measurement.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rto_sentinel.api.main import create_app
from rto_sentinel.contracts.evaluation import CohortResult, FairnessAudit
from rto_sentinel.contracts.monitoring import (
    DriftReport,
    DriftSignal,
    EnvironmentSpec,
    ShiftResult,
    ShiftStudy,
    WindowSummary,
)
from rto_sentinel.eval.responsible_report import (
    write_drift_artifacts,
    write_fairness_artifacts,
    write_shift_artifacts,
)
from rto_sentinel.settings import get_settings


@pytest.fixture
def artefacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("RTO_ARTIFACT_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def client(artefacts: Path) -> Iterator[TestClient]:
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def a_cohort(group: str, *, sufficient: bool = True) -> CohortResult:
    return CohortResult(
        cohort="pincode_tier",
        group=group,
        n_orders=500 if sufficient else 12,
        flag_rate=0.28,
        precision=0.515,
        recall=0.613,
        net_inr_per_1000=9482.0,
        rto_rate=0.235,
        n_positives=137,
        n_flagged=163,
        flag_rate_ci=(0.25, 0.32),
        precision_ci=(0.44, 0.59),
        rto_rate_ci=(0.20, 0.27),
        sufficient=sufficient,
        insufficient_reason="" if sufficient else "only 12 orders",
    )


class TestFairnessEndpoint:
    def test_it_returns_the_saved_audit(self, client: TestClient, artefacts: Path) -> None:
        slices = (a_cohort("tier_1"), a_cohort("tier_3"), a_cohort("tier_9", sufficient=False))
        audit = FairnessAudit(
            slices=slices,
            max_flag_rate_ratio=2.21,
            worst_precision_drop=0.036,
            triggered=False,
            narrative="Does not trip the configured review.",
            cohorts_examined=("pincode_tier",),
            groups_below_support=("pincode_tier=tier_9 (12 orders)",),
            min_support=100,
            most_flagged_group="pincode_tier=tier_3",
            least_flagged_group="pincode_tier=tier_1",
        )
        write_fairness_artifacts(
            audit,
            slices,
            artifact_root=artefacts,
            dataset_run_id="run-1",
            split="validation",
            model_version="v-test",
            threshold=0.3481,
        )

        body = client.get("/v1/evaluation/fairness?split=validation").json()

        assert body["model_version"] == "v-test"
        assert body["threshold"] == pytest.approx(0.3481)
        assert body["audit"]["triggered"] is False
        assert body["audit"]["max_flag_rate_ratio"] == pytest.approx(2.21)
        assert len(body["slices"]) == 3

    def test_thin_groups_are_served_rather_than_hidden(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """Suppressing them would hide exactly what the audit exists to look at."""
        slices = (a_cohort("tier_1"), a_cohort("tier_9", sufficient=False))
        write_fairness_artifacts(
            FairnessAudit(
                slices=slices,
                max_flag_rate_ratio=1.0,
                worst_precision_drop=0.0,
                triggered=False,
                groups_below_support=("pincode_tier=tier_9 (12 orders)",),
                min_support=100,
            ),
            slices,
            artifact_root=artefacts,
            dataset_run_id="run-1",
            split="validation",
            model_version="v-test",
            threshold=0.3481,
        )

        body = client.get("/v1/evaluation/fairness").json()
        thin = [entry for entry in body["slices"] if not entry["sufficient"]]

        assert len(thin) == 1
        assert thin[0]["group"] == "tier_9"
        assert body["audit"]["groups_below_support"]

    def test_a_triggered_audit_is_served_exactly_as_a_clean_one(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """Reporting only the flattering runs would defeat the point of an audit."""
        slices = (a_cohort("tier_3"),)
        write_fairness_artifacts(
            FairnessAudit(
                slices=slices,
                max_flag_rate_ratio=4.0,
                worst_precision_drop=0.25,
                triggered=True,
                narrative="Cost transferred without justification.",
                min_support=100,
            ),
            slices,
            artifact_root=artefacts,
            dataset_run_id="run-1",
            split="validation",
            model_version="v-test",
            threshold=0.3481,
        )

        body = client.get("/v1/evaluation/fairness").json()
        assert body["audit"]["triggered"] is True
        assert "without justification" in body["audit"]["narrative"]

    def test_the_response_carries_its_disclaimer(self, client: TestClient, artefacts: Path) -> None:
        """A cohort table must not reach a consumer without the qualifying sentence."""
        slices = (a_cohort("tier_1"),)
        write_fairness_artifacts(
            FairnessAudit(
                slices=slices, max_flag_rate_ratio=1.0, worst_precision_drop=0.0, triggered=False
            ),
            slices,
            artifact_root=artefacts,
            dataset_run_id="run-1",
            split="validation",
            model_version="v-test",
            threshold=0.3481,
        )

        body = client.get("/v1/evaluation/fairness").json()
        assert "NOT evidence of" in body["disclaimer"]
        assert "synthetic" in body["disclaimer"].lower()

    def test_the_two_splits_are_separate_artefacts(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """Validation and sealed test must never be served from the same file."""
        slices = (a_cohort("tier_1"),)
        write_fairness_artifacts(
            FairnessAudit(
                slices=slices, max_flag_rate_ratio=1.0, worst_precision_drop=0.0, triggered=False
            ),
            slices,
            artifact_root=artefacts,
            dataset_run_id="run-1",
            split="validation",
            model_version="v-test",
            threshold=0.3481,
        )

        assert client.get("/v1/evaluation/fairness?split=validation").status_code == 200
        assert client.get("/v1/evaluation/fairness?split=test").status_code == 501


class TestShiftEndpoint:
    def _study(self) -> ShiftStudy:
        return ShiftStudy(
            generated_at=datetime.now(UTC),
            model_version="v-test",
            threshold=0.3481,
            environments=(
                EnvironmentSpec(
                    name="reference", description="control", overrides={}, seed=1, n_orders=8000
                ),
                EnvironmentSpec(
                    name="cod_surge",
                    description="COD share rises",
                    overrides={"payment.cod_share": 0.80},
                    seed=2,
                    n_orders=8000,
                ),
            ),
            results=(
                ShiftResult(
                    environment="reference",
                    n_orders=8766,
                    observed_rto_rate=0.167,
                    pr_auc=0.430,
                    pr_auc_lift=2.57,
                    roc_auc=0.78,
                    brier_score=0.12,
                    expected_calibration_error=0.025,
                    threshold=0.3481,
                    flag_rate=0.181,
                    precision=0.410,
                    recall=0.44,
                    net_inr_per_1000=2276.0,
                ),
                ShiftResult(
                    environment="cod_surge",
                    n_orders=8782,
                    observed_rto_rate=0.214,
                    pr_auc=0.465,
                    pr_auc_lift=2.17,
                    pr_auc_lift_delta=-0.40,
                    pr_auc_delta=0.035,
                    roc_auc=0.77,
                    brier_score=0.14,
                    expected_calibration_error=0.034,
                    ece_delta=0.009,
                    threshold=0.3481,
                    flag_rate=0.248,
                    precision=0.430,
                    recall=0.50,
                    net_inr_per_1000=4122.0,
                    net_delta=1846.0,
                ),
            ),
            findings=("cod_surge: ranking lift fell by 0.40x",),
        )

    def test_it_returns_the_saved_study(self, client: TestClient, artefacts: Path) -> None:
        write_shift_artifacts(self._study(), artifact_root=artefacts)

        body = client.get("/v1/evaluation/shift").json()

        assert body["threshold"] == pytest.approx(0.3481)
        assert len(body["results"]) == 2
        assert body["findings"]

    def test_the_lift_column_survives_serialisation(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """The whole point of the lift column is that a consumer can read it.

        Raw PR-AUC rose in this environment while lift fell. A response that
        dropped the lift field would leave the console reporting that the model
        improved when the world got riskier.
        """
        write_shift_artifacts(self._study(), artifact_root=artefacts)

        body = client.get("/v1/evaluation/shift").json()
        surge = next(r for r in body["results"] if r["environment"] == "cod_surge")

        assert surge["pr_auc_delta"] > 0
        assert surge["pr_auc_lift_delta"] < 0

    def test_the_reference_environment_carries_no_deltas(
        self, client: TestClient, artefacts: Path
    ) -> None:
        write_shift_artifacts(self._study(), artifact_root=artefacts)

        body = client.get("/v1/evaluation/shift").json()
        reference = next(r for r in body["results"] if r["environment"] == "reference")

        assert reference["pr_auc_delta"] is None
        assert reference["net_delta"] is None

    def test_the_written_artefact_carries_a_disclaimer_the_contract_does_not(
        self, artefacts: Path, client: TestClient
    ) -> None:
        """The file is annotated for humans; the API contract stays clean.

        `extra="forbid"` on the contract means the annotation must be stripped on
        read - which is what `read_contract_payload` exists to do, and what a 500
        here would mean somebody forgot.
        """
        paths = write_shift_artifacts(self._study(), artifact_root=artefacts)
        raw = json.loads(paths[0].read_text(encoding="utf-8"))

        assert "disclaimer" in raw
        assert client.get("/v1/evaluation/shift").status_code == 200


class TestDriftEndpoint:
    def _report(self, *, labels: bool) -> DriftReport:
        return DriftReport(
            generated_at=datetime.now(UTC),
            baseline=WindowSummary(
                label="baseline", n_orders=1220, n_matured=1220 if labels else 0
            ),
            current=WindowSummary(label="current", n_orders=814, n_matured=814 if labels else 0),
            signals=(
                DriftSignal(
                    name="order_value_inr",
                    kind="feature",
                    statistic="psi",
                    distance=0.1016,
                    severity="watch",
                    baseline_n=1220,
                    current_n=814,
                ),
            ),
            warnings=("Input drift is expected in a seasonal business.",),
            model_version="v-test",
            labels_available=labels,
        )

    def test_it_returns_the_saved_report(self, client: TestClient, artefacts: Path) -> None:
        write_drift_artifacts(
            self._report(labels=True), artifact_root=artefacts, dataset_run_id="run-1"
        )

        body = client.get("/v1/monitoring/drift").json()

        assert body["labels_available"] is True
        assert body["signals"][0]["severity"] == "watch"
        assert body["warnings"]

    def test_a_report_with_no_labels_says_so_in_the_payload(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """A consumer must be able to tell that nothing could be measured.

        Without this flag a console would render an all-stable signal table and
        show green, which is the most misleading thing this system could display.
        """
        write_drift_artifacts(
            self._report(labels=False), artifact_root=artefacts, dataset_run_id="run-1"
        )

        body = client.get("/v1/monitoring/drift").json()

        assert body["labels_available"] is False
        assert body["current"]["n_matured"] == 0
        assert body["performance"] == []

    def test_signals_and_performance_are_separate_fields(
        self, client: TestClient, artefacts: Path
    ) -> None:
        """Drift is not failure, and the payload keeps the two structurally apart."""
        write_drift_artifacts(
            self._report(labels=True), artifact_root=artefacts, dataset_run_id="run-1"
        )

        body = client.get("/v1/monitoring/drift").json()

        assert "signals" in body
        assert "performance" in body
        # A drift signal has nowhere to record a verdict about model quality.
        assert set(body["signals"][0]) == {
            "name",
            "kind",
            "statistic",
            "distance",
            "severity",
            "baseline_value",
            "current_value",
            "baseline_n",
            "current_n",
            "sufficient",
            "note",
        }

    def test_the_dataset_run_annotation_is_stripped_from_the_response(
        self, client: TestClient, artefacts: Path
    ) -> None:
        paths = write_drift_artifacts(
            self._report(labels=True), artifact_root=artefacts, dataset_run_id="run-1"
        )
        raw = json.loads(paths[0].read_text(encoding="utf-8"))

        assert raw["dataset_run_id"] == "run-1"
        body = client.get("/v1/monitoring/drift").json()
        assert "dataset_run_id" not in body
