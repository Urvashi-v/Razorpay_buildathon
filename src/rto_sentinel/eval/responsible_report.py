"""Artefacts and the responsible-AI report.

Every number in ``docs/responsible_ai.md`` is read from a saved artefact and
rendered by this module. There is no code path here that writes a rupee figure,
a rate or a delta from a literal, which is what stops the document drifting away
from the runs that produced it - the same discipline
:mod:`rto_sentinel.eval.economics_report` and
:mod:`rto_sentinel.eval.model_card` apply.

A consequence worth stating: if an experiment has not been run, the report says
so in the place the numbers would have been. It does not fall back to prose that
implies a result.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rto_sentinel.contracts.monitoring import DriftReport, ShiftStudy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.evaluation import CohortResult, FairnessAudit

#: Where Phase 10 artefacts live, under the artefact root.
RESPONSIBLE_DIR = "responsible"

#: The disclaimer that appears on every artefact and at the top of the report.
#:
#: Repeated rather than referenced because these artefacts get read in
#: isolation - a CSV opened in a spreadsheet carries no link back to a README,
#: and a fairness table without this line is exactly the kind of thing that ends
#: up in a slide claiming the model was audited for production fairness.
SYNTHETIC_DISCLAIMER = (
    "Controlled benchmark experiment on synthetic data. Cohorts, shifts and drift "
    "windows are properties of the documented simulator in docs/simulator.md, not "
    "observations of real customers, real distribution shift, or real production "
    "behaviour. These results are evidence that the audit machinery works and that "
    "this model behaves a certain way on this benchmark. They are NOT evidence of "
    "production fairness or production robustness, and no such claim should be made "
    "from them."
)


def _directory(artifact_root: Path) -> Path:
    directory = artifact_root / RESPONSIBLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_fairness_artifacts(
    audit: FairnessAudit,
    slices: tuple[CohortResult, ...],
    *,
    artifact_root: Path,
    dataset_run_id: str,
    split: str,
    model_version: str,
    threshold: float,
) -> tuple[Path, ...]:
    """Write the audit as JSON and the cohort table as CSV."""
    directory = _directory(artifact_root)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_run_id": dataset_run_id,
        "split": split,
        "model_version": model_version,
        "threshold": threshold,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "audit": audit.model_dump(mode="json"),
        "slices": [entry.model_dump(mode="json") for entry in slices],
    }
    json_path = directory / f"fairness__{split}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = directory / f"fairness__{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cohort",
                "group",
                "n_orders",
                "n_positives",
                "n_flagged",
                "rto_rate",
                "rto_rate_ci_low",
                "rto_rate_ci_high",
                "flag_rate",
                "flag_rate_ci_low",
                "flag_rate_ci_high",
                "precision",
                "precision_ci_low",
                "precision_ci_high",
                "recall",
                "net_inr_per_1000",
                "sufficient",
                "insufficient_reason",
            ]
        )
        for entry in slices:
            writer.writerow(
                [
                    entry.cohort,
                    entry.group,
                    entry.n_orders,
                    entry.n_positives,
                    entry.n_flagged,
                    _round(entry.rto_rate),
                    _round(entry.rto_rate_ci[0] if entry.rto_rate_ci else None),
                    _round(entry.rto_rate_ci[1] if entry.rto_rate_ci else None),
                    _round(entry.flag_rate),
                    _round(entry.flag_rate_ci[0] if entry.flag_rate_ci else None),
                    _round(entry.flag_rate_ci[1] if entry.flag_rate_ci else None),
                    _round(entry.precision),
                    _round(entry.precision_ci[0] if entry.precision_ci else None),
                    _round(entry.precision_ci[1] if entry.precision_ci else None),
                    _round(entry.recall),
                    _round(entry.net_inr_per_1000, 2),
                    entry.sufficient,
                    entry.insufficient_reason,
                ]
            )
    return (json_path, csv_path)


def write_shift_artifacts(study: ShiftStudy, *, artifact_root: Path) -> tuple[Path, ...]:
    """Write the shift study as JSON and its result table as CSV."""
    directory = _directory(artifact_root)
    json_path = directory / "shift_study.json"
    payload = study.model_dump(mode="json")
    payload["disclaimer"] = SYNTHETIC_DISCLAIMER
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = directory / "shift_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "environment",
                "description",
                "n_orders",
                "observed_rto_rate",
                "pr_auc",
                "pr_auc_delta",
                "roc_auc",
                "brier_score",
                "expected_calibration_error",
                "ece_delta",
                "flag_rate",
                "precision",
                "recall",
                "net_inr_per_1000",
                "net_delta",
            ]
        )
        for result in study.results:
            writer.writerow(
                [
                    result.environment,
                    result.description,
                    result.n_orders,
                    _round(result.observed_rto_rate),
                    _round(result.pr_auc),
                    _round(result.pr_auc_delta),
                    _round(result.roc_auc),
                    _round(result.brier_score),
                    _round(result.expected_calibration_error),
                    _round(result.ece_delta),
                    _round(result.flag_rate),
                    _round(result.precision),
                    _round(result.recall),
                    _round(result.net_inr_per_1000, 2),
                    _round(result.net_delta, 2),
                ]
            )
    return (json_path, csv_path)


def write_drift_artifacts(
    report: DriftReport, *, artifact_root: Path, dataset_run_id: str
) -> tuple[Path, ...]:
    """Write the drift report as JSON, for the API to serve unchanged."""
    directory = _directory(artifact_root)
    path = directory / "drift_report.json"
    payload = report.model_dump(mode="json")
    payload["dataset_run_id"] = dataset_run_id
    payload["disclaimer"] = SYNTHETIC_DISCLAIMER
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return (path,)


def _round(value: float | None, digits: int = 4) -> str | float:
    """Blank for a missing value, never a zero.

    A CSV cell containing 0 for "not computed" is the spreadsheet equivalent of a
    fabricated measurement: it sorts, it averages, and nothing about it says it
    was never measured.
    """
    if value is None:
        return ""
    return round(value, digits)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def load_artifacts(artifact_root: Path) -> dict[str, object]:
    """Read whatever Phase 10 artefacts exist. Missing ones stay missing."""
    directory = artifact_root / RESPONSIBLE_DIR
    found: dict[str, object] = {}
    if not directory.is_dir():
        return found

    for split in ("validation", "test"):
        path = directory / f"fairness__{split}.json"
        if path.is_file():
            found[f"fairness_{split}"] = json.loads(path.read_text(encoding="utf-8"))

    shift_path = directory / "shift_study.json"
    if shift_path.is_file():
        found["shift"] = ShiftStudy.model_validate_json(shift_path.read_text(encoding="utf-8"))

    drift_path = directory / "drift_report.json"
    if drift_path.is_file():
        payload = json.loads(drift_path.read_text(encoding="utf-8"))
        payload.pop("dataset_run_id", None)
        payload.pop("disclaimer", None)
        found["drift"] = DriftReport.model_validate(payload)

    return found


def render_responsible_report(*, artifact_root: Path, output: Path | None = None) -> Path:
    """Render ``docs/responsible_ai.md`` from the saved artefacts.

    Raises ``FileNotFoundError`` when nothing has been run. A responsible-AI
    report generated from no experiments is worse than no report: it looks like
    diligence and contains none.
    """
    found = load_artifacts(artifact_root)
    if not found:
        msg = (
            f"no responsible-AI artefacts under {artifact_root / RESPONSIBLE_DIR}. Run "
            "`rto-sentinel fairness`, `rto-sentinel shift` and `rto-sentinel monitor` "
            "first. A report rendered from nothing would claim diligence that did not "
            "happen."
        )
        raise FileNotFoundError(msg)

    lines: list[str] = []
    lines.append("# Responsible AI and robustness report")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`rto-sentinel responsible-report`. Every number below is read from a saved "
        "artefact under `artifacts/responsible/`; none is written by hand."
    )
    lines.append("")
    lines.append("> **Read this first.**")
    for sentence in SYNTHETIC_DISCLAIMER.split(". "):
        if sentence.strip():
            lines.append(f"> {sentence.strip().rstrip('.')}.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(_fairness_section(found))
    lines.extend(_shift_section(found))
    lines.extend(_drift_section(found))
    lines.extend(_limitations_section(found))

    destination = output or Path("docs/responsible_ai.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _fairness_section(found: dict[str, object]) -> list[str]:
    lines = ["## 1. Fairness across operational cohorts", ""]

    lines.append("### What was and was not examined")
    lines.append("")
    lines.append(
        "The cohorts are operational: delivery-area tier, order-value band, "
        "customer-history depth, and payment method. Every one is a fact recorded on "
        "the order."
    )
    lines.append("")
    lines.append(
        "**No sensitive characteristic was examined, inferred, or approximated.** There "
        "is no gender, religion, caste, ethnicity, age or income field in this data - "
        "not withheld, not present - and none is derived from names, addresses or any "
        "other field. `eval/fairness.py` refuses to group by a column whose name matches "
        "a sensitive token, and that refusal is a hard error rather than a skipped "
        "cohort, so a misconfigured audit fails loudly instead of quietly examining less "
        "than it claims."
    )
    lines.append("")
    lines.append(
        "Pincode tier is the closest thing here to a proxy for something sensitive, and "
        "that is exactly why it is the headline cohort rather than an omitted one. A "
        "delivery-area tier is an operational fact about logistics and it is also "
        "correlated with income. Auditing it openly is the alternative to pretending the "
        "correlation is not there."
    )
    lines.append("")

    any_found = False
    for split in ("validation", "test"):
        payload = found.get(f"fairness_{split}")
        if not isinstance(payload, dict):
            continue
        any_found = True
        audit = payload["audit"]
        slices = payload["slices"]
        lines.append(f"### Results on the {split} split")
        lines.append("")
        lines.append(
            f"Model `{payload['model_version']}` at threshold "
            f"{float(payload['threshold']):.4f}, minimum support "
            f"{audit['min_support']} orders."
        )
        lines.append("")
        lines.append(
            "| Cohort | Group | n | RTO rate | Flag rate | Precision | Recall "
            "| Net INR/1k | Evidence |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for entry in slices:
            lines.append(
                f"| {entry['cohort']} | {entry['group']} | {entry['n_orders']:,} | "
                f"{_pct(entry['rto_rate'])} {_ci(entry.get('rto_rate_ci'))} | "
                f"{_pct(entry['flag_rate'])} {_ci(entry.get('flag_rate_ci'))} | "
                f"{_num(entry['precision'])} {_ci(entry.get('precision_ci'))} | "
                f"{_num(entry['recall'])} | {_inr(entry['net_inr_per_1000'])} | "
                f"{'yes' if entry['sufficient'] else 'too thin'} |"
            )
        lines.append("")
        lines.append(
            f"**Disparity review: {'TRIGGERED' if audit['triggered'] else 'not triggered'}.** "
            f"Maximum flag-rate ratio {audit['max_flag_rate_ratio']:.2f} "
            f"({audit['most_flagged_group']} vs {audit['least_flagged_group']}); "
            f"worst precision drop {audit['worst_precision_drop']:.3f}."
        )
        lines.append("")
        lines.append(str(audit["narrative"]))
        lines.append("")
        if audit["groups_below_support"]:
            lines.append(
                "Groups shown but excluded from the comparison for insufficient support: "
                + ", ".join(audit["groups_below_support"])
                + "."
            )
            lines.append("")

    if not any_found:
        lines.append(
            "**Not run.** No fairness artefact exists. No fairness claim may be made "
            "about this model."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _shift_section(found: dict[str, object]) -> list[str]:
    lines = ["## 2. Distribution shift", ""]
    study = found.get("shift")
    if not isinstance(study, ShiftStudy):
        lines.append("**Not run.** No shift study artefact exists.")
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    lines.append(
        "Each environment is a **named change to a generator parameter**, not a fresh "
        "draw from the same distribution. Regenerating with a new seed and calling that "
        "robustness would measure sampling variance: every draw would come from the "
        "distribution the model was trained on."
    )
    lines.append("")
    lines.append(
        f"The model (`{study.model_version}`) is **not retrained** between environments, "
        f"and the threshold is held fixed at {study.threshold:.4f}. Re-deriving the "
        "threshold per environment would repair part of the damage and understate what a "
        "deployed model suffers - in production the threshold is a configuration value "
        "and does not follow the distribution around."
    )
    lines.append("")

    lines.append("### Environments")
    lines.append("")
    lines.append("| Environment | What changed | Overrides |")
    lines.append("|---|---|---|")
    for spec in study.environments:
        overrides = (
            ", ".join(f"`{key}={value}`" for key, value in sorted(spec.overrides.items()))
            or "_none (control)_"
        )
        lines.append(f"| `{spec.name}` | {spec.description} | {overrides} |")
    lines.append("")

    lines.append("### Measured degradation")
    lines.append("")
    lines.append(
        "| Environment | n | RTO rate | PR-AUC | ΔPR-AUC | ECE | ΔECE | Flag rate | "
        "Precision | Net ₹/1k | ΔNet |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in study.results:
        lines.append(
            f"| `{result.environment}` | {result.n_orders:,} | "
            f"{_pct(result.observed_rto_rate)} | {result.pr_auc:.3f} | "
            f"{_delta(result.pr_auc_delta)} | {result.expected_calibration_error:.3f} | "
            f"{_delta(result.ece_delta)} | {_pct(result.flag_rate)} | "
            f"{_num(result.precision)} | {_inr(result.net_inr_per_1000)} | "
            f"{_delta(result.net_delta, money=True)} |"
        )
    lines.append("")

    lines.append("### Findings")
    lines.append("")
    for finding in study.findings:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _drift_section(found: dict[str, object]) -> list[str]:
    lines = ["## 3. Monitoring and drift", ""]
    report = found.get("drift")
    if not isinstance(report, DriftReport):
        lines.append("**Not run.** No drift report artefact exists.")
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    lines.append(
        "**Drift is not failure.** A moved input distribution is a fact about the world; "
        "whether quality degraded is a separate question that needs labels. The two are "
        "kept structurally apart: a drift signal has no field in which to record a "
        "verdict, and a performance delta cannot be constructed without mature outcomes."
    )
    lines.append("")
    lines.append(
        "This matters operationally. Indian e-commerce moves hard during festive season - "
        "COD share rises, order values rise, category mix swings. Every one of those "
        "shows up as real drift and none of them means the model stopped working. A "
        "monitor that pages someone every Diwali gets muted by March."
    )
    lines.append("")

    lines.append(
        f"Baseline: {report.baseline.n_orders:,} orders "
        f"({report.baseline.n_matured:,} matured). "
        f"Current: {report.current.n_orders:,} orders "
        f"({report.current.n_matured:,} matured). "
        f"Labelled comparison possible: **{'yes' if report.labels_available else 'no'}**."
    )
    lines.append("")

    lines.append("| Kind | Quantity | Statistic | Baseline | Current | Distance | Reading |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for signal in report.signals:
        lines.append(
            f"| {signal.kind} | `{signal.name}` | {signal.statistic} | "
            f"{_num(signal.baseline_value)} | {_num(signal.current_value)} | "
            f"{signal.distance:.4f} | "
            f"{signal.severity}{'' if signal.sufficient else ' (too thin to read)'} |"
        )
    lines.append("")

    if report.performance:
        lines.append("### Labelled comparisons")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Δ | Evidence |")
        lines.append("|---|---:|---:|---:|---|")
        for delta in report.performance:
            lines.append(
                f"| {delta.metric} | {delta.baseline:.4f} | {delta.current:.4f} | "
                f"{delta.delta:+.4f} | "
                f"{'yes' if delta.sufficient else 'insufficient matured rows'} |"
            )
        lines.append("")

    lines.append("### Warnings")
    lines.append("")
    for warning in report.warnings:
        lines.append(f"- {warning}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _limitations_section(found: dict[str, object]) -> list[str]:
    lines = ["## 4. Limitations", ""]
    lines.append(
        "**These are controlled benchmark experiments, not evidence of production "
        "fairness or production robustness.** Stated plainly because this is the "
        "sentence most likely to be dropped when results are quoted onward."
    )
    lines.append("")
    lines.append(
        "1. **The labels are simulated.** Every outcome here was drawn from the causal "
        "process in `docs/simulator.md`. A cohort disparity measured on this data is a "
        "property of that process. If the simulator makes tier-3 riskier - and it does, "
        "by an explicit `tier_risk_offset` - then a model that flags tier-3 more is "
        "recovering a fact the simulator put there, not discovering one about India."
    )
    lines.append("")
    lines.append(
        "2. **The cohorts are operational, not demographic.** This audit cannot answer "
        "whether the system disadvantages any protected group, because no such attribute "
        "exists in the data and inferring one would be worse than not asking. A "
        "production deployment would need a separate, consented, legally-reviewed process "
        "for that question."
    )
    lines.append("")
    lines.append(
        "3. **The shift environments are the ones we thought of.** Robustness against "
        "nine named perturbations is not robustness in general. The failure modes that "
        "matter most in production are usually the ones nobody enumerated - a courier "
        "changing its scanning behaviour, a checkout redesign changing session features, "
        "an upstream field going null."
    )
    lines.append("")
    lines.append(
        "4. **Drift bands are conventions, not calibrated thresholds.** The PSI bands "
        "(0.10 / 0.25) come from credit-risk practice. They were not tuned against a "
        "labelled history of this system's incidents, because no such history exists. "
        'They select the words "watch" and "investigate" rather than "warn" and '
        '"fail" for exactly that reason.'
    )
    lines.append("")
    lines.append(
        "5. **The audit ran on one dataset run and one model version.** Both are recorded "
        "in the artefacts. Neither result transfers automatically to a retrained model, "
        "and re-running the audit is part of shipping one."
    )
    lines.append("")

    fairness = found.get("fairness_validation") or found.get("fairness_test")
    if isinstance(fairness, dict):
        thin = fairness["audit"]["groups_below_support"]
        if thin:
            lines.append(
                f"6. **{len(thin)} cohort group(s) were too small to support a "
                "conclusion.** They are shown in the table because suppressing them would "
                "hide exactly what an audit exists to look at, but they are excluded from "
                "the disparity comparison in both directions - they cannot fire the "
                "trigger and they cannot hold down a ratio that would otherwise have "
                "fired it."
            )
            lines.append("")

    return lines


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _inr(value: float | None) -> str:
    return "—" if value is None else f"₹{value:,.0f}"


def _ci(bounds: object) -> str:
    """A Wilson interval, rendered small.

    Printed next to every rate so a reader sees the precision of the estimate at
    the same moment they see the estimate. A cohort table without intervals
    invites comparing 0.44 against 0.47 as though the difference were real.
    """
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return ""
    low, high = bounds
    if low is None or high is None:
        return ""
    return f"<sub>[{float(low):.2f}, {float(high):.2f}]</sub>"


def _delta(value: float | None, *, money: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+,.0f}" if money else f"{value:+.3f}"
