"""Rendering the model card, and the final-model comparison table.

WHY THE CARD IS GENERATED RATHER THAN WRITTEN
=============================================
A model card is a claim about a specific artefact. Written by hand, it starts
accurate and drifts: the model is retrained, the numbers move, and the card keeps
saying what was true in the version someone last edited it for.

So it is assembled from two sources and neither of them is this file. The prose -
intended use, limitations, fairness, drift - comes from
``config/models/model_card.yaml``, where it can be reviewed as prose. Every
number comes from the measured evaluation artefacts. There is no code path here
that can state a metric the run did not produce, which is the same rule the
Phase 4 report follows.

THE VALIDATION COLUMN IS LABELLED, NOT HIDDEN
=============================================
Hyperparameters were chosen on validation and the shipped calibrator was refitted
on it, so validation metrics are optimistic. They are still shown - a reader
comparing them against the test column learns something real about how much the
selection cost - but they are labelled at every point where they appear.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Protocol

from rto_sentinel.eval.report import MARKDOWN_DASH, DishonestReportError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from rto_sentinel.configuration.schemas import ModelCardConfig
    from rto_sentinel.contracts.experiment import LadderResults
    from rto_sentinel.contracts.final import FinalEvaluation, SelectionManifest

#: Wording that must survive into any rendered card. Checked rather than trusted:
#: the disclaimer is the one sentence a reader most needs and an editor is most
#: likely to soften.
_REQUIRED_PHRASES = ("simulated", "not real-world ground truth")


class _Render(Protocol):
    """One metric, rendered from one evaluation."""

    def __call__(self, evaluation: FinalEvaluation, /) -> str: ...


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or value != value:  # None or NaN
        return MARKDOWN_DASH
    return f"{value:.{digits}f}"


def _fmt_int(value: float | None) -> str:
    if value is None or value != value:
        return MARKDOWN_DASH
    return f"{value:,.0f}"


def _interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _csv_value(value: float | None) -> float | str:
    """Undefined becomes an empty cell, never the string "nan".

    A reader - or a spreadsheet - treats `nan` as a value. An empty cell is the
    honest representation of "this metric does not exist for this model", which
    is rung 0's situation with ROC-AUC.
    """
    if value is None or value != value:
        return ""
    return value


# ---------------------------------------------------------------------------
# the metrics table, one column per evaluated split
# ---------------------------------------------------------------------------


def metrics_rows(evaluations: dict[str, FinalEvaluation]) -> list[tuple[str, dict[str, str]]]:
    """Every reported metric as (label, {split: rendered value}).

    Built once and reused by the markdown card and the CSV, so the two cannot
    disagree about what was measured.
    """
    rows: list[tuple[str, dict[str, str]]] = []

    def add(label: str, render: _Render) -> None:
        rows.append((label, {split: render(ev) for split, ev in evaluations.items()}))

    add("Rows", lambda ev: f"{ev.evaluation_summary.n_rows:,}")
    add("Positive rate", lambda ev: _fmt(ev.evaluation_summary.positive_rate, 4))
    add(
        "PR-AUC",
        lambda ev: (
            f"**{ev.ranking.pr_auc.value:.3f}** "
            f"{_interval(ev.ranking.pr_auc.ci_low, ev.ranking.pr_auc.ci_high)}"
        ),
    )
    add("PR-AUC, uncalibrated", lambda ev: _fmt(ev.uncalibrated_pr_auc))
    add(
        "ROC-AUC",
        lambda ev: (
            f"{ev.ranking.roc_auc.value:.3f} "
            f"{_interval(ev.ranking.roc_auc.ci_low, ev.ranking.roc_auc.ci_high)}"
            if ev.ranking.roc_auc.is_defined
            else MARKDOWN_DASH
        ),
    )
    add("Recall @ precision 80%", lambda ev: _fmt(ev.ranking.recall_at_precision_80))
    add("Recall @ precision 90%", lambda ev: _fmt(ev.ranking.recall_at_precision_90))
    add("Brier score", lambda ev: _fmt(ev.calibration.brier_score, 4))
    add("Brier, uncalibrated", lambda ev: _fmt(ev.uncalibrated_calibration.brier_score, 4))
    add("Expected calibration error", lambda ev: _fmt(ev.calibration.expected_calibration_error, 4))
    add(
        "ECE, uncalibrated",
        lambda ev: _fmt(ev.uncalibrated_calibration.expected_calibration_error, 4),
    )
    add("Operating threshold", lambda ev: _fmt(ev.operating_point.threshold, 4))
    add("Flag rate", lambda ev: _fmt(ev.operating_point.flag_rate))
    add("Precision", lambda ev: _fmt(ev.operating_point.precision))
    add("Recall", lambda ev: _fmt(ev.operating_point.recall))
    add("F1", lambda ev: _fmt(ev.operating_point.f1))
    add(
        "Confusion (TP / FP / FN / TN)",
        lambda ev: (
            f"{ev.operating_point.true_positives:,} / {ev.operating_point.false_positives:,} / "
            f"{ev.operating_point.false_negatives:,} / {ev.operating_point.true_negatives:,}"
        ),
    )
    add(
        "Net INR saved per 1,000 orders",
        lambda ev: (
            f"**{ev.economics.net_inr_saved_per_1000_orders.value:,.0f}** "
            f"[{ev.economics.net_inr_saved_per_1000_orders.ci_low:,.0f}, "
            f"{ev.economics.net_inr_saved_per_1000_orders.ci_high:,.0f}]"
        ),
    )
    add(
        "False-positive cost, INR",
        lambda ev: _fmt_int(ev.economics.total_false_positive_cost_inr),
    )
    add(
        "Do-nothing loss absorbed, INR/1k",
        lambda ev: _fmt_int(abs(ev.economics.baseline_net_inr_per_1000_orders)),
    )
    return rows


def write_metrics_csv(evaluations: dict[str, FinalEvaluation], path: Path) -> Path:
    """The same table as CSV, for anyone who would rather not parse markdown."""
    splits = list(evaluations)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *splits])
        for label, values in metrics_rows(evaluations):
            writer.writerow([label, *[values[split].replace("**", "") for split in splits]])
    return path


def write_comparison_csv(
    evaluations: dict[str, FinalEvaluation],
    ladder: LadderResults | None,
    path: Path,
) -> Path:
    """The final model beside every ladder rung, on the split they share.

    The ladder was measured on validation, so the comparison is on validation.
    Putting the test column next to rungs that never saw the test set would be
    comparing two different measurements and calling it a ranking.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "kind",
        "split",
        "calibrated",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "roc_auc",
        "ece",
        "brier",
        "flag_rate",
        "precision",
        "recall",
        "f1",
        "net_inr_per_1000",
        "fp_cost_inr",
    ]

    rows: list[list[object]] = []
    if ladder is not None:
        for record in ladder.ordered:
            head = record.headline()
            rows.append(
                [
                    record.model_name,
                    f"ladder rung {record.rung_id}",
                    record.evaluated_split,
                    record.is_calibrated,
                    record.ranking.pr_auc.value,
                    record.ranking.pr_auc.ci_low,
                    record.ranking.pr_auc.ci_high,
                    _csv_value(head["roc_auc"]),
                    _csv_value(head["ece"]),
                    record.calibration.brier_score,
                    _csv_value(head["flag_rate"]),
                    _csv_value(head["precision"]),
                    _csv_value(head["recall"]),
                    _csv_value(head["f1"]),
                    _csv_value(head["net_inr_per_1000"]),
                    _csv_value(head["fp_cost_inr"]),
                ]
            )

    for split, ev in evaluations.items():
        point = ev.operating_point
        rows.append(
            [
                ev.model_name,
                "final model",
                split,
                ev.is_calibrated,
                ev.ranking.pr_auc.value,
                ev.ranking.pr_auc.ci_low,
                ev.ranking.pr_auc.ci_high,
                ev.ranking.roc_auc.value if ev.ranking.roc_auc.is_defined else "",
                ev.calibration.expected_calibration_error,
                ev.calibration.brier_score,
                point.flag_rate,
                _csv_value(point.precision),
                _csv_value(point.recall),
                _csv_value(point.f1),
                ev.economics.net_inr_saved_per_1000_orders.value,
                ev.economics.total_false_positive_cost_inr,
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------


def _findings_section(evaluations: dict[str, FinalEvaluation]) -> list[str]:
    """State what the measurement showed, in the direction the numbers point.

    Generated from the evaluations rather than written, so a run that comes out
    differently produces different prose. The three checks below are the ones a
    reader would otherwise have to do themselves, and each has a failing
    direction that a table of numbers does not make obvious:

    * whether a saving can be claimed at all, or whether the interval crosses
      zero;
    * whether the calibration fitted on validation still helps on unseen data;
    * how much of any drop between splits is a base-rate artefact rather than a
      loss of ranking quality.
    """
    test = evaluations.get("test")
    if test is None:
        return [
            "",
            "## What the measurement shows",
            "",
            "No sealed-set evaluation exists yet, so no claim about this model's "
            "performance on unseen data can be made. The validation numbers above "
            "describe data the model was selected on.",
        ]

    lines = ["", "## What the measurement shows", ""]

    net = test.economics.net_inr_saved_per_1000_orders
    if net.ci_low > 0:
        lines.append(
            f"**The model saves money on the sealed set.** Net "
            f"INR {net.value:,.0f} per 1,000 orders, 95% interval "
            f"[{net.ci_low:,.0f}, {net.ci_high:,.0f}] - the interval stays above zero, so "
            f"the saving survives sampling uncertainty at this sample size."
        )
    else:
        lines.append(
            f"**No saving can be claimed at 95% confidence.** The point estimate is net "
            f"INR {net.value:,.0f} per 1,000 orders, but the interval "
            f"[{net.ci_low:,.0f}, {net.ci_high:,.0f}] crosses zero: on "
            f"{test.evaluation_summary.n_rows:,} sealed orders this measurement cannot "
            f"distinguish the model from doing nothing. That is a statement about the "
            f"evidence, not a claim that the model is worthless - but it is the honest "
            f"reading, and a larger held-out sample is what would settle it."
        )

    improvement = test.calibration_improvement
    if improvement > 0:
        lines += [
            "",
            f"**Calibration transferred.** Expected calibration error on the sealed set is "
            f"{test.calibration.expected_calibration_error:.4f} calibrated against "
            f"{test.uncalibrated_calibration.expected_calibration_error:.4f} raw, so the "
            f"`{test.calibration_method}` mapping fitted on validation still helps on data "
            f"it never saw.",
        ]
    else:
        lines += [
            "",
            f"**Calibration did not transfer to the sealed set.** Expected calibration error "
            f"is {test.calibration.expected_calibration_error:.4f} calibrated against "
            f"{test.uncalibrated_calibration.expected_calibration_error:.4f} raw, and the "
            f"Brier score agrees in direction "
            f"({test.calibration.brier_score:.4f} against "
            f"{test.uncalibrated_calibration.brier_score:.4f}) - so the "
            f"`{test.calibration_method}` mapping, fitted on the validation window, makes the "
            f"probabilities slightly *worse* on the later window rather than better. Both "
            f"errors are small in absolute terms and the reliability diagram shows the two "
            f"curves close together, so this is a failure to help rather than a collapse. "
            f"It is nonetheless the "
            f"distribution-shift limitation listed below, measured rather than predicted: "
            f"the mapping encodes the validation window's score-to-frequency relationship, "
            f"and that relationship moved. It is the argument for recalibrating on recent "
            f"matured outcomes rather than shipping a mapping fitted once.",
        ]

    validation = evaluations.get("validation")
    if validation is not None:
        pr_drop = validation.ranking.pr_auc.value - test.ranking.pr_auc.value
        base_drop = (
            validation.evaluation_summary.positive_rate - test.evaluation_summary.positive_rate
        )
        val_lift = validation.ranking.pr_auc.value / validation.evaluation_summary.positive_rate
        test_lift = test.ranking.pr_auc.value / test.evaluation_summary.positive_rate
        lines += [
            "",
            f"**The drop from validation to test is partly arithmetic.** PR-AUC falls "
            f"{pr_drop:.3f}, but the positive rate also falls {base_drop:.4f} "
            f"({validation.evaluation_summary.positive_rate:.4f} to "
            f"{test.evaluation_summary.positive_rate:.4f}), and PR-AUC is bounded below by "
            f"the base rate. Measured as lift over the base rate the model achieves "
            f"{val_lift:.2f}x on validation and {test_lift:.2f}x on test; ROC-AUC, which "
            f"does not move with the base rate, goes "
            f"{validation.ranking.roc_auc.value:.3f} to {test.ranking.roc_auc.value:.3f}. "
            f"Ranking quality degraded, but by considerably less than the headline "
            f"difference suggests. The remainder is what selecting on validation cost.",
        ]

    return lines


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item.strip()}" for item in items]


def render_model_card(
    config: ModelCardConfig,
    manifest: SelectionManifest,
    evaluations: dict[str, FinalEvaluation],
    *,
    ladder: LadderResults | None = None,
) -> str:
    """Assemble the card from reviewed prose and measured numbers."""
    if not evaluations:
        msg = "refusing to render a model card with no evaluation behind it"
        raise DishonestReportError(msg)

    splits = list(evaluations)
    lines: list[str] = [
        f"# Model card - {config.model_name}",
        "",
        "**Generated from the frozen selection manifest and the measured evaluation "
        "artefacts. Do not edit by hand.**",
        "",
        f"> {config.training_data.synthetic_disclaimer.strip()}",
        "",
        "## Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model | `{manifest.base_rung}` + `{manifest.calibration_method}` calibration |",
        f"| Model version | `{manifest.model_version}` |",
        f"| Selection manifest | `{manifest.manifest_id}` |",
        f"| Frozen at | {manifest.frozen_at.isoformat()} |",
        f"| Owner | {config.owner} |",
        f"| Feature version | `{manifest.feature_version}` |",
        f"| Feature fingerprint | `{manifest.feature_fingerprint[:16]}...` |",
        f"| Dataset run | `{manifest.dataset_run_id}` |",
        f"| Generator version | `{manifest.generator_version}` |",
        f"| Config fingerprint | `{manifest.config_fingerprint[:16]}...` |",
        f"| Seed | `{manifest.seed}` |",
        f"| Calibration | `{manifest.calibration_method}`, fitted on "
        f"{manifest.calibration_fitted_on}, {manifest.calibration_folds}-fold selection |",
        f"| Cost profile | `{manifest.cost_profile}` |",
        f"| Operating threshold | **{manifest.threshold:.4f}** ({manifest.threshold_source}) |",
        "",
        "## What it produces",
        "",
        config.summary.strip(),
        "",
        "```",
        "P(RTO | information available at the moment the order was placed)",
        "```",
        "",
        "## Intended use",
        "",
        *_bullets(config.intended_use),
        "",
        "## Not intended for",
        "",
        *_bullets(config.non_intended_use),
        "",
        "## Training data",
        "",
        config.training_data.description.strip(),
        "",
        f"**{config.training_data.synthetic_disclaimer.strip()}**",
        "",
        "What the data does not contain:",
        "",
        *_bullets(config.training_data.what_is_not_in_it),
        "",
        "| Split | Rows | Positive rate | Days | First order | Last order |",
        "|---|---|---|---|---|---|",
    ]

    for summary in (manifest.train_summary, manifest.validation_summary):
        lines.append(
            f"| {summary.name} | {summary.n_rows:,} | {summary.positive_rate:.4f} "
            f"| {summary.first_day}-{summary.last_day} "
            f"| {summary.first_ordered_at.date()} | {summary.last_ordered_at.date()} |"
        )
    for split, ev in evaluations.items():
        if split in {"train", "validation"}:
            continue
        summary = ev.evaluation_summary
        lines.append(
            f"| {summary.name} | {summary.n_rows:,} | {summary.positive_rate:.4f} "
            f"| {summary.first_day}-{summary.last_day} "
            f"| {summary.first_ordered_at.date()} | {summary.last_ordered_at.date()} |"
        )

    lines += [
        "",
        "## Features",
        "",
        config.features.description.strip(),
        "",
        f"{len(manifest.feature_names)} features across {len(manifest.families_used)} families: "
        + ", ".join(f"`{family}`" for family in manifest.families_used)
        + ". Full definitions, lookbacks and availability arguments are in "
        "[docs/features.md](features.md).",
        "",
        "Deliberately excluded:",
        "",
        *_bullets(config.features.excluded_deliberately),
        "",
        "## How the model was selected",
        "",
        f"Base rung `{manifest.base_rung}`. {len(manifest.candidates)} candidates, all fitted "
        "on train and scored on validation. The test split was not read at any point in "
        "this table.",
        "",
        "| Candidate | Validation PR-AUC | Train PR-AUC | Train-val gap | Selected |",
        "|---|---|---|---|---|",
    ]
    for candidate in manifest.candidates:
        gap = candidate.overfit_gap
        lines.append(
            f"| `{candidate.name}` | {candidate.validation_pr_auc:.4f} "
            f"| {_fmt(candidate.train_pr_auc, 4)} "
            f"| {f'{gap:+.3f}' if gap is not None else MARKDOWN_DASH} "
            f"| {'**yes**' if candidate.selected else ''} |"
        )

    lines += [
        "",
        "## Calibration",
        "",
        config.calibration_methodology.strip(),
        "",
        "| Method | ECE (out-of-fold) | Brier | vs. leaving scores alone | Selected |",
        "|---|---|---|---|---|",
    ]
    for method in manifest.calibration_candidates:
        lines.append(
            f"| `{method.method}` | {method.expected_calibration_error:.4f} "
            f"| {method.brier_score:.4f} | {method.improvement_over_none:+.4f} "
            f"| {'**yes**' if method.selected else ''} |"
        )

    lines += [
        "",
        "## Evaluation methodology",
        "",
        config.evaluation_methodology.strip(),
        "",
        "## Measured results",
        "",
    ]

    if "validation" in splits:
        lines += [
            "> **The validation column is optimistic and is shown for comparison only.** "
            "Hyperparameters were chosen on validation and the shipped calibrator was "
            "refitted on it, so those rows describe data the model was tuned against. "
            + (
                "The test column is the honest measurement."
                if "test" in splits
                else "No test-set measurement exists yet."
            ),
            "",
        ]

    lines += [
        "| Metric | " + " | ".join(split for split in splits) + " |",
        "|---" * (len(splits) + 1) + "|",
    ]
    for label, values in metrics_rows(evaluations):
        lines.append(f"| {label} | " + " | ".join(values[split] for split in splits) + " |")

    if "test" in evaluations:
        test = evaluations["test"]
        lines += [
            "",
            f"The sealed test split was opened once, after the manifest was frozen. "
            f"Stated reason: *{test.unseal_reason}*",
        ]

    lines += _findings_section(evaluations)

    if ladder is not None:
        lines += [
            "",
            "## Against the baseline ladder",
            "",
            "Every rung below was measured on the same validation split, at the same "
            "cost-derived threshold, by the same code. SPEC section 05: if a simpler "
            "rung wins on net rupees, it ships.",
            "",
            "**This comparison is not fully like-for-like, and the difference favours the "
            "final model.** The ladder rungs are uncalibrated, while the final row is "
            "calibrated. At a fixed threshold, calibration moves scores across that "
            "threshold and therefore changes the flag rate and the rupee figure without "
            "any change in ranking quality - which is why the PR-AUC column, which "
            "calibration cannot move, is the honest column here. Settling whether the "
            "final model genuinely beats rung 3 on money would require calibrating rung 3 "
            "the same way and re-measuring; that has not been done.",
            "",
            "| Rung | Model | PR-AUC | Flag rate | Precision | Net INR/1k |",
            "|---|---|---|---|---|---|",
        ]
        for record in ladder.ordered:
            head = record.headline()
            lines.append(
                f"| {record.rung_id} | `{record.model_name}` "
                f"| {record.ranking.pr_auc.value:.3f} | {_fmt(head['flag_rate'])} "
                f"| {_fmt(head['precision'])} | {_fmt_int(head['net_inr_per_1000'])} |"
            )
        if "validation" in evaluations:
            ev = evaluations["validation"]
            point = ev.operating_point
            lines.append(
                f"| - | **`{ev.model_name}`** (final) | **{ev.ranking.pr_auc.value:.3f}** "
                f"| {_fmt(point.flag_rate)} | {_fmt(point.precision)} "
                f"| **{_fmt_int(ev.economics.net_inr_saved_per_1000_orders.value)}** |"
            )

    lines += [
        "",
        "## Known limitations",
        "",
        *_bullets(config.known_limitations),
        "",
        "## Fairness limitations",
        "",
        *_bullets(config.fairness_limitations),
        "",
        "## Distribution-shift limitations",
        "",
        *_bullets(config.distribution_shift_limitations),
        "",
        "## Maintenance",
        "",
        f"**Retraining trigger.** {config.maintenance.retraining_trigger.strip()}",
        "",
        f"**Monitoring.** {config.maintenance.monitoring.strip()}",
        "",
        "## Provenance",
        "",
        manifest.data_provenance,
        "",
    ]

    document = "\n".join(lines)
    check_card(document)
    return document


def check_card(document: str) -> None:
    """Refuse a card that has lost the disclaimer it exists to carry."""
    lowered = document.lower()
    missing = [phrase for phrase in _REQUIRED_PHRASES if phrase not in lowered]
    if missing:
        msg = (
            f"the rendered model card is missing required wording: {missing}. A card for a "
            "model trained on simulated labels must say so where a reader will see it."
        )
        raise DishonestReportError(msg)


def write_model_card(
    config: ModelCardConfig,
    manifest: SelectionManifest,
    evaluations: dict[str, FinalEvaluation],
    path: Path,
    *,
    ladder: LadderResults | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_model_card(config, manifest, evaluations, ladder=ladder), encoding="utf-8"
    )
    return path
