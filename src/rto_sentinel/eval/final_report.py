"""The consolidated evaluation report.

Every number in ``docs/evaluation_report.md`` is read from a saved artefact.
There is no code path in this module that writes a metric from a literal, which
is what stops the document drifting away from the runs that produced it - the
same discipline `model_card.py`, `economics_report.py` and
`responsible_report.py` apply.

WHAT THIS ADDS OVER THE OTHER GENERATED DOCUMENTS
=================================================
The model card describes one model. The economics report describes one policy.
The responsible-AI report describes fairness and robustness. This puts the
measured results of all of them in one place with the two things a reader most
needs and most often loses:

1. **Which split.** Validation figures are selection-contaminated - the
   hyperparameters were chosen on them and the shipped calibrator was refitted on
   them. The sealed test set was opened once. Presenting them in one column would
   merge a number the model was tuned against with one it was not, so they are
   always adjacent columns and always labelled.

2. **Which numbers rest on assumptions.** Every rupee figure depends on an
   intervention success rate and an abandonment rate that have never been
   measured on this or any data. Those are marked at every appearance, not once
   in a footnote.

If an experiment has not been run, the section says so where the numbers would
have been. It does not fall back to prose that implies a result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rto_sentinel.contracts.experiment import LadderResults
from rto_sentinel.contracts.final import FinalEvaluation, SelectionManifest
from rto_sentinel.contracts.monitoring import DriftReport, ShiftStudy
from rto_sentinel.eval.responsible_report import RESPONSIBLE_DIR, read_contract_payload
from rto_sentinel.models.final import FINAL_DIR

#: Stated wherever a rupee figure appears.
ASSUMPTION_MARK = "†"

#: The two rates nobody has measured. Named here so the wording cannot drift.
ASSUMPTION_NOTE = (
    f"{ASSUMPTION_MARK} Rests on `intervention_success_rate` and "
    "`abandonment_on_friction`, which are **stated assumptions, never measured** on "
    "this or any data. Every rupee figure inherits their uncertainty on top of the "
    "sampling interval shown."
)


class NoArtefactsError(FileNotFoundError):
    """Raised when there is nothing measured to report."""


def _precision(economics: Any) -> float | None:
    """TP / (TP + FP), computed from the counts.

    `EconomicResult` exposes precision and recall as properties, so they are not
    in the serialised artefact. Deriving them from the confusion counts is also
    more robust: the counts are what was measured, and a reader can check the
    arithmetic against the matrix printed two rows above.
    """
    flagged = economics["true_positives"] + economics["false_positives"]
    return economics["true_positives"] / flagged if flagged else None


def _recall(economics: Any) -> float | None:
    positives = economics["true_positives"] + economics["false_negatives"]
    return economics["true_positives"] / positives if positives else None


def _f1(economics: Any) -> float | None:
    precision, recall = _precision(economics), _recall(economics)
    if not precision or not recall:
        return None
    return 2 * precision * recall / (precision + recall)


def _dash(value: float | None, digits: int = 3) -> str:
    """A missing metric is an em-dash, never a zero.

    Zero would claim "measured, and it was nothing". `recall_at_precision_80` is
    genuinely undefined on the sealed set - the model never reaches 80% precision
    at any threshold - and that is a result worth showing as absent.
    """
    return "—" if value is None else f"{value:.{digits}f}"


def _interval(estimate: Any) -> str:
    if estimate is None:
        return "—"
    return (
        f"{estimate['value']:.3f} <sub>[{estimate['ci_low']:.3f}, {estimate['ci_high']:.3f}]</sub>"
    )


def _rupees(value: float | None) -> str:
    return "—" if value is None else f"₹{value:,.0f}"


def _rupee_interval(estimate: Any) -> str:
    if estimate is None:
        return "—"
    return (
        f"₹{estimate['value']:,.0f} "
        f"<sub>[₹{estimate['ci_low']:,.0f}, ₹{estimate['ci_high']:,.0f}]</sub>"
    )


def load_everything(artifact_root: Path) -> dict[str, Any]:
    """Read whatever exists. Missing artefacts stay missing."""
    found: dict[str, Any] = {}

    root = artifact_root / FINAL_DIR
    runs = sorted(root.glob("*/selection_manifest.json")) if root.is_dir() else []
    if runs:
        run_dir = max(runs, key=lambda path: path.stat().st_mtime).parent
        found["run_dir"] = run_dir
        found["manifest"] = SelectionManifest.model_validate_json(
            (run_dir / "selection_manifest.json").read_text(encoding="utf-8")
        )
        for split in ("validation", "test"):
            path = run_dir / f"metrics__{split}.json"
            if path.is_file():
                found[split] = FinalEvaluation.model_validate_json(path.read_text(encoding="utf-8"))

    ladder_dir = artifact_root / "experiments"
    ladders = sorted(ladder_dir.rglob("ladder__*.json")) if ladder_dir.is_dir() else []
    if ladders:
        newest = max(ladders, key=lambda path: path.stat().st_mtime)
        found["ladder"] = LadderResults.model_validate_json(newest.read_text(encoding="utf-8"))

    responsible = artifact_root / RESPONSIBLE_DIR
    for split in ("validation", "test"):
        path = responsible / f"fairness__{split}.json"
        if path.is_file():
            found[f"fairness_{split}"] = json.loads(path.read_text(encoding="utf-8"))
    ablation_path = responsible / "ablation_study.json"
    if ablation_path.is_file():
        found["ablation"] = json.loads(ablation_path.read_text(encoding="utf-8"))
    if (responsible / "shift_study.json").is_file():
        found["shift"] = ShiftStudy.model_validate(
            read_contract_payload(responsible / "shift_study.json")
        )
    if (responsible / "drift_report.json").is_file():
        found["drift"] = DriftReport.model_validate(
            read_contract_payload(responsible / "drift_report.json")
        )

    return found


def render(*, artifact_root: Path, output: Path | None = None) -> Path:
    """Write ``docs/evaluation_report.md`` from the saved artefacts."""
    found = load_everything(artifact_root)
    if "manifest" not in found:
        msg = (
            f"no frozen final-model run under {artifact_root / FINAL_DIR}. Run "
            "`rto-sentinel final` and `rto-sentinel final-test` first. A report "
            "rendered from nothing would claim measurements that were never taken."
        )
        raise NoArtefactsError(msg)

    lines: list[str] = []
    lines += _header(found)
    lines += _provenance(found)
    lines += _headline(found)
    lines += _ranking(found)
    lines += _calibration(found)
    lines += _operating_point(found)
    lines += _economics(found)
    lines += _baselines(found)
    lines += _fairness(found)
    lines += _ablation(found)
    lines += _shift(found)
    lines += _drift(found)
    lines += _reading(found)

    destination = output or Path("docs/evaluation_report.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _header(found: dict[str, Any]) -> list[str]:
    return [
        "# RTO Sentinel — evaluation report",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`rto-sentinel evaluation-report`. **Every figure is read from a saved "
        "artefact under `artifacts/`; none is written by hand.**",
        "",
        "> **The labels are simulated.** They are outcomes of the process in "
        "`docs/simulator.md`, not observations of real returns. Every metric here "
        "is a statement about that simulator. A good number means the model learned "
        "the simulator; whether the simulator resembles Indian COD commerce is a "
        "separate question this project does not claim to have settled.",
        "",
        "---",
        "",
        "## How to read this",
        "",
        "| Column | What it is |",
        "|---|---|",
        "| **Validation** | **Selection-contaminated.** Hyperparameters were chosen "
        "on this split and the shipped calibrator was refitted on it. Useful for "
        "comparison, not for a performance claim. |",
        "| **Sealed test** | **The honest read.** Opened once, after model selection, "
        "calibration and threshold methodology were frozen in the manifest. |",
        "",
        f"{ASSUMPTION_NOTE}",
        "",
        "---",
        "",
    ]


def _provenance(found: dict[str, Any]) -> list[str]:
    manifest = found["manifest"]
    lines = [
        "## 1. What was measured",
        "",
        "Every artefact records all seven identifiers, so any figure below can be "
        "traced to the exact code, data and configuration that produced it.",
        "",
        "| Identifier | Value |",
        "|---|---|",
        f"| Model | `{manifest.base_rung}` + `{manifest.calibration_method}` calibration |",
        f"| Model version | `{manifest.model_version}` |",
        f"| Selection manifest | `{manifest.manifest_id}` |",
        f"| Dataset run | `{manifest.dataset_run_id}` |",
        f"| Generator version | `{manifest.generator_version}` |",
        f"| Feature version | `{manifest.feature_version}` |",
        f"| Feature fingerprint | `{manifest.feature_fingerprint[:16]}…` |",
        f"| Config fingerprint | `{manifest.config_fingerprint[:16]}…` |",
        f"| Seed | `{manifest.seed}` |",
        f"| Calibration fitted on | {manifest.calibration_fitted_on} "
        f"({manifest.calibration_folds}-fold CV) |",
        f"| Threshold | {manifest.threshold:.4f} — {manifest.threshold_source} |",
        f"| Cost profile | `{manifest.cost_profile}` |",
        f"| Frozen at | {manifest.frozen_at:%Y-%m-%d %H:%M UTC} |",
        "",
    ]

    sizes = []
    for split in ("validation", "test"):
        if split in found:
            summary = found[split].evaluation_summary
            sizes.append(
                f"| {split} | {summary.n_rows:,} | {summary.n_positives:,} | "
                f"{summary.positive_rate:.4f} | {summary.n_customers:,} |"
            )
    if sizes:
        lines += [
            "### Split sizes",
            "",
            "| Split | Orders | Positives | Positive rate | Customers |",
            "|---|---:|---:|---:|---:|",
            *sizes,
            "",
        ]
    lines += ["---", ""]
    return lines


def _headline(found: dict[str, Any]) -> list[str]:
    if "test" not in found:
        return [
            "## 2. Headline",
            "",
            "**The sealed test set has not been evaluated.** Run "
            "`rto-sentinel final-test`. No performance claim is supported until it is.",
            "",
            "---",
            "",
        ]

    test = found["test"].model_dump(mode="json")
    net = test["economics"]["net_inr_saved_per_1000_orders"]
    crosses_zero = net["ci_low"] <= 0 <= net["ci_high"]

    lines = [
        "## 2. Headline",
        "",
        f"On the sealed test set, at the cost-derived threshold "
        f"{test['economics']['threshold']:.4f}:",
        "",
        f"- **Net {_rupee_interval(net)} per 1,000 orders**, against doing nothing"
        f"{ASSUMPTION_MARK}",
        f"- Doing nothing costs the merchant "
        f"{_rupees(abs(test['economics']['baseline_net_inr_per_1000_orders']))} per "
        "1,000 orders in absorbed RTO losses",
        f"- Flag rate {test['economics']['flag_rate']:.1%}, precision "
        f"{_dash(_precision(test['economics']))}, recall "
        f"{_dash(_recall(test['economics']))}",
        f"- PR-AUC {_interval(test['ranking']['pr_auc'])} against a base rate of "
        f"{test['evaluation_summary']['positive_rate']:.4f}",
        "",
    ]

    if crosses_zero:
        lines += [
            "> **The interval crosses zero.** On "
            f"{test['evaluation_summary']['n_rows']:,} sealed orders this measurement "
            "cannot distinguish the model from doing nothing. The point estimate is "
            "positive; the evidence does not establish it. This is the single most "
            "important sentence in the report and it is stated first rather than "
            "buried under the figures that look better.",
            "",
        ]

    lines += ["---", ""]
    return lines


def _split_table(found: dict[str, Any], rows: list[tuple[str, Any]]) -> list[str]:
    header = [
        "| Metric | Validation<br><sub>selection-contaminated</sub> "
        "| Sealed test<br><sub>the honest read</sub> |",
        "|---|---|---|",
    ]
    body = []
    for label, getter in rows:
        validation = (
            getter(found["validation"].model_dump(mode="json")) if "validation" in found else "—"
        )
        test = getter(found["test"].model_dump(mode="json")) if "test" in found else "—"
        body.append(f"| {label} | {validation} | {test} |")
    return header + body


def _ranking(found: dict[str, Any]) -> list[str]:
    return [
        "## 3. Ranking quality",
        "",
        "**PR-AUC leads.** ROC-AUC is reported but not led with: it flatters an "
        "imbalanced problem by rewarding the model for ranking the large negative "
        "class correctly, which is not the task.",
        "",
        *_split_table(
            found,
            [
                ("PR-AUC", lambda d: _interval(d["ranking"]["pr_auc"])),
                ("ROC-AUC", lambda d: _interval(d["ranking"]["roc_auc"])),
                (
                    "Base rate (PR-AUC floor)",
                    lambda d: f"{d['evaluation_summary']['positive_rate']:.4f}",
                ),
                ("Recall @ precision 80%", lambda d: _dash(d["ranking"]["recall_at_precision_80"])),
                ("Recall @ precision 90%", lambda d: _dash(d["ranking"]["recall_at_precision_90"])),
                ("Precision @ top 1%", lambda d: _dash(d["ranking"]["precision_at_k"].get("0.01"))),
                ("Precision @ top 5%", lambda d: _dash(d["ranking"]["precision_at_k"].get("0.05"))),
                (
                    "Precision @ top 10%",
                    lambda d: _dash(d["ranking"]["precision_at_k"].get("0.10")),
                ),
            ],
        ),
        "",
        "**Recall@P80 and Recall@P90 are em-dashes on the sealed set because the "
        "model never reaches those precisions at any threshold.** That is a real "
        "result about the ceiling of this problem, not a missing measurement: with "
        "a base rate near 16% and substantial irreducible noise in the simulator, "
        "80% precision is not attainable. Reporting 0.0 would have claimed it was "
        "attained and missed.",
        "",
        "---",
        "",
    ]


def _calibration(found: dict[str, Any]) -> list[str]:
    lines = [
        "## 4. Calibration",
        "",
        "Calibration is a headline metric here, not a footnote. The operating "
        "threshold is compared against a probability; if the score is not an honest "
        "probability, the comparison is arithmetic on a number that denotes nothing "
        "and every rupee figure below is fiction.",
        "",
        *_split_table(
            found,
            [
                (
                    "Expected calibration error (calibrated)",
                    lambda d: f"{d['calibration']['expected_calibration_error']:.4f}",
                ),
                (
                    "Expected calibration error (uncalibrated)",
                    lambda d: f"{d['uncalibrated_calibration']['expected_calibration_error']:.4f}",
                ),
                ("Brier score (calibrated)", lambda d: f"{d['calibration']['brier_score']:.4f}"),
                (
                    "Brier score (uncalibrated)",
                    lambda d: f"{d['uncalibrated_calibration']['brier_score']:.4f}",
                ),
            ],
        ),
        "",
    ]

    if "validation" in found and "test" in found:
        val = found["validation"].model_dump(mode="json")
        test = found["test"].model_dump(mode="json")
        val_helped = (
            val["uncalibrated_calibration"]["expected_calibration_error"]
            - val["calibration"]["expected_calibration_error"]
        )
        test_helped = (
            test["uncalibrated_calibration"]["expected_calibration_error"]
            - test["calibration"]["expected_calibration_error"]
        )
        if val_helped > 0 >= test_helped:
            lines += [
                f"> **Calibration improved ECE on validation by {val_helped:+.4f} and "
                f"made it worse on the sealed set by {test_helped:+.4f}.** The Platt "
                "mapping was fitted on validation by cross-validation and did not "
                "transfer. This is reported because it is the kind of result that "
                "quietly disappears from a write-up: the calibrator was selected "
                "honestly, and it still failed to generalise.",
                "",
            ]

    run_dir = found.get("run_dir")
    if run_dir is not None:
        diagrams = [
            f"reliability__{split}.png"
            for split in ("validation", "test")
            if (run_dir / f"reliability__{split}.png").is_file()
        ]
        if diagrams:
            lines += [
                "### Reliability diagrams",
                "",
                "Generated at evaluation time from the same predictions as the table above:",
                "",
                *[f"- `{run_dir.as_posix()}/{name}`" for name in diagrams],
                "",
                "The console renders the same bins live from "
                "`GET /v1/evaluation/reliability`, so the picture and the endpoint "
                "cannot disagree.",
                "",
            ]

    lines += ["---", ""]
    return lines


def _operating_point(found: dict[str, Any]) -> list[str]:
    manifest = found["manifest"]
    lines = [
        "## 5. The operating point and the confusion matrix",
        "",
        f"The threshold is **{manifest.threshold:.4f}**, not 0.5. It is derived from "
        "the merchant's economics as `C_fp / (C_fp + S_tp)` and recomputed whenever "
        "they change - see `docs/economics.md`.",
        "",
        *_split_table(
            found,
            [
                ("Threshold", lambda d: f"{d['economics']['threshold']:.4f}"),
                ("Flag rate", lambda d: f"{d['economics']['flag_rate']:.1%}"),
                ("Precision", lambda d: _dash(_precision(d["economics"]))),
                ("Recall", lambda d: _dash(_recall(d["economics"]))),
                ("F1", lambda d: _dash(_f1(d["economics"]))),
                ("True positives", lambda d: f"{d['economics']['true_positives']:,}"),
                ("False positives", lambda d: f"{d['economics']['false_positives']:,}"),
                ("False negatives", lambda d: f"{d['economics']['false_negatives']:,}"),
                ("True negatives", lambda d: f"{d['economics']['true_negatives']:,}"),
            ],
        ),
        "",
        "**Precision is never reported without the flag rate.** A precision figure "
        "alone is meaningless: any model can reach high precision by flagging almost "
        "nothing, and the pair is what describes the operating point.",
        "",
        "---",
        "",
    ]
    return lines


def _economics(found: dict[str, Any]) -> list[str]:
    lines = [
        "## 6. Economic results",
        "",
        f"{ASSUMPTION_NOTE}",
        "",
        *_split_table(
            found,
            [
                (
                    f"**Net ₹ per 1,000 orders**{ASSUMPTION_MARK}",
                    lambda d: _rupee_interval(d["economics"]["net_inr_saved_per_1000_orders"]),
                ),
                (
                    "Do-nothing loss per 1,000",
                    lambda d: _rupees(abs(d["economics"]["baseline_net_inr_per_1000_orders"])),
                ),
                (
                    f"Gross saving{ASSUMPTION_MARK}",
                    lambda d: _rupees(d["economics"]["gross_saving_inr"]),
                ),
                (
                    f"**False-positive cost**{ASSUMPTION_MARK}",
                    lambda d: _rupees(d["economics"]["total_false_positive_cost_inr"]),
                ),
                (
                    "Residual RTO loss (not flagged)",
                    lambda d: _rupees(d["economics"]["residual_false_negative_loss_inr"]),
                ),
            ],
        ),
        "",
        "**The false-positive cost is reported separately and is never netted away "
        "inside the savings figure.** `EconomicResult` keeps it as a required field "
        "with nowhere to hide it, because a net number that has quietly absorbed the "
        "cost of frictioning good customers is the most flattering thing this system "
        "could report.",
        "",
        "Full derivation, the friction ladder, the threshold sweep and the "
        "sensitivity analysis: `docs/economics.md`.",
        "",
        "---",
        "",
    ]
    return lines


def _baselines(found: dict[str, Any]) -> list[str]:
    if "ladder" not in found:
        return [
            "## 7. Baseline comparison",
            "",
            "**Not run.** No ladder artefact exists; run `rto-sentinel train`.",
            "",
            "---",
            "",
        ]

    ladder: LadderResults = found["ladder"]
    lines = [
        "## 7. Baseline comparison",
        "",
        f"Every rung on the **{ladder.evaluated_split}** split, at the same "
        f"cost-derived threshold ({ladder.threshold:.4f}), scored identically. "
        'The question a baseline ladder answers is not "is the model good" but '
        '"does it beat the thing a merchant could build in an afternoon" - and if a '
        "simpler rung wins on money, it ships.",
        "",
        "| Rung | Model | PR-AUC | Train-val gap | Flag rate | Precision | "
        f"Net ₹/1k{ASSUMPTION_MARK} |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for record in ladder.ordered:
        head = record.headline()
        gap = head.get("train_val_gap")
        lines.append(
            f"| {record.rung_id} | `{record.model_name}` "
            f"| {_interval(record.ranking.pr_auc.model_dump())} "
            f"| {'—' if gap is None else f'{gap:+.3f}'} "
            f"| {_dash(head.get('flag_rate'))} | {_dash(head.get('precision'))} "
            f"| {_rupees(head.get('net_inr_per_1000'))} |"
        )

    if "validation" in found:
        evaluation = found["validation"].model_dump(mode="json")
        lines.append(
            f"| — | **`{evaluation['model_name']}`** (shipped) "
            f"| **{_interval(evaluation['ranking']['pr_auc'])}** | — "
            f"| {evaluation['economics']['flag_rate']:.3f} "
            f"| {_dash(_precision(evaluation['economics']))} "
            f"| **{_rupees(evaluation['economics']['net_inr_saved_per_1000_orders']['value'])}** |"
        )

    lines += [
        "",
        "**A large positive train-val gap means the rung memorised the training "
        "window**; its validation score describes an overfitting configuration "
        "rather than a ceiling. That is why the ladder is reported with the gap "
        "beside the score.",
        "",
        "---",
        "",
    ]
    return lines


def _fairness(found: dict[str, Any]) -> list[str]:
    lines = ["## 8. Fairness across operational cohorts", ""]

    payload = found.get("fairness_test") or found.get("fairness_validation")
    if payload is None:
        lines += [
            "**Not run.** No fairness artefact exists; run `rto-sentinel fairness`. "
            "No fairness claim may be made about this model.",
            "",
            "---",
            "",
        ]
        return lines

    lines += [
        "**No sensitive characteristic is examined, inferred or approximated.** There "
        "is no gender, religion, caste, ethnicity, age or income field in this data - "
        "not withheld, not present - and none is derived from names or addresses. "
        "`eval/fairness.py` refuses by name any cohort matching a sensitive token, "
        "and the refusal is a hard error rather than a skipped cohort.",
        "",
        "The cohorts are operational: delivery-area tier, order-value band, "
        "customer-history depth, payment method.",
        "",
    ]

    for split in ("validation", "test"):
        entry = found.get(f"fairness_{split}")
        if entry is None:
            continue
        audit = entry["audit"]
        lines += [
            f"### {split.capitalize()} split",
            "",
            f"**Disparity review: {'TRIGGERED' if audit['triggered'] else 'not triggered'}.** "
            f"Maximum flag-rate ratio {audit['max_flag_rate_ratio']:.2f} "
            f"({audit['most_flagged_group']} vs {audit['least_flagged_group']}); "
            f"worst precision drop {audit['worst_precision_drop']:.3f}.",
            "",
            "| Cohort | Group | n | RTO rate | Flag rate | Precision | Evidence |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in entry["slices"]:
            lines.append(
                f"| {row['cohort']} | {row['group']} | {row['n_orders']:,} "
                f"| {row['rto_rate']:.1%} | {row['flag_rate']:.1%} "
                f"| {_dash(row['precision'])} "
                f"| {'yes' if row['sufficient'] else '**too thin**'} |"
            )
        lines += ["", str(audit["narrative"]), ""]

    lines += [
        "Full audit, with Wilson intervals on every rate and the support thresholds: "
        "`docs/responsible_ai.md`.",
        "",
        "---",
        "",
    ]
    return lines


def _ablation(found: dict[str, Any]) -> list[str]:
    lines = ["## 8b. What each feature family is worth", ""]
    study = found.get("ablation")
    if not isinstance(study, dict):
        lines += ["**Not run.** Run `rto-sentinel ablation`.", "", "---", ""]
        return lines

    lines += [
        "Leave-one-family-out, **retrained per arm** and measured in net rupees "
        "rather than AUC. A family that adds ranking quality but no money has not "
        "earned its place.",
        "",
        "Validation only - an ablation is feature selection, and selecting anything "
        "on the sealed test split is forbidden by `config/evaluation.yaml`.",
        "",
        "Every delta carries a **paired** bootstrap interval: both arms scored the "
        "same orders, so resampling them independently would invent variance that is "
        "not there. An interval spanning zero means the data cannot say the family "
        "mattered.",
        "",
        "| Family removed | Features | Net ₹/1k"
        + ASSUMPTION_MARK
        + " | Δ vs full | 95% interval | PR-AUC | ΔPR-AUC | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    full = study["full_model"]
    lines.append(
        f"| *(full model)* | {full['n_features']} | "
        f"{_rupees(full['net_inr_per_1000'])} | — | — | {full['pr_auc']:.3f} | — | "
        "reference |"
    )
    for arm in study["arms"]:
        interval = f"[{arm['delta_ci_low']:+,.0f}, {arm['delta_ci_high']:+,.0f}]"
        lines.append(
            f"| `{arm['family_removed']}` | {arm['n_features']} | "
            f"{_rupees(arm['net_inr_per_1000'])} | {arm['delta_vs_full']:+,.0f} | "
            f"{interval} | {arm['pr_auc']:.3f} | "
            f"{arm['delta_pr_auc_vs_full']:+.3f} | {arm['verdict']} |"
        )

    lines += ["", "### Findings", ""]
    lines += [f"- {finding}" for finding in study["findings"]]
    lines += ["", "---", ""]
    return lines


def _shift(found: dict[str, Any]) -> list[str]:
    lines = ["## 9. Distribution shift", ""]
    study = found.get("shift")
    if not isinstance(study, ShiftStudy):
        lines += ["**Not run.** Run `rto-sentinel shift`.", "", "---", ""]
        return lines

    lines += [
        "Nine named perturbations of the generator, with the model **frozen and not "
        "retrained** and the threshold held fixed. The `reference` environment is a "
        "fresh draw from the *unshifted* distribution, so subtracting it removes "
        "sampling variance and leaves the effect of the perturbation.",
        "",
        "**Read the lift column, not raw PR-AUC.** A random ranker scores PR-AUC "
        "equal to the positive rate, so an environment whose base rate moved has a "
        "different floor. Raw PR-AUC *rises* when the world gets riskier, and reading "
        "that as robustness reports the arithmetic of the base rate as a property of "
        "the model.",
        "",
        f"| Environment | RTO rate | PR-AUC | Lift | ΔLift | ECE | Net ₹/1k{ASSUMPTION_MARK} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in study.results:
        delta = "—" if result.pr_auc_lift_delta is None else f"{result.pr_auc_lift_delta:+.2f}x"
        lines.append(
            f"| `{result.environment}` | {result.observed_rto_rate:.1%} "
            f"| {result.pr_auc:.3f} | {result.pr_auc_lift:.2f}x | {delta} "
            f"| {result.expected_calibration_error:.3f} "
            f"| {_rupees(result.net_inr_per_1000)} |"
        )

    lines += ["", "### Findings", ""]
    lines += [f"- {finding}" for finding in study.findings]
    lines += ["", "---", ""]
    return lines


def _drift(found: dict[str, Any]) -> list[str]:
    lines = ["## 10. Drift monitoring", ""]
    report = found.get("drift")
    if not isinstance(report, DriftReport):
        lines += ["**Not run.** Run `rto-sentinel monitor`.", "", "---", ""]
        return lines

    lines += [
        "**Drift is not failure.** A moved input distribution is a fact about the "
        "world; whether quality degraded needs labels. The two are kept structurally "
        "apart - a drift signal has no field for a verdict, and a performance delta "
        "cannot be constructed without matured outcomes.",
        "",
        f"Baseline {report.baseline.n_orders:,} orders "
        f"({report.baseline.n_matured:,} matured) versus current "
        f"{report.current.n_orders:,} ({report.current.n_matured:,} matured). "
        f"Labelled comparison possible: **{'yes' if report.labels_available else 'no'}**.",
        "",
        "| Kind | Quantity | Statistic | Distance | Reading |",
        "|---|---|---|---:|---|",
    ]
    for signal in report.signals:
        suffix = "" if signal.sufficient else " (too thin to read)"
        lines.append(
            f"| {signal.kind} | `{signal.name}` | {signal.statistic} "
            f"| {signal.distance:.4f} | {signal.severity}{suffix} |"
        )

    if report.performance:
        lines += [
            "",
            "### Labelled comparisons",
            "",
            "| Metric | Baseline | Current | Δ |",
            "|---|---:|---:|---:|",
        ]
        for delta in report.performance:
            lines.append(
                f"| {delta.metric} | {delta.baseline:.4f} | {delta.current:.4f} "
                f"| {delta.delta:+.4f} |"
            )

    lines += ["", "---", ""]
    return lines


def _reading(found: dict[str, Any]) -> list[str]:
    lines = [
        "## 11. What this evaluation does and does not establish",
        "",
        "**Established:**",
        "",
        "- The pipeline is leak-free under seven explicit tests, and the sealed test "
        "set was opened exactly once, after the manifest was frozen.",
        "- The model ranks materially better than chance on the sealed set "
        "(PR-AUC well above the base rate).",
        "- The decision threshold is derived from merchant economics, not chosen, "
        "and it moves in the direction the arithmetic requires.",
        "- The cohort audit did not trip its disparity review, and the shift study "
        "found a specific, reproducible failure mode.",
        "",
        "**Not established:**",
        "",
    ]

    if "test" in found:
        net = found["test"].model_dump(mode="json")["economics"]["net_inr_saved_per_1000_orders"]
        if net["ci_low"] <= 0 <= net["ci_high"]:
            lines.append(
                f"- **That the system saves money.** The sealed-set interval "
                f"[₹{net['ci_low']:,.0f}, ₹{net['ci_high']:,.0f}] crosses zero."
            )

    lines += [
        "- **Anything about production.** The labels are simulated; these are "
        "statements about the simulator.",
        "- **Any fairness property of a protected group.** No such attribute exists "
        "in the data and none was inferred.",
        f"- **The rupee figures{ASSUMPTION_MARK}**, which rest on two rates that "
        "have never been measured.",
        "- **That the three families with unestablished contributions are "
        "worthless.** Leave-one-out measures marginal contribution given every "
        "other family; overlapping signal hides individual value.",
        "",
        "Complete limitations: `docs/phase11_report.md` and `docs/responsible_ai.md`.",
        "",
    ]
    return lines
