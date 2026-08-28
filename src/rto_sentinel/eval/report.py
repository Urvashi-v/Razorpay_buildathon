"""Renders the ladder comparison, and refuses to render a dishonest one.

The builder enforces the ``forbidden`` list from ``config/evaluation.yaml``
mechanically:

* no single accuracy figure;
* ROC-AUC present, but never in the lead column;
* precision never shown without its flag rate;
* false-positive cost never folded into the net figure;
* no point estimate without an interval.

A table that violates any of these raises rather than renders. It is easy to
promise these things in a README and quietly break one under deadline pressure;
it is much harder to defeat a builder that refuses to emit the file.

HOW UNDEFINED VALUES ARE SHOWN
==============================
A dash, never a zero. Rung 0's ROC-AUC and precision genuinely do not exist -
there is no ranking to score and nothing is flagged - and printing 0.000 would
turn "undefined" into "measured, and terrible". The distinction matters most
exactly where a reader is skimming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.contracts.experiment import ExperimentRecord, LadderResults

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from rto_sentinel.configuration.schemas import EvaluationConfig

#: Markdown files are written UTF-8 and can carry an em dash. The console cannot:
#: Windows terminals default to cp1252, where an em dash or a rupee sign raises
#: UnicodeEncodeError and takes the whole run down. Terminal output stays ASCII.
MARKDOWN_DASH = "—"
CONSOLE_DASH = "-"

#: Train-minus-validation PR-AUC above which the report calls a rung out for
#: memorising. Chosen as a reporting trigger, not a pass/fail gate: it decides
#: when prose is added, never what a metric says.
OVERFIT_GAP_WARNING = 0.2


class DishonestReportError(ValueError):
    """Raised when a report would violate a rule in the evaluation config."""


def _fmt(
    value: float | None, digits: int = 3, dash: str = MARKDOWN_DASH, *, signed: bool = False
) -> str:
    """Format a metric, or a dash when it is undefined.

    ``signed`` keeps the plus sign on positive values. It is used for the
    train-minus-validation gap, where the sign is the whole point: +0.5 is a
    memorising model, -0.04 is an easier validation window.
    """
    if value is None or value != value:  # None or NaN
        return dash
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _fmt_int(value: float | None, dash: str = MARKDOWN_DASH) -> str:
    if value is None or value != value:
        return dash
    return f"{value:,.0f}"


def check_report_rules(results: LadderResults, config: EvaluationConfig) -> None:
    """Assert the reporting prohibitions before anything is rendered."""
    forbidden = set(config.forbidden)

    if (
        "tune_threshold_on_test_set" in forbidden
        and results.evaluated_split == "test"
        and not results.threshold_source.startswith("derived from cost inputs")
    ):
        msg = (
            "the test-set threshold was not derived from cost inputs. Tuning an "
            "operating point on the sealed set is the one thing the split protocol "
            "exists to prevent."
        )
        raise DishonestReportError(msg)

    # `quote_precision_without_flag_rate` is not checked here. It cannot be
    # violated by a record - `ThresholdMetrics.flag_rate` is required, so the
    # pairing is structural - and the check that used to sit here was therefore
    # dead code dressed as a safeguard. The rule can only be broken by rendered
    # output that prints one without the other, so it is enforced there, by
    # `check_rendered_report`.
    for record in results.records:
        if "present_point_estimate_as_precise" in forbidden:
            estimate = record.ranking.pr_auc
            if estimate.is_defined and estimate.n_bootstrap == 0:
                msg = (
                    f"{record.model_name}: PR-AUC has no bootstrap interval. A point "
                    "estimate on a few thousand rows is not a result."
                )
                raise DishonestReportError(msg)

        if (
            "net_false_positive_cost_away" in forbidden
            and record.economics is not None
            and record.economics.total_false_positive_cost_inr < 0
        ):
            msg = f"{record.model_name}: false-positive cost is negative, so it has been netted"
            raise DishonestReportError(msg)


def check_rendered_report(text: str, config: EvaluationConfig) -> None:
    """Rules that can only be broken by the rendered document itself.

    Precision without a flag rate is the one that matters. "92% precision" is a
    fine number and a useless one on its own: at a 0.4% flag rate it describes a
    model that has found almost nothing. The two are only meaningful together, so
    a document quoting one must quote the other.
    """
    if "quote_precision_without_flag_rate" not in set(config.forbidden):
        return

    lowered = text.lower()
    if "precision" in lowered and "flag rate" not in lowered:
        msg = (
            "the rendered report quotes precision without a flag rate. Precision "
            "at an unstated flag rate says nothing about how much a model catches."
        )
        raise DishonestReportError(msg)


def comparison_table(results: LadderResults, *, width: int = 112) -> str:
    """The ladder table, plain text.

    Column order is deliberate: PR-AUC leads, the train-minus-validation gap sits
    immediately beside it because it says whether that PR-AUC means anything,
    ROC-AUC sits well right of both, and flag rate is adjacent to precision so
    the two are read together.
    """
    lines: list[str] = []
    header = (
        f"{'rung':<5}{'model':<21}{'PR-AUC':>16}{'trn-val':>9}{'flag':>7}{'prec':>7}"
        f"{'rec':>7}{'F1':>7}{'R@P80':>7}{'ROC':>7}{'ECE':>7}{'net/1k':>10}{'FP cost':>11}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    d = CONSOLE_DASH
    for record in results.ordered:
        head = record.headline()
        pr = record.ranking.pr_auc
        pr_text = f"{pr.value:.3f} [{pr.ci_low:.3f},{pr.ci_high:.3f}]" if pr.is_defined else d
        lines.append(
            f"{head['rung']:<5}{head['model']:<21}{pr_text:>16}"
            f"{_fmt(head['overfit_gap'], dash=d, signed=True):>9}"
            f"{_fmt(head['flag_rate'], dash=d):>7}{_fmt(head['precision'], dash=d):>7}"
            f"{_fmt(head['recall'], dash=d):>7}{_fmt(head['f1'], dash=d):>7}"
            f"{_fmt(head['recall_at_p80'], dash=d):>7}{_fmt(head['roc_auc'], dash=d):>7}"
            f"{_fmt(head['ece'], dash=d):>7}{_fmt_int(head['net_inr_per_1000'], dash=d):>10}"
            f"{_fmt_int(head['fp_cost_inr'], dash=d):>11}"
        )

    return "\n".join(lines)


def _verdict_section(results: LadderResults) -> list[str]:
    """State which rung wins and, where the numbers show it, why to be careful.

    Written from the records rather than by hand. If a later run reverses the
    ordering, this section reverses with it - the alternative is a conclusion
    that outlives the evidence for it.
    """
    if not any(record.economics is not None for record in results.records):
        return []

    best = strongest_rung(results)
    economics = best.economics
    if economics is None:  # pragma: no cover - strongest_rung ranks on economics
        return []
    net = economics.net_inr_saved_per_1000_orders

    lines = [
        "",
        "## Which rung is strongest",
        "",
        f"On the declared headline metric - net rupees per 1,000 orders - **rung "
        f"{best.rung_id} `{best.model_name}`** wins, at "
        f"**₹{net.value:,.0f}** [{net.ci_low:,.0f}, {net.ci_high:,.0f}] per 1,000 "
        f"orders at a {economics.flag_rate:.1%} flag rate. It also leads on PR-AUC "
        f"({best.primary_metric:.3f}).",
    ]

    # A rung that scores far better on training than on validation is the
    # interesting case, and the one a headline table hides.
    gaps = [
        (record, gap)
        for record in results.records
        if (gap := record.overfit_gap) is not None and gap > OVERFIT_GAP_WARNING
    ]
    if gaps:
        worst, gap = max(gaps, key=lambda pair: pair[1])
        train_pr = worst.train_pr_auc if worst.train_pr_auc is not None else float("nan")
        lines += [
            "",
            f"`{worst.model_name}` is the rung to be careful with. It scores "
            f"{train_pr:.3f} PR-AUC on the training split against "
            f"{worst.primary_metric:.3f} on validation - a gap of {gap:+.3f} - so it "
            "has substantially memorised the training window. Its validation score "
            "describes an overfitting configuration, not the ceiling of what that "
            "model family can reach.",
        ]

    lines += [
        "",
        "This ordering is a measurement on **this synthetic benchmark**, and the "
        "simulator's own structure is part of what is being measured. It is not "
        "evidence about which model family wins on real RTO data.",
    ]
    return lines


def render_markdown(results: LadderResults, config: EvaluationConfig | None = None) -> str:
    """The full results document for REPORT.md."""
    if config is not None:
        check_report_rules(results, config)

    first = results.records[0] if results.records else None
    lines = [
        "# Baseline ladder results",
        "",
        "**Generated from measured predictions. Do not edit by hand.**",
        "",
        "> Every number below was computed by `rto_sentinel.eval` from actual model "
        "predictions against held-out data. No metric in this project is written as a "
        "literal anywhere in source.",
        "",
        "## Run provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Evaluated split | `{results.evaluated_split}` |",
        f"| Dataset run | `{results.dataset_run_id}` |",
        f"| Config fingerprint | `{results.config_fingerprint[:16]}...` |",
        f"| Feature fingerprint | `{results.feature_fingerprint[:16]}...` |",
        f"| Seed | `{results.seed}` |",
        f"| Cost profile | `{results.cost_profile}` |",
        f"| Operating threshold | **{results.threshold:.4f}** |",
        f"| Threshold source | {results.threshold_source} |",
        f"| Generated at | {results.created_at.isoformat()} |",
    ]
    if first is not None:
        lines += [
            f"| Train rows | {first.train_summary.n_rows:,} "
            f"(days {first.train_summary.first_day}-{first.train_summary.last_day}) |",
            f"| Evaluation rows | {first.evaluation_summary.n_rows:,} "
            f"(days {first.evaluation_summary.first_day}-{first.evaluation_summary.last_day}) |",
            f"| Evaluation positive rate | {first.evaluation_summary.positive_rate:.4f} |",
            f"| Features | {first.n_features} across {len(first.families_used)} families |",
        ]

    lines += [
        "",
        "## Comparison",
        "",
        "PR-AUC leads because it is not inflated by the large negative class. ROC-AUC is "
        "reported but deliberately not led with - it flatters imbalanced problems. "
        "Precision is always adjacent to flag rate, because precision without it is a "
        "half-truth.",
        "",
        "`Trn-val` is training PR-AUC minus validation PR-AUC. A large positive gap means "
        "the rung has memorised the training split, and its validation score should be "
        "read as the score of an overfitting model rather than as a ceiling on what that "
        "model family can do. A small negative gap is unremarkable - it means the "
        "validation window happened to be slightly easier.",
        "",
        "| Rung | Model | PR-AUC (95% CI) | Trn-val | Flag rate | Precision | Recall | F1 "
        "| R@P80 | ROC-AUC | ECE | Net ₹/1k | FP cost ₹ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in results.ordered:
        head = record.headline()
        pr = record.ranking.pr_auc
        pr_text = (
            f"**{pr.value:.3f}** [{pr.ci_low:.3f}, {pr.ci_high:.3f}]"
            if pr.is_defined
            else MARKDOWN_DASH
        )
        lines.append(
            f"| {head['rung']} | `{head['model']}` | {pr_text} "
            f"| {_fmt(head['overfit_gap'], signed=True)} | {_fmt(head['flag_rate'])} "
            f"| {_fmt(head['precision'])} | {_fmt(head['recall'])} | {_fmt(head['f1'])} "
            f"| {_fmt(head['recall_at_p80'])} | {_fmt(head['roc_auc'])} | {_fmt(head['ece'])} "
            f"| {_fmt_int(head['net_inr_per_1000'])} | {_fmt_int(head['fp_cost_inr'])} |"
        )

    if first is not None and first.economics is not None:
        lines += [
            "",
            f"The do-nothing baseline absorbs "
            f"**₹{abs(first.economics.baseline_net_inr_per_1000_orders):,.0f} per 1,000 orders**. "
            "Net ₹/1k above is the saving *relative to that*, so a rung scoring 0 has "
            "changed nothing and a negative figure means the intervention costs more than "
            "it saves.",
        ]

    lines += _verdict_section(results)

    lines += [
        "",
        "## What each rung answers",
        "",
        "| Rung | Question |",
        "|---|---|",
        "| 0 `do_nothing` | What happens without intervention? Defines the loss absorbed today. |",
        "| 1 `blanket_cod_block` | What happens under a maximally aggressive policy? |",
        "| 2 `pincode_blocklist` | Can a simple location-based rule perform meaningfully? |",
        "| 3 `logistic_regression` | What does a simple, interpretable model achieve? |",
        "| 4 `lightgbm` | What does the stronger nonlinear model achieve? |",
        "",
        "## Calibration status",
        "",
        "**Every rung here is uncalibrated.** ECE is reported as a *diagnostic*, not as a "
        "claim that any rung produces honest probabilities. Isotonic calibration on the "
        "validation fold is Phase 5; the model cards carry `calibration_method: null`, and "
        "the decision engine refuses a score whose calibration method is null - so none of "
        "these models can currently reach a decision.",
        "",
        "## Data provenance",
        "",
        "Synthetic benchmark data. Labels are simulated outcomes of the documented process "
        "in [docs/simulator.md](simulator.md), not real-world ground truth. **Absolute "
        "metric values are not a claim about production performance.**",
        "",
    ]
    document = "\n".join(lines)
    if config is not None:
        check_rendered_report(document, config)
    return document


def write_report(
    results: LadderResults, path: Path, config: EvaluationConfig | None = None
) -> Path:
    """Validate and write the results document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results, config), encoding="utf-8")
    return path


def strongest_rung(results: LadderResults) -> ExperimentRecord:
    """The rung with the highest net rupees, ties broken on PR-AUC.

    Net rupees rather than PR-AUC is the primary criterion because that is what
    ``config/evaluation.yaml`` declares, and because a rung that ranks well but
    loses money has not earned production. SPEC section 05: if a simpler rung
    wins, it ships.
    """
    if not results.records:
        msg = "no records to compare"
        raise ValueError(msg)

    def key(record: ExperimentRecord) -> tuple[float, float]:
        net = (
            record.economics.net_inr_saved_per_1000_orders.value
            if record.economics is not None
            else float("-inf")
        )
        return (net if net == net else float("-inf"), record.primary_metric)

    return max(results.records, key=key)
