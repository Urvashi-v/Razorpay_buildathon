"""The economic evaluation report: what the policy costs, and what rests on what.

WHY THIS DOCUMENT IS GENERATED
==============================
Same rule as the ladder results and the model card: every number here is
computed at render time from the scored book and the configuration. There is no
code path that writes a rupee figure from a literal, so the document cannot drift
away from the run that produced it.

THE PROVENANCE TABLE IS THE FIRST SECTION FOR A REASON
======================================================
An economic report is where a measured metric and an unmeasured assumption most
easily get mistaken for each other. "₹4,648 saved per 1,000 orders" reads like a
result; it is arithmetic over a measured confusion matrix and two rates nobody
has ever checked. So the report opens by separating the five kinds of number it
contains, and every rupee total downstream carries the qualifier.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from rto_sentinel.contracts.provenance import ASSUMPTION_PROVENANCES
from rto_sentinel.decision.threshold import derive_threshold, threshold_sensitivity
from rto_sentinel.eval.report import MARKDOWN_DASH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from rto_sentinel.configuration.schemas import CostModelConfig, PolicyConfig
    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.contracts.economics import PortfolioEconomics, ThresholdSweep
    from rto_sentinel.decision.simulation import PolicyComparison, SimulationResult

#: Rows of the sweep to print. The full grid goes to CSV; a hundred-row markdown
#: table is not a document anyone reads.
_SWEEP_STRIDE = 10


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or value != value:
        return MARKDOWN_DASH
    return f"{value:.{digits}f}"


def _count(value: int | None) -> str:
    """An integer, or a dash when the labels needed to compute it do not exist."""
    return MARKDOWN_DASH if value is None else f"{value:,}"


def _money(value: float | None) -> str:
    if value is None or value != value:
        return MARKDOWN_DASH
    return f"{value:,.0f}"


def write_sweep_csv(sweep: ThresholdSweep, path: Path) -> Path:
    """The full sweep, every row, for anyone who wants to plot it themselves."""
    from rto_sentinel.decision.threshold_analysis import sweep_to_rows

    rows = sweep_to_rows(sweep)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return path


def write_band_csv(economics: PortfolioEconomics, path: Path) -> Path:
    """Per-band volumes and rupee outcomes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "band",
                "action",
                "lower_bound",
                "upper_bound",
                "n_orders",
                "share_of_book",
                "expected_rto_orders",
                "realized_rto_orders",
                "assumed_intervention_success",
                "assumed_abandonment",
                "expected_saving_inr",
                "expected_false_positive_cost_inr",
                "expected_net_inr",
            ]
        )
        for band in economics.bands:
            writer.writerow(
                [
                    band.band.value,
                    band.action.value,
                    band.lower_bound,
                    "" if band.upper_bound is None else band.upper_bound,
                    band.n_orders,
                    band.share_of_book,
                    band.expected_rto_orders,
                    "" if band.realized_rto_orders is None else band.realized_rto_orders,
                    band.intervention_success_rate,
                    band.abandonment_rate,
                    band.expected_saving_inr,
                    band.expected_false_positive_cost_inr,
                    band.expected_net_inr,
                ]
            )
    return path


def _provenance_section(economics: PortfolioEconomics) -> list[str]:
    lines = [
        "## Where every number comes from",
        "",
        "Five kinds of number appear below and they are not equally established. "
        "This table separates them; every rupee total further down inherits the "
        "weakest provenance among its inputs.",
        "",
        "| Quantity | Value | Kind | Source |",
        "|---|---|---|---|",
    ]
    for quantity in economics.quantities:
        marker = " **[ASSUMPTION]**" if quantity.is_assumption else ""
        lines.append(
            f"| `{quantity.name}` | {quantity.value:,.4g} {quantity.unit} "
            f"| {quantity.label}{marker} | {quantity.source} |"
        )

    assumptions = [q for q in economics.quantities if q.provenance in ASSUMPTION_PROVENANCES]
    if assumptions:
        names = ", ".join(f"`{q.name}`" for q in assumptions)
        lines += [
            "",
            f"**{names} have never been measured.** They cannot be measured without "
            "running the interventions and observing what would otherwise have "
            "happened, which is what the randomised control slice in "
            "`config/policy.yaml` exists to make possible. Until that has run, every "
            "rupee figure in this document is arithmetic over an unverified rate, and "
            "the sensitivity section below is the closest thing to a bound on how "
            "wrong it could be.",
        ]
    return lines


def _ladder_section(economics: PortfolioEconomics) -> list[str]:
    lines = [
        "",
        "## The intervention ladder at this threshold",
        "",
        "Band cut points are multipliers on the derived threshold, so the whole "
        "ladder moves when the merchant's economics move. No absolute probability "
        "is configured anywhere.",
        "",
        "| Band | Action | Range | Orders | Share | Expected RTOs | Observed RTOs "
        "| Assumed success | Expected net INR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for band in economics.bands:
        upper = "1.0" if band.upper_bound is None else f"{band.upper_bound:.4f}"
        lines.append(
            f"| {band.band.value} | `{band.action.value}` "
            f"| [{band.lower_bound:.4f}, {upper}) | {band.n_orders:,} "
            f"| {band.share_of_book:.1%} | {band.expected_rto_orders:.1f} "
            f"| {_count(band.realized_rto_orders)} "
            f"| {band.intervention_success_rate:.3f} | {_money(band.expected_net_inr)} |"
        )

    empty = [band for band in economics.bands if band.n_orders == 0]
    if empty:
        names = ", ".join(band.band.value for band in empty)
        lines += [
            "",
            f"**{names} received no orders.** The tier exists in configuration and "
            "cannot fire at this threshold on this book - no order scores high enough. "
            "Reported rather than hidden: a ladder rung that never fires is a rung the "
            "merchant is not actually using.",
        ]
    if economics.collapsed_bands:
        lines += [
            "",
            "Bands that cannot exist at this threshold at all:",
            "",
            *[f"- {entry}" for entry in economics.collapsed_bands],
        ]
    return lines


def _economics_section(economics: PortfolioEconomics) -> list[str]:
    lines = [
        "",
        "## Economic outcome",
        "",
        "**Expected** figures are computed from calibrated probabilities alone and "
        "need no labels - that is what a merchant can compute on today's unlabelled "
        "book. **Realized** figures are measured against observed outcomes. The gap "
        "between them is a calibration check.",
        "",
        "| Quantity | Expected | Realized |",
        "|---|---|---|",
        f"| Orders scored | {economics.n_orders:,} | {economics.n_orders:,} |",
        f"| Flag rate (at or above threshold) | {economics.flag_rate:.4f} | "
        f"{economics.flag_rate:.4f} |",
        f"| Intervention rate (any action) | {economics.intervention_rate:.4f} | "
        f"{economics.intervention_rate:.4f} |",
        f"| Orders affected | {economics.expected_orders_affected:,} | "
        f"{economics.expected_orders_affected:,} |",
        f"| Savings, INR | {_money(economics.expected_savings_inr)} | {MARKDOWN_DASH} |",
        f"| False-positive cost, INR | {_money(economics.expected_false_positive_cost_inr)} "
        f"| {MARKDOWN_DASH} |",
        f"| Residual false-negative loss, INR | "
        f"{_money(economics.expected_false_negative_loss_inr)} | {MARKDOWN_DASH} |",
        f"| Total cost, INR | {_money(economics.expected_total_cost_inr)} | {MARKDOWN_DASH} |",
        f"| **Net INR per 1,000 orders** | "
        f"**{_money(economics.expected_net_inr_per_1000_orders)}** | "
        f"**{_money(economics.realized_net_inr_per_1000_orders)}** |",
        f"| True positives | {economics.expected_true_positives:.1f} | "
        f"{_count(economics.realized_true_positives)} |",
        f"| Precision | {MARKDOWN_DASH} | {_fmt(economics.realized_precision)} |",
        f"| Recall | {MARKDOWN_DASH} | {_fmt(economics.realized_recall)} |",
        "",
        f"The do-nothing baseline absorbs "
        f"**INR {abs(economics.do_nothing_loss_inr_per_1000_orders):,.0f} per 1,000 orders**. "
        "Net is the saving relative to that, so zero means the policy changed nothing "
        "and a negative figure means the friction costs more than it saves.",
    ]

    gap = economics.calibration_gap
    if gap is not None:
        lines += [
            "",
            f"**Calibration check.** The probabilities predicted "
            f"{economics.expected_true_positives:.1f} true positives among frictioned "
            f"orders; {economics.realized_true_positives} occurred, a gap of {gap:+.1f}. "
            + (
                "That is small relative to the volume, so the expected figures can be "
                "read as meaning roughly what they say."
                if abs(gap) < 0.1 * max(economics.expected_true_positives, 1.0)
                else "That is large enough to matter: the expected figures inherit the "
                "error, and the direction of the gap is the direction they are wrong in."
            ),
        ]

    if economics.support_cost_on_true_positives_inr > 0:
        lines += [
            "",
            f"**A known understatement.** The specification folds the per-friction support "
            f"cost into `C_fp`, which charges it only to false positives. Ops pays it on "
            f"every frictioned order, so a further "
            f"**INR {economics.support_cost_on_true_positives_inr:,.0f}** of support spend "
            f"on true positives is not reflected in the net figure above. The spec's "
            f"formula is kept rather than silently improved; the omission is reported "
            f"instead.",
        ]

    if economics.holdout_fraction_of_flagged > 0:
        lines += [
            "",
            f"**After the control slice.** {economics.holdout_fraction_of_flagged:.1%} of "
            f"flagged orders receive no friction so that precision stays measurable once "
            f"the system acts. Net after withholding that slice: "
            f"**INR {_money(economics.net_inr_per_1000_after_holdout)} per 1,000 orders**. "
            f"The difference is the price of continuing to know whether the model works.",
        ]
    return lines


def _sweep_section(sweep: ThresholdSweep) -> list[str]:
    lines = [
        "",
        "## Threshold analysis",
        "",
        f"> **{sweep.selection_methodology}**",
        "",
        f"Derived operating point: **{sweep.derived_threshold:.4f}**. "
        f"The curve peaks at **{sweep.best_net_threshold:.4f}**.",
        "",
        "| Threshold | Flag rate | Precision | Recall | F1 | Expected cost INR "
        "| Expected net INR/1k | Realized net INR/1k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    shown = [
        point
        for index, point in enumerate(sweep.points)
        if index % _SWEEP_STRIDE == 0 or point.is_derived_operating_point
    ]
    for point in shown:
        marker = " **<-- operating point**" if point.is_derived_operating_point else ""
        lines.append(
            f"| {point.threshold:.4f}{marker} | {point.flag_rate:.3f} "
            f"| {_fmt(point.precision)} | {_fmt(point.recall)} | {_fmt(point.f1)} "
            f"| {_money(point.expected_cost_inr)} "
            f"| {_money(point.expected_net_inr_per_1000_orders)} "
            f"| {_money(point.realized_net_inr_per_1000_orders)} |"
        )

    distance = abs(sweep.best_net_threshold - sweep.derived_threshold)
    lines += [
        "",
        f"The derived point sits {distance:.4f} from the peak. "
        + (
            "That is close, which is reassuring but not a validation: the two agreeing "
            "means the merchant's stated economics happen to match what the labels "
            "imply, and the derivation would still be the right choice if they did not."
            if distance < 0.05
            else "That gap is worth investigating - either the cost inputs do not match "
            "reality, or the probabilities are miscalibrated in that region. It is not "
            "by itself a reason to move the threshold: doing so would mean the "
            "merchant's economics no longer justify the operating point."
        ),
    ]
    return lines


def _sensitivity_section(cost_inputs: CostInputs, parameters: list[str]) -> list[str]:
    lines = [
        "",
        "## Sensitivity: how wrong can the assumptions be?",
        "",
        "The threshold is a function of four inputs, two of which are assumptions. "
        "This is how far it moves when each is wrong by up to 30%.",
        "",
        "| Parameter | -30% | -15% | baseline | +15% | +30% |",
        "|---|---|---|---|---|---|",
    ]
    perturbations = [-0.30, -0.15, 0.0, 0.15, 0.30]
    for parameter in parameters:
        results = threshold_sensitivity(
            cost_inputs, parameter=parameter, perturbations=perturbations
        )
        cells = " | ".join(f"{derivation.threshold:.4f}" for _, derivation in results)
        lines.append(f"| `{parameter}` | {cells} |")

    lines += [
        "",
        "A threshold that swings widely under a 30% error in an assumed rate is a "
        "threshold resting on that assumption. The two intervention rates are exactly "
        "the ones nobody has measured.",
    ]
    return lines


def _comparison_section(comparison: PolicyComparison) -> list[str]:
    lines = [
        "",
        "## Is the graduated ladder worth having?",
        "",
        "The ladder applies a gentler action to the lower flagged band and a stronger "
        "one above. Whether graduation pays is a question about the assumed "
        "multipliers, so it is answered here rather than assumed.",
        "",
        "| Policy | Net INR per 1,000 orders |",
        "|---|---|",
        f"| Graduated ladder (configured) | {_money(comparison.graduated_net_inr_per_1000)} |",
    ]
    for action, value in sorted(
        comparison.uniform_net_inr_per_1000.items(), key=lambda item: -item[1]
    ):
        lines.append(f"| Uniform `{action}` above threshold | {_money(value)} |")

    best = comparison.uniform_net_inr_per_1000.get(comparison.best_uniform_action)
    if comparison.graduated_wins:
        verdict = (
            "The graduated ladder wins under these assumptions, which is the outcome "
            "the design intends."
        )
    else:
        margin = (best or 0.0) - comparison.graduated_net_inr_per_1000
        verdict = (
            f"**The graduated ladder loses.** Applying `{comparison.best_uniform_action}` "
            f"uniformly to everything above the threshold is worth "
            f"INR {margin:,.0f} per 1,000 orders more. The mechanism is visible in the "
            f"ladder table: the gentlest rung carries most of the flagged volume and is "
            f"assumed to convert least often, so graduating downwards gives up more "
            f"saving than it avoids in abandonment. This is a finding about the assumed "
            f"multipliers, not about reality - but under the assumptions the system "
            f"actually ships with, the simpler policy is better, and that is worth "
            f"saying rather than burying."
        )
    lines += ["", verdict, "", comparison.note]
    return lines


def render_economics_report(
    *,
    economics: PortfolioEconomics,
    sweep: ThresholdSweep,
    comparison: PolicyComparison,
    cost_inputs: CostInputs,
    cost_config: CostModelConfig,
    policy: PolicyConfig,
    simulations: list[tuple[str, SimulationResult]],
) -> str:
    """The full economic evaluation document."""
    derivation = derive_threshold(cost_inputs)
    lines: list[str] = [
        "# Economic evaluation",
        "",
        "**Generated from the scored book and the configuration. Do not edit by hand.**",
        "",
        "> Every rupee figure below is arithmetic over declared inputs applied to "
        "measured model output. The model output was measured on **synthetic** labels, "
        "and two of the inputs are **assumptions nobody has measured**. Neither fact is "
        "a reason not to compute these numbers; both are reasons to read them with the "
        "provenance table in hand.",
        "",
        "## The decision rule",
        "",
        "```",
        "C_fp      = abandonment_on_friction x contribution_margin + friction_support_cost",
        "S_tp      = intervention_success_rate x rto_cost",
        "threshold = C_fp / (C_fp + S_tp)",
        "```",
        "",
        f"At the `{economics.cost_profile}` profile:",
        "",
        "```",
        f"C_fp      = {cost_inputs.abandonment_on_friction} x "
        f"{cost_inputs.contribution_margin_inr} + {cost_inputs.friction_support_cost_inr} "
        f"= {derivation.cost_false_positive_inr:.2f}",
        f"S_tp      = {cost_inputs.intervention_success_rate} x {cost_inputs.rto_cost_inr} "
        f"= {derivation.saving_true_positive_inr:.2f}",
        f"threshold = {derivation.cost_false_positive_inr:.2f} / "
        f"({derivation.cost_false_positive_inr:.2f} + "
        f"{derivation.saving_true_positive_inr:.2f}) = {derivation.threshold:.4f}",
        "```",
        "",
        "**Not 0.5.** The threshold is a function of the merchant's economics and moves "
        "when they move; the simulation section below shows it moving. No absolute "
        "probability is hardcoded anywhere in the policy.",
        "",
        f"Scored book: `{economics.split}` split, {economics.n_orders:,} orders, "
        f"engine version `{economics.engine_version}`.",
        "",
    ]

    lines += _provenance_section(economics)
    lines += _ladder_section(economics)
    lines += _economics_section(economics)
    lines += _sweep_section(sweep)
    lines += _sensitivity_section(cost_inputs, list(cost_config.sensitivity.parameters))
    lines += _comparison_section(comparison)

    lines += [
        "",
        "## Merchant simulation",
        "",
        "Each row below was produced by the same server-side recomputation the API "
        "exposes at `POST /v1/economics/simulate`: new economics in, new threshold, new "
        "band boundaries, new per-order assignment, new rupee totals out. Nothing is "
        "scaled from a cached result and nothing is computed in a browser.",
        "",
        "| Scenario | Margin INR | RTO cost INR | Threshold | Flag rate | Orders affected "
        "| Net INR/1k | Rungs that fire |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, result in simulations:
        inputs = result.threshold.inputs
        rungs = ", ".join(rung.band for rung in result.ladder if rung.action != "none")
        lines.append(
            f"| {label} | {inputs.contribution_margin_inr:,.0f} | {inputs.rto_cost_inr:,.0f} "
            f"| {result.threshold.threshold:.4f} | {result.economics.flag_rate:.3f} "
            f"| {result.economics.expected_orders_affected:,} "
            f"| {_money(result.economics.expected_net_inr_per_1000_orders)} "
            f"| {rungs or MARKDOWN_DASH} |"
        )

    lines += [
        "",
        "A higher margin raises the threshold - losing a good customer costs more, so "
        "the bar for frictioning one rises - and the flag rate falls with it. That "
        "direction is a property of the formula, asserted in "
        "`tests/unit/test_decision_engine.py`, not an artefact of this particular book.",
        "",
        "## Safeguards",
        "",
        f"- No hard block exists at any rung: `hard_block_allowed` is "
        f"`{policy.safeguards.hard_block_allowed}` and `Decision` refuses "
        f"`appeal_available=False` at construction.",
        "- SEVERE routes to a human review queue and carries an appeal path.",
        f"- Ops overrides are enabled ({policy.safeguards.ops_override_enabled}) and "
        f"logged ({policy.safeguards.ops_override_logged}); overrides are counterfactual "
        f"evidence, not noise.",
        f"- A randomised {policy.holdout_control.fraction_of_flagged:.0%} of flagged orders "
        f"receives no friction, so precision remains measurable after the system starts "
        f"acting. Those orders are exempt from review as well - routing them to a human "
        f"would destroy the counterfactual they exist to preserve.",
        "",
        "## What this document does not establish",
        "",
        "- **That the interventions work.** Their effectiveness is assumed. The rupee "
        "figures scale linearly with those assumptions and would move accordingly.",
        "- **That these numbers transfer to a real merchant.** The labels are simulated; "
        "the model was measured on a synthetic benchmark.",
        "- **That the flagged population is fair.** The cohort audit defined in "
        "`config/evaluation.yaml` has not been run.",
        "",
    ]
    return "\n".join(lines)


def write_economics_report(path: Path, document: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
