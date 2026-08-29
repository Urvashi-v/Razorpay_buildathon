"""Evaluation plots.

Four figures, each answering a question the comparison table cannot:

``pr_curves.png``
    Precision-recall curves for every rung. Shows *where* along the operating
    range a rung is strong, which a single PR-AUC number hides. The base-rate
    line is drawn because a PR curve without it is unreadable - the floor moves
    with the positive rate.

``reliability.png``
    Predicted probability against observed frequency. This is the figure that
    justifies Phase 5: a rung far from the diagonal ranks fine and lies about
    magnitude, and the entire decision layer depends on magnitude.

``threshold_sweep.png``
    Net rupees against threshold, with the cost-derived operating point marked.
    Shows whether the derived threshold sits near the optimum or merely near
    something reasonable.

``flag_rate_vs_precision.png``
    The operational trade-off an ops team actually negotiates: how much precision
    is bought by flagging less.

WHY MATPLOTLIB AND NOT THE CONSOLE
==================================
These are build-time artefacts for the report, not the merchant-facing UI. The
console reads evaluation JSON and draws its own charts; duplicating that here
would mean two renderers to keep in agreement. These exist so a reviewer reading
the repository sees the same curves without starting a server.

The Agg backend is set explicitly so the harness runs headless - on CI, or over
SSH, where a default interactive backend would fail at import.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from rto_sentinel.eval.economics import economic_result
from rto_sentinel.eval.metrics import confusion_at_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.contracts.economics import PortfolioEconomics, ThresholdSweep
    from rto_sentinel.contracts.experiment import LadderResults
    from rto_sentinel.contracts.final import FinalEvaluation
    from rto_sentinel.decision.simulation import SimulationResult

FIGSIZE = (9.0, 6.0)
DPI = 130


def _rung_scores(scores: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column.removeprefix("score__"): scores[column].to_numpy(dtype="float64")
        for column in scores.columns
        if column.startswith("score__")
    }


def plot_pr_curves(scores: pd.DataFrame, path: Path) -> Path:
    """Precision-recall curves for every rung, with the base-rate floor drawn.

    TWO DELIBERATE DEPARTURES FROM A PLAIN sklearn PLOT
    ---------------------------------------------------
    1. ``precision_recall_curve`` appends a sentinel ``(recall=0, precision=1)``
       point that no threshold achieves. Plotted, it draws every curve up into
       the top-left corner and suggests each rung reaches perfect precision at
       low recall. It is dropped.
    2. A constant predictor has no ranking, so it has no curve. Rung 0 scores
       every order identically; the only operating points it has are "flag
       everything" and "flag nothing", and its precision is the base rate at the
       first and undefined at the second. It is drawn as a flat line at the base
       rate and labelled as having no ranking, rather than as a diagonal that
       would imply a discriminating model.
    """
    y_true = scores["label"].to_numpy(dtype=int)
    base_rate = float(y_true.mean())

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name, y_prob in _rung_scores(scores).items():
        if np.unique(y_prob).size == 1:
            axes.plot(
                [0.0, 1.0],
                [base_rate, base_rate],
                label=f"{name} (constant, no ranking)",
                linewidth=1.8,
                linestyle=":",
            )
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # Drop sklearn's trailing sentinel point, which is not an operating point.
        axes.plot(recall[:-1], precision[:-1], label=name, linewidth=1.8)

    axes.axhline(
        base_rate,
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label=f"base rate ({base_rate:.3f})",
    )
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title("Precision-recall by rung\n(the base rate is the floor, not zero)")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_reliability(results: LadderResults, path: Path) -> Path:
    """Predicted probability against observed frequency, per rung.

    Bin populations are annotated, because a bin holding four orders sitting far
    from the diagonal is noise and a bin holding four hundred is a problem.
    """
    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.0, label="perfect")

    for record in results.ordered:
        bins = record.calibration.reliability_bins
        if not bins:
            continue
        predicted = [entry[0] for entry in bins]
        observed = [entry[1] for entry in bins]
        counts = [entry[2] for entry in bins]
        axes.plot(
            predicted,
            observed,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=f"{record.model_name} (ECE {record.calibration.expected_calibration_error:.3f})",
        )
        for x, y, count in zip(predicted, observed, counts, strict=False):
            if count > 0:
                axes.annotate(
                    str(count),
                    (x, y),
                    fontsize=6,
                    alpha=0.6,
                    xytext=(2, 3),
                    textcoords="offset points",
                )

    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed RTO frequency")
    axes.set_title(
        "Reliability diagram - UNCALIBRATED (Phase 4)\n"
        "Distance from the diagonal is what Phase 5's isotonic step must remove"
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_threshold_sweep(
    scores: pd.DataFrame, cost_inputs: CostInputs, operating_threshold: float, path: Path
) -> Path:
    """Net rupees per 1,000 orders across the threshold range."""
    y_true = scores["label"].to_numpy(dtype=int)
    grid = np.linspace(0.02, 0.98, 49)

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name, y_prob in _rung_scores(scores).items():
        nets = [
            economic_result(
                y_true, y_prob, threshold=float(t), cost_inputs=cost_inputs
            ).net_inr_saved_per_1000_orders.value
            for t in grid
        ]
        axes.plot(grid, nets, label=name, linewidth=1.8)

    axes.axvline(
        operating_threshold,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label=f"cost-derived threshold ({operating_threshold:.3f})",
    )
    axes.axhline(0.0, color="grey", linewidth=0.8)
    axes.set_xlabel("Decision threshold")
    axes.set_ylabel("Net ₹ saved per 1,000 orders (vs doing nothing)")
    axes.set_title(
        "Net rupees across the threshold range\n"
        "The dashed line is derived from merchant economics, not fitted to these labels"
    )
    axes.legend(loc="best", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_flag_rate_vs_precision(scores: pd.DataFrame, path: Path) -> Path:
    """The trade-off an ops team negotiates: precision bought by flagging less."""
    y_true = scores["label"].to_numpy(dtype=int)
    grid = np.linspace(0.02, 0.98, 49)

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for name, y_prob in _rung_scores(scores).items():
        points = [confusion_at_threshold(y_true, y_prob, float(t)) for t in grid]
        usable = [(m.flag_rate, m.precision) for m in points if m.n_flagged > 0]
        if not usable:
            continue
        axes.plot(
            [p[0] for p in usable],
            [p[1] for p in usable],
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=name,
        )

    axes.axhline(
        float(y_true.mean()),
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label="base rate (flag everything)",
    )
    axes.set_xlabel("Flag rate (share of orders receiving friction)")
    axes.set_ylabel("Precision")
    axes.set_title(
        "Precision against flag rate\n"
        "A model that flags 40% of orders is unusable regardless of its precision"
    )
    axes.set_xlim(0.0, 1.0)
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def generate_all(
    results: LadderResults,
    scores: pd.DataFrame,
    cost_inputs: CostInputs,
    output_dir: Path,
) -> list[Path]:
    """Write every figure. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_pr_curves(scores, output_dir / "pr_curves.png"),
        plot_reliability(results, output_dir / "reliability.png"),
        plot_threshold_sweep(
            scores, cost_inputs, results.threshold, output_dir / "threshold_sweep.png"
        ),
        plot_flag_rate_vs_precision(scores, output_dir / "flag_rate_vs_precision.png"),
    ]


# ---------------------------------------------------------------------------
# Phase 5: the final model
# ---------------------------------------------------------------------------


def plot_calibration_before_after(
    evaluation: FinalEvaluation, path: Path, *, title_suffix: str = ""
) -> Path:
    """Reliability diagram for the raw and calibrated scores, on one axis.

    Two curves rather than one, because "the model is calibrated" is a
    comparative claim. The distance each curve sits from the diagonal is what
    the calibration step did and did not fix, and bin populations are annotated
    so a point built from nine orders is not read as a finding.
    """
    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.0, label="perfect")

    series = (
        ("uncalibrated", evaluation.uncalibrated_calibration, "tab:orange", "o"),
        (
            f"calibrated ({evaluation.calibration_method})",
            evaluation.calibration,
            "tab:blue",
            "s",
        ),
    )
    for label, metrics, colour, marker in series:
        bins = metrics.reliability_bins
        if not bins:
            continue
        predicted = [entry[0] for entry in bins]
        observed = [entry[1] for entry in bins]
        axes.plot(
            predicted,
            observed,
            marker=marker,
            color=colour,
            linewidth=1.6,
            label=f"{label} (ECE {metrics.expected_calibration_error:.3f})",
        )
        for x, y, count in bins:
            axes.annotate(
                f"{count:,}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=6
            )

    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Observed RTO frequency")
    axes.set_title(
        f"Reliability - {evaluation.evaluated_split} split{title_suffix}\n"
        "Distance from the diagonal is what calibration had to remove"
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_precision_recall_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: Path, *, split: str
) -> Path:
    """Precision, recall and flag rate against the operating threshold.

    The chart an operations team reads before agreeing to a threshold: what share
    of orders gets frictioned, how many of those are right, and how much of the
    problem is caught. The derived operating point is marked so the trade being
    made is visible rather than asserted.
    """
    grid = np.linspace(0.01, 0.99, 99)
    precision, recall, flag_rate = [], [], []
    for value in grid:
        matrix = confusion_at_threshold(y_true, y_prob, float(value))
        precision.append(matrix.precision)
        recall.append(matrix.recall)
        flag_rate.append(matrix.flag_rate)

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.plot(grid, precision, label="precision", linewidth=1.8)
    axes.plot(grid, recall, label="recall", linewidth=1.8)
    axes.plot(grid, flag_rate, label="flag rate", linewidth=1.8, linestyle="--")
    axes.axhline(
        float(np.mean(y_true)),
        color="grey",
        linestyle=":",
        linewidth=1.0,
        label=f"base rate ({np.mean(y_true):.3f})",
    )
    axes.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"cost-derived threshold ({threshold:.3f})",
    )
    axes.set_xlabel("Decision threshold (calibrated probability)")
    axes.set_ylabel("Rate")
    axes.set_title(
        f"Precision, recall and flag rate - {split} split\n"
        "Precision alone says nothing; the flag rate beside it is what it costs"
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_final_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, path: Path, *, split: str, pr_auc_value: float
) -> Path:
    """The final model's precision-recall curve against the base-rate floor."""
    base_rate = float(np.mean(y_true))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    # Drop sklearn's trailing sentinel point, which no threshold achieves.
    axes.plot(recall[:-1], precision[:-1], linewidth=1.8, label=f"PR-AUC {pr_auc_value:.3f}")
    axes.axhline(
        base_rate,
        linestyle="--",
        color="grey",
        linewidth=1.0,
        label=f"base rate ({base_rate:.3f})",
    )
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(
        f"Precision-recall - final model, {split} split\nThe base rate is the floor, not zero"
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def generate_final_plots(
    evaluation: FinalEvaluation,
    y_true: np.ndarray,
    calibrated: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> list[Path]:
    """Every figure for one final-model evaluation. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split = evaluation.evaluated_split
    return [
        plot_calibration_before_after(evaluation, output_dir / f"reliability__{split}.png"),
        plot_precision_recall_analysis(
            y_true,
            calibrated,
            threshold,
            output_dir / f"precision_recall__{split}.png",
            split=split,
        ),
        plot_final_pr_curve(
            y_true,
            calibrated,
            output_dir / f"pr_curve__{split}.png",
            split=split,
            pr_auc_value=evaluation.ranking.pr_auc.value,
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 6: the economics of the decision policy
# ---------------------------------------------------------------------------


def plot_threshold_economics(sweep: ThresholdSweep, path: Path) -> Path:
    """Net rupees and expected cost across the threshold range.

    The derived operating point and the curve's peak are marked separately and
    deliberately. Showing only the peak would invite the reader to conclude the
    threshold should move there; showing both makes the distinction between "what
    the labels would pick" and "what the merchant's economics pick" visible.
    """
    thresholds = [point.threshold for point in sweep.points]
    expected = [point.expected_net_inr_per_1000_orders for point in sweep.points]
    realized = [
        point.realized_net_inr_per_1000_orders
        for point in sweep.points
        if point.realized_net_inr_per_1000_orders is not None
    ]

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.plot(thresholds, expected, linewidth=1.8, label="expected (from probabilities)")
    if len(realized) == len(thresholds):
        axes.plot(
            thresholds, realized, linewidth=1.8, linestyle="--", label="realized (from labels)"
        )
    axes.axhline(0.0, color="grey", linewidth=1.0)
    axes.axvline(
        sweep.derived_threshold,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"derived operating point ({sweep.derived_threshold:.3f})",
    )
    axes.axvline(
        sweep.best_net_threshold,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label=f"curve peak ({sweep.best_net_threshold:.3f}) - NOT selected",
    )
    axes.set_xlabel("Decision threshold (calibrated probability)")
    axes.set_ylabel("Net INR saved per 1,000 orders")
    axes.set_title(
        "Net rupees across the threshold range\n"
        "The operating point is derived from economics, not read off this curve"
    )
    axes.set_xlim(0.0, 1.0)
    axes.legend(loc="upper right", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_band_economics(economics: PortfolioEconomics, path: Path) -> Path:
    """Savings, false-positive cost and net, per rung of the ladder.

    Volume is annotated on each bar because a rung's rupee contribution is mostly
    a story about how many orders land in it.
    """
    bands = [band for band in economics.bands if band.applies_friction]
    if not bands:
        bands = list(economics.bands)

    labels = [band.band.value for band in bands]
    positions = np.arange(len(labels))
    width = 0.27

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.bar(
        positions - width,
        [band.expected_saving_inr for band in bands],
        width,
        label="expected saving",
        color="tab:green",
    )
    axes.bar(
        positions,
        [-band.expected_false_positive_cost_inr for band in bands],
        width,
        label="false-positive cost",
        color="tab:red",
    )
    axes.bar(
        positions + width,
        [band.expected_net_inr for band in bands],
        width,
        label="net",
        color="tab:blue",
    )
    for index, band in enumerate(bands):
        axes.annotate(
            f"{band.n_orders:,} orders\n{band.share_of_book:.1%}",
            (index, 0),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )

    axes.axhline(0.0, color="black", linewidth=1.0)
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [f"{label}\n{band.action.value}" for label, band in zip(labels, bands, strict=True)]
    )
    axes.set_ylabel("INR")
    axes.set_title(
        "Where the money comes from, rung by rung\n"
        "Every saving here rests on an ASSUMED intervention success rate"
    )
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_margin_response(sweep: list[tuple[float, SimulationResult]], path: Path) -> Path:
    """Threshold and flag rate against contribution margin, one input at a time.

    A CONTROLLED sweep, deliberately. An earlier version of this figure plotted
    the three configured cost profiles against each other, which looked like the
    same chart and was not: those profiles differ in RTO cost and abandonment as
    well as margin, so the line through them confounded three effects and did not
    come out monotone. Here every point shares one set of economics with the
    margin alone varied, which is the only way the curve can be read as showing
    what the title claims.
    """
    margins = [margin for margin, _ in sweep]
    thresholds = [result.threshold.threshold for _, result in sweep]
    flag_rates = [result.economics.flag_rate for _, result in sweep]

    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axes.plot(margins, thresholds, marker="o", linewidth=1.8, label="derived threshold")
    axes.plot(margins, flag_rates, marker="s", linewidth=1.8, linestyle="--", label="flag rate")
    axes.set_xlabel("Contribution margin (INR), all other inputs held fixed")
    axes.set_ylabel("Probability")
    axes.set_title(
        "The threshold is not a constant\n"
        "A higher margin makes a lost order costlier, so the bar rises"
    )
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="upper left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
