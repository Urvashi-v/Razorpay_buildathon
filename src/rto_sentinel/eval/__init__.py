"""Evaluation harness: metrics, economics, uncertainty, reporting, plots.

Built before any model exists, so the scoreboard cannot be bent around a result
after the fact. Depends on: configuration, contracts, decision (for the
cost-derived threshold). Never imports models - the harness scores arrays and
stays ignorant of what produced them, which is what lets every rung of the ladder
be judged on identical footing.
"""

from rto_sentinel.eval.bootstrap import bootstrap_metric
from rto_sentinel.eval.economics import (
    cost_sensitivity_curve,
    do_nothing_net_per_1000,
    economic_result,
)
from rto_sentinel.eval.metrics import (
    ConfusionMatrix,
    calibration_metrics,
    confusion_at_threshold,
    expected_calibration_error,
    pr_auc,
    precision_at_k,
    ranking_metrics,
    recall_at_precision,
    roc_auc,
)
from rto_sentinel.eval.report import (
    DishonestReportError,
    check_rendered_report,
    check_report_rules,
    comparison_table,
    render_markdown,
    strongest_rung,
    write_report,
)

__all__ = [
    "ConfusionMatrix",
    "DishonestReportError",
    "bootstrap_metric",
    "calibration_metrics",
    "check_rendered_report",
    "check_report_rules",
    "comparison_table",
    "confusion_at_threshold",
    "cost_sensitivity_curve",
    "do_nothing_net_per_1000",
    "economic_result",
    "expected_calibration_error",
    "pr_auc",
    "precision_at_k",
    "ranking_metrics",
    "recall_at_precision",
    "render_markdown",
    "roc_auc",
    "strongest_rung",
    "write_report",
]
