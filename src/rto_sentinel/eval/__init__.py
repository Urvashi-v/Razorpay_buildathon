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
from rto_sentinel.eval.model_card import (
    check_card,
    metrics_rows,
    render_model_card,
    write_comparison_csv,
    write_metrics_csv,
    write_model_card,
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
    "check_card",
    "check_rendered_report",
    "check_report_rules",
    "comparison_table",
    "confusion_at_threshold",
    "cost_sensitivity_curve",
    "do_nothing_net_per_1000",
    "economic_result",
    "expected_calibration_error",
    "metrics_rows",
    "pr_auc",
    "precision_at_k",
    "ranking_metrics",
    "recall_at_precision",
    "render_markdown",
    "render_model_card",
    "roc_auc",
    "strongest_rung",
    "write_comparison_csv",
    "write_metrics_csv",
    "write_model_card",
    "write_report",
]
