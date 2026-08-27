"""Evaluation harness: metrics, economics, uncertainty, fairness, reporting.

Built before any model exists, so the scoreboard cannot be bent around a result
after the fact. Depends on: configuration, contracts. Never imports models - the
harness scores arrays and stays ignorant of what produced them, which is what
lets every rung of the ladder be judged on identical footing.
"""

from rto_sentinel.eval.ablation import AblationResult, run_ablation
from rto_sentinel.eval.bootstrap import bootstrap_metric
from rto_sentinel.eval.economics import cost_sensitivity_curve, economic_result
from rto_sentinel.eval.fairness import cohort_breakdown, fairness_audit
from rto_sentinel.eval.metrics import calibration_metrics, ranking_metrics, recall_at_precision
from rto_sentinel.eval.report import DishonestReportError, render_markdown, write_report

__all__ = [
    "AblationResult",
    "DishonestReportError",
    "bootstrap_metric",
    "calibration_metrics",
    "cohort_breakdown",
    "cost_sensitivity_curve",
    "economic_result",
    "fairness_audit",
    "ranking_metrics",
    "recall_at_precision",
    "render_markdown",
    "run_ablation",
    "write_report",
]
