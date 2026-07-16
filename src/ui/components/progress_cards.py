"""Progress metric rendering helpers."""

from __future__ import annotations

from src.models.study_session import ProgressSnapshot
from src.ui.layout import Metric, metric_row


def render_progress_cards(
    snapshot: ProgressSnapshot,
    *,
    include_due_reviews: bool = True,
) -> None:
    """Show meaningful progress signals without zero-value filler cards."""

    metrics: list[Metric] = []
    if snapshot.total_questions_answered > 0:
        metrics.extend(
            [
                ("Questions answered", snapshot.total_questions_answered, None),
                ("Accuracy", f"{snapshot.overall_accuracy:.0%}", "Correct answers across all attempts."),
            ]
        )
        if snapshot.average_confidence is not None:
            metrics.append(
                ("Average confidence", f"{snapshot.average_confidence:.1f} / 5", "Self-reported confidence."),
            )
    if include_due_reviews and snapshot.review_due_count > 0:
        metrics.append(("Due reviews", snapshot.review_due_count, "Questions ready for scheduled review."))
    metric_row(metrics)
