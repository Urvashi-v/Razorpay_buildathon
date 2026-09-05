"""Shared rate-limit counters, so a limit survives multiple workers.

The in-process limiter keeps its buckets in memory, so N uvicorn workers permit
N times the configured rate. This table gives them one counter to share. Opt-in
via ``RTO_RATE_LIMIT_BACKEND=database``; a single-worker deployment can leave the
table empty and pay nothing.

Revision ID: 8a3e5c1f9b24
Revises: 4f1c2a7d8e30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "8a3e5c1f9b24"
down_revision: str | None = "4f1c2a7d8e30"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_windows",
        sa.Column("caller", sa.String(length=64), nullable=False),
        # Unix epoch second the window opened, floored to the window size. An
        # integer rather than a timestamp: the limiter does arithmetic on it on
        # every request, and integer division is the whole operation.
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("caller", "window_start"),
        sa.CheckConstraint("hits >= 0", name="ck_rate_limit_hits_non_negative"),
    )
    # Sweeping expired windows is a range scan on window_start alone.
    op.create_index("ix_rate_limit_windows_window_start", "rate_limit_windows", ["window_start"])


def downgrade() -> None:
    op.drop_index("ix_rate_limit_windows_window_start", table_name="rate_limit_windows")
    op.drop_table("rate_limit_windows")
