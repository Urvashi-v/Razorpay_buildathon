"""scope order, customer and address identifiers to their dataset run

``order_id``, ``customer_hash`` and ``address_fingerprint`` were declared
globally unique. They are not: they identify a row *inside one dataset run*. Two
generator runs both number orders from ``ORD-00000001``, and two runs render the
same address text and hash it to the same fingerprint. The result was a database
that could hold exactly one benchmark dataset - the second ``rto-sentinel
seed-db`` failed with

    duplicate key value violates unique constraint "ix_addresses_address_fingerprint"

even though ``dataset_runs`` and ``DatasetRepository.delete_run`` exist so that
several runs can coexist and be compared.

This replaces each global unique index with:

* a composite unique constraint on ``(dataset_run_id, <identifier>)`` - unique
  within a run;
* a **partial** unique index on the identifier alone, ``WHERE dataset_run_id IS
  NULL`` - so serving-path rows, which belong to no benchmark run, keep global
  uniqueness. A composite constraint alone would not give them that: SQL treats
  NULLs as distinct, so every such row would trivially satisfy it.
* a plain non-unique index on the identifier, since lookups by it stay common.

No foreign key referenced these columns - every relationship points at a
surrogate integer primary key - so nothing downstream changes.

The downgrade restores the global unique indexes and therefore only succeeds on a
database holding at most one dataset run. That is not an oversight: data that
needs the new schema cannot fit the old one, and failing loudly on the index
build is better than dropping rows to make it fit.

Revision ID: 4f1c2a7d8e30
Revises: 27bd09cdc56a
Create Date: 2026-08-29 00:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "4f1c2a7d8e30"
down_revision: str | None = "27bd09cdc56a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: (table, identifier column). The index names match what the initial migration
#: created via ``op.f()``.
_SCOPED: tuple[tuple[str, str], ...] = (
    ("addresses", "address_fingerprint"),
    ("customers", "customer_hash"),
    ("orders", "order_id"),
)


def upgrade() -> None:
    for table, column in _SCOPED:
        op.drop_index(f"ix_{table}_{column}", table_name=table)
        op.create_index(f"ix_{table}_{column}", table, [column], unique=False)
        op.create_unique_constraint(
            f"{table}_{column}_run_unique", table, ["dataset_run_id", column]
        )
        op.create_index(
            f"ix_{table}_{column}_standalone",
            table,
            [column],
            unique=True,
            postgresql_where="dataset_run_id IS NULL",
            sqlite_where="dataset_run_id IS NULL",
        )


def downgrade() -> None:
    for table, column in _SCOPED:
        op.drop_index(f"ix_{table}_{column}_standalone", table_name=table)
        op.drop_constraint(f"{table}_{column}_run_unique", table, type_="unique")
        op.drop_index(f"ix_{table}_{column}", table_name=table)
        op.create_index(f"ix_{table}_{column}", table, [column], unique=True)
