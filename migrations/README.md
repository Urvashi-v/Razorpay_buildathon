# Migrations

Alembic migrations for the RTO Sentinel database.

## Creating a migration

```bash
alembic revision --autogenerate -m "add decision holdout flag"
```

**Always read the generated file before committing it.** Autogenerate is a
starting point, not an authority: it does not detect renames (it emits a drop
plus an add, which loses data), and it cannot know whether a new non-nullable
column needs a backfill.

## Applying

```bash
alembic upgrade head      # apply
alembic downgrade -1      # roll back one revision
alembic upgrade head --sql > review.sql   # emit SQL without connecting
```

## Why there is no initial revision committed yet

The schema in `src/rto_sentinel/db/models.py` is settled but has not been
exercised against a live PostgreSQL instance. Generating an initial migration
from an unexercised schema produces a revision that will need editing almost
immediately, and a migration history that starts with a correction is worse than
one that starts a step later. The initial revision lands in Phase 4, alongside
the repository implementations that first write to these tables.

Until then, `Base.metadata.create_all()` is used in tests, which run against
SQLite.

## Where the URL comes from

`alembic.ini` has an **empty** `sqlalchemy.url`. `env.py` reads the real URL from
`rto_sentinel.settings`, the same path the application uses. This keeps the
password out of a committed file and guarantees migrations and the application
target the same database.
