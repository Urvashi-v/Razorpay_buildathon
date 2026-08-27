"""Alembic environment.

The database URL is read from :mod:`rto_sentinel.settings` rather than from
``alembic.ini``. Two reasons:

* **No password in a committed file.** ``alembic.ini`` ships with an empty URL.
* **Migrations and the application cannot diverge.** Both resolve the URL
  through the same settings object, so ``alembic upgrade head`` always targets
  the database the API will actually use.

``target_metadata`` points at the declarative base, which is what makes
``alembic revision --autogenerate`` produce a real diff against the models.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from rto_sentinel.db.base import Base
from rto_sentinel.db.models import (  # noqa: F401  (imported so Alembic sees every table)
    Decision,
    ModelRun,
    OpsOverrideRecord,
    Order,
    OrderOutcomeRecord,
)
from rto_sentinel.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the URL through application settings, never from alembic.ini."""
    return get_settings().database.url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting - useful for review before applying."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply migrations."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detect type and default changes as well as added/dropped columns:
            # a silently widened column is exactly the kind of drift that turns
            # into a production incident.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
