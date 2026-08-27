"""Writing and reading the benchmark dataset as local artefacts.

The ML pipeline reads files, not tables. That separation is deliberate: a model
can be retrained on a laptop with no database running, and the database can be
migrated without touching the pipeline. The database is the system of record for
the *application*; the artefact is the input to the *experiment*.

WHAT GETS WRITTEN
-----------------
``orders.parquet``
    The benchmark table. This is what training reads.
``customers.parquet`` / ``addresses.parquet`` / ``delivery_events.parquet``
    The dimensions and the event trail.
``simulation_latents.parquet``
    The simulator's ground truth. Written to a **separate file**, never joined
    into ``orders.parquet``, so that loading the benchmark cannot accidentally
    pull the true probability along with it.
``dataset_run.json``
    Provenance: seed, generator version, configuration snapshot and fingerprint,
    creation timestamp, realised base rates, and the data-provenance statement.

Parquet rather than CSV because a CSV round-trip loses dtypes - and the two it
loses most readily are exactly the two that matter here: timezone-aware
timestamps, and the difference between a NULL label and the string "None".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.data.generator import GenerationResult

ORDERS_FILE = "orders.parquet"
CUSTOMERS_FILE = "customers.parquet"
ADDRESSES_FILE = "addresses.parquet"
EVENTS_FILE = "delivery_events.parquet"
LATENTS_FILE = "simulation_latents.parquet"
METADATA_FILE = "dataset_run.json"


class ArtifactError(RuntimeError):
    """Raised when a dataset artefact is missing or unreadable."""


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """A dataset on disk, loaded back into frames."""

    orders: pd.DataFrame
    customers: pd.DataFrame
    addresses: pd.DataFrame
    delivery_events: pd.DataFrame
    metadata: dict[str, object]
    latents: pd.DataFrame | None = None


def dataset_dir(artifact_root: Path, run_id: str) -> Path:
    """Where one dataset run lives. Keyed by run id, so runs never collide."""
    return artifact_root / "datasets" / run_id


def write_dataset(result: GenerationResult, artifact_root: Path) -> Path:
    """Write a generated dataset to disk and return its directory."""
    target = dataset_dir(artifact_root, result.metadata.run_id)
    target.mkdir(parents=True, exist_ok=True)

    result.orders.to_parquet(target / ORDERS_FILE, index=False)
    result.customers.to_parquet(target / CUSTOMERS_FILE, index=False)
    result.addresses.to_parquet(target / ADDRESSES_FILE, index=False)
    result.delivery_events.to_parquet(target / EVENTS_FILE, index=False)
    # Separate file, never merged into orders.parquet.
    result.latents.to_parquet(target / LATENTS_FILE, index=False)

    (target / METADATA_FILE).write_text(result.metadata.to_json(), encoding="utf-8")
    return target


def read_dataset(directory: Path, *, include_latents: bool = False) -> DatasetArtifact:
    """Load a dataset artefact back from disk.

    ``include_latents`` defaults to False on purpose. The latents file holds the
    true per-order probability, which is perfect leakage if it reaches a model,
    so reading it has to be an explicit request rather than something that
    happens because it was in the folder.
    """
    if not directory.is_dir():
        msg = f"dataset directory not found: {directory}"
        raise ArtifactError(msg)

    required = (ORDERS_FILE, CUSTOMERS_FILE, ADDRESSES_FILE, EVENTS_FILE, METADATA_FILE)
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        msg = f"dataset at {directory} is incomplete; missing {missing}"
        raise ArtifactError(msg)

    latents = (
        pd.read_parquet(directory / LATENTS_FILE)
        if include_latents and (directory / LATENTS_FILE).is_file()
        else None
    )

    return DatasetArtifact(
        orders=pd.read_parquet(directory / ORDERS_FILE),
        customers=pd.read_parquet(directory / CUSTOMERS_FILE),
        addresses=pd.read_parquet(directory / ADDRESSES_FILE),
        delivery_events=pd.read_parquet(directory / EVENTS_FILE),
        metadata=json.loads((directory / METADATA_FILE).read_text(encoding="utf-8")),
        latents=latents,
    )


def latest_dataset_dir(artifact_root: Path) -> Path | None:
    """The most recently written dataset directory, or None if there are none."""
    root = artifact_root / "datasets"
    if not root.is_dir():
        return None
    candidates = [
        child for child in root.iterdir() if child.is_dir() and (child / METADATA_FILE).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / METADATA_FILE).stat().st_mtime)
