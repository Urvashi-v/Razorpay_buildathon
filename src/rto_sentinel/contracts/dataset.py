"""Dataset provenance - the serialisable half of the dataset contract.

WHY ONLY THE METADATA LIVES HERE
================================
``contracts/`` is the bottom of the dependency graph: it imports nothing above
itself and nothing heavy. That rule is enforced by
``tests/architecture/test_layering.py``, and it caught an earlier version of this
file importing pandas and ``rto_sentinel.features``.

The test was right. Two different things had been put in one module:

* :class:`DatasetMetadata` is a genuine contract - pure Pydantic, serialisable,
  written into model artefacts and evaluation reports, read by anyone asking
  "what produced this number". It belongs here.
* ``ModelingDataset`` holds pandas frames and a ``FeatureSet``. It is an
  in-process pipeline container, not a wire format, and it now lives in
  :mod:`rto_sentinel.features.dataset` next to the code that builds it.

Splitting them keeps the wire boundary light and puts the heavyweight container
where its dependencies already are.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetMetadata(BaseModel):
    """Everything needed to reproduce a modelling dataset exactly.

    Five versions travel together because a result is only reproducible if all
    five are pinned. A model artefact records this object, so "what produced this
    number" is answerable months later without archaeology.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- what produced the raw data ---------------------------------------
    dataset_run_id: str = Field(description="Deterministic id of the generator run")
    generator_version: str = Field(description="Version of the generative process")
    seed: int = Field(description="Random seed for generation")
    config_fingerprint: str = Field(description="SHA-256 over the YAML configuration bundle")

    # --- what produced the features ---------------------------------------
    feature_version: str = Field(description="Version of the feature engineering code")
    feature_fingerprint: str = Field(description="SHA-256 over the feature declarations")
    families_used: tuple[str, ...] = Field(description="Feature families that were enabled")

    # --- how it was split --------------------------------------------------
    split_strategy: str
    split_pool_shares: dict[str, float]
    split_pool_salt: str
    train_days: tuple[int, int]
    validation_days: tuple[int, int]
    test_days: tuple[int, int]

    # --- when -------------------------------------------------------------
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    data_provenance: str = Field(
        default=(
            "Synthetic benchmark data. Labels are simulated outcomes of the documented "
            "process in docs/simulator.md, not real-world ground truth."
        )
    )

    def summary_lines(self) -> list[str]:
        return [
            f"dataset run       : {self.dataset_run_id}",
            f"generator version : {self.generator_version}",
            f"seed              : {self.seed}",
            f"config fingerprint: {self.config_fingerprint[:16]}...",
            f"feature version   : {self.feature_version}",
            f"feature fingerprint: {self.feature_fingerprint[:16]}...",
            f"families          : {', '.join(self.families_used)}",
            f"split strategy    : {self.split_strategy}",
            f"pool shares       : {self.split_pool_shares}",
            f"built at          : {self.built_at.isoformat()}",
        ]


__all__ = ["DatasetMetadata"]
