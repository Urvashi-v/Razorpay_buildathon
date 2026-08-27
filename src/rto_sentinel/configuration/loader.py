"""Load and validate the YAML configuration bundle.

The loader also computes a content hash over every file it reads. That hash is
stamped into run metadata so that a configuration edited *after* results were
produced is detectable rather than deniable - which is the whole point of
writing the split protocol down before modelling (SPEC section 03).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from rto_sentinel.configuration.schemas import (
    AppConfig,
    CostModelConfig,
    EvaluationConfig,
    FeaturesConfig,
    GeneratorConfig,
    LadderConfig,
    PolicyConfig,
    SplitsConfig,
)
from rto_sentinel.settings import Settings, get_settings

T = TypeVar("T")

# Relative paths within the configured config directory.
CONFIG_FILES: dict[str, str] = {
    "generator": "generator.yaml",
    "splits": "splits.yaml",
    "features": "features.yaml",
    "cost_model": "cost_model.yaml",
    "policy": "policy.yaml",
    "ladder": "models/ladder.yaml",
    "evaluation": "evaluation.yaml",
}


class ConfigurationError(RuntimeError):
    """Raised when a configuration file is missing, malformed or out of range."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"configuration file not found: {path}"
        raise ConfigurationError(msg)
    try:
        # safe_load: never construct arbitrary Python objects from a config file.
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"could not parse {path}: {exc}"
        raise ConfigurationError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"{path} must contain a YAML mapping at the top level"
        raise ConfigurationError(msg)
    return loaded


def _parse(model: type[T], payload: dict[str, Any], path: Path) -> T:
    try:
        return model(**payload)
    except ValidationError as exc:
        msg = f"invalid configuration in {path}:\n{exc}"
        raise ConfigurationError(msg) from exc


def config_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).config_path


def load_generator_config(settings: Settings | None = None) -> GeneratorConfig:
    path = config_dir(settings) / CONFIG_FILES["generator"]
    return _parse(GeneratorConfig, _read_yaml(path), path)


def load_splits_config(settings: Settings | None = None) -> SplitsConfig:
    path = config_dir(settings) / CONFIG_FILES["splits"]
    return _parse(SplitsConfig, _read_yaml(path), path)


def load_features_config(settings: Settings | None = None) -> FeaturesConfig:
    path = config_dir(settings) / CONFIG_FILES["features"]
    return _parse(FeaturesConfig, _read_yaml(path), path)


def load_cost_model_config(settings: Settings | None = None) -> CostModelConfig:
    path = config_dir(settings) / CONFIG_FILES["cost_model"]
    return _parse(CostModelConfig, _read_yaml(path), path)


def load_policy_config(settings: Settings | None = None) -> PolicyConfig:
    path = config_dir(settings) / CONFIG_FILES["policy"]
    return _parse(PolicyConfig, _read_yaml(path), path)


def load_ladder_config(settings: Settings | None = None) -> LadderConfig:
    path = config_dir(settings) / CONFIG_FILES["ladder"]
    return _parse(LadderConfig, _read_yaml(path), path)


def load_evaluation_config(settings: Settings | None = None) -> EvaluationConfig:
    path = config_dir(settings) / CONFIG_FILES["evaluation"]
    return _parse(EvaluationConfig, _read_yaml(path), path)


def load_app_config(settings: Settings | None = None) -> AppConfig:
    """Load and validate every configuration file as one bundle."""
    settings = settings or get_settings()
    return AppConfig(
        generator=load_generator_config(settings),
        splits=load_splits_config(settings),
        features=load_features_config(settings),
        cost_model=load_cost_model_config(settings),
        policy=load_policy_config(settings),
        ladder=load_ladder_config(settings),
        evaluation=load_evaluation_config(settings),
    )


def config_fingerprint(settings: Settings | None = None) -> str:
    """SHA-256 over the raw bytes of every configuration file, in fixed order.

    Recorded alongside every dataset, model artefact and evaluation report so a
    result can always be traced back to the exact configuration that produced it.
    """
    base = config_dir(settings)
    digest = hashlib.sha256()
    for key in sorted(CONFIG_FILES):
        path = base / CONFIG_FILES[key]
        if not path.is_file():
            msg = f"cannot fingerprint configuration; missing {path}"
            raise ConfigurationError(msg)
        digest.update(key.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
