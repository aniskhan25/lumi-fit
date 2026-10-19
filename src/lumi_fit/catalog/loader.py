"""Safe YAML loaders for versioned catalog contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from lumi_fit.domain.catalogs import (
    BillingCatalog,
    HardwareProfile,
    PartitionCatalog,
    RuntimeCatalog,
    ScoringCatalog,
)
from lumi_fit.domain.models import ModelCatalog, ModelCoverageTargets
from lumi_fit.domain.requests import ModelSelector, ResourceEnvelope, ServingRequirements

ContractT = TypeVar("ContractT", bound=BaseModel)


def _load_yaml(path: str | Path) -> Any:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        raise ValueError(f"{resolved} is empty")
    return value


def load_contract(path: str | Path, model: type[ContractT]) -> ContractT:
    return model.model_validate(_load_yaml(path))


def load_hardware(path: str | Path) -> HardwareProfile:
    return load_contract(path, HardwareProfile)


def load_partitions(path: str | Path) -> PartitionCatalog:
    return load_contract(path, PartitionCatalog)


def load_billing(path: str | Path) -> BillingCatalog:
    return load_contract(path, BillingCatalog)


def load_runtime(path: str | Path) -> RuntimeCatalog:
    return load_contract(path, RuntimeCatalog)


def load_scoring(path: str | Path) -> ScoringCatalog:
    return load_contract(path, ScoringCatalog)


def load_models(path: str | Path) -> ModelCatalog:
    return load_contract(path, ModelCatalog)


def load_model_coverage(path: str | Path) -> ModelCoverageTargets:
    return load_contract(path, ModelCoverageTargets)


def load_resources(path: str | Path) -> ResourceEnvelope:
    return load_contract(path, ResourceEnvelope)


def load_serving(path: str | Path) -> ServingRequirements:
    return load_contract(path, ServingRequirements)


def load_model_selector(path: str | Path) -> ModelSelector:
    return load_contract(path, ModelSelector)
