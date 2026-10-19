"""Cross-catalog validation that cannot be expressed inside one YAML file."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from lumi_fit.domain.catalogs import (
    AllocationGranularity,
    BillingCatalog,
    HardwareProfile,
    PartitionCatalog,
    RuntimeCatalog,
)
from lumi_fit.domain.models import ModelCatalog, ModelCoverageTargets

from .coverage import build_model_coverage_report
from .loader import (
    load_billing,
    load_hardware,
    load_model_coverage,
    load_models,
    load_partitions,
    load_runtime,
)


@dataclass(frozen=True)
class CatalogSet:
    hardware: HardwareProfile
    partitions: PartitionCatalog
    billing: BillingCatalog
    runtimes: RuntimeCatalog
    models: ModelCatalog
    model_coverage: ModelCoverageTargets


def load_default_catalogs(repository_root: str | Path) -> CatalogSet:
    root = Path(repository_root)
    data_root = root / "data"
    if not (data_root / "systems/lumi-g.yaml").is_file():
        installed = Path(sys.prefix) / "share/lumi-fit/data"
        if not (installed / "systems/lumi-g.yaml").is_file():
            raise ValueError(f"catalog data was not found under {data_root} or {installed}")
        data_root = installed
    return CatalogSet(
        hardware=load_hardware(data_root / "systems/lumi-g.yaml"),
        partitions=load_partitions(data_root / "partitions/lumi.yaml"),
        billing=load_billing(data_root / "billing/lumi.yaml"),
        runtimes=load_runtime(data_root / "runtimes/vllm-lumi.yaml"),
        models=load_models(data_root / "models/initial.yaml"),
        model_coverage=load_model_coverage(data_root / "models/coverage.yaml"),
    )


def validate_catalog_references(catalogs: CatalogSet) -> tuple[str, ...]:
    errors: list[str] = []
    system_id = catalogs.hardware.system_id

    for name, referenced_system in (
        ("partitions", catalogs.partitions.system_id),
        ("billing", catalogs.billing.system_id),
        ("runtimes", catalogs.runtimes.system_id),
    ):
        if referenced_system != system_id:
            errors.append(f"{name} references system {referenced_system!r}, expected {system_id!r}")

    billing_ids = {rule.rule_id for rule in catalogs.billing.rules}
    for partition in catalogs.partitions.partitions:
        if partition.hardware_profile != system_id:
            errors.append(
                f"partition {partition.partition_id!r} references unknown hardware "
                f"{partition.hardware_profile!r}"
            )
        if partition.billing_rule not in billing_ids:
            errors.append(
                f"partition {partition.partition_id!r} references unknown billing rule "
                f"{partition.billing_rule!r}"
            )
        if partition.max_nodes > catalogs.hardware.node_count:
            errors.append(
                f"partition {partition.partition_id!r} allows more nodes than the "
                "hardware profile contains"
            )
        if (
            partition.allocation_granularity is AllocationGranularity.FULL_NODE
            and not partition.exclusive
        ):
            errors.append(f"full-node partition {partition.partition_id!r} must be exclusive")

    architecture = catalogs.hardware.gcd.accelerator_architecture
    runtime_ids = {runtime.runtime_id for runtime in catalogs.runtimes.runtimes}
    for runtime in catalogs.runtimes.runtimes:
        if architecture not in runtime.accelerator_architectures:
            errors.append(
                f"runtime {runtime.runtime_id!r} does not support architecture {architecture!r}"
            )

    for artifact in catalogs.models.artifacts:
        for runtime_id in artifact.compatible_runtime_ids:
            if runtime_id not in runtime_ids:
                errors.append(
                    f"artifact {artifact.artifact_id!r} references unknown runtime {runtime_id!r}"
                )

    coverage = build_model_coverage_report(catalogs.models, catalogs.model_coverage)
    errors.extend(f"model coverage: {target}" for target in coverage.unmet_targets)

    return tuple(errors)


def assert_valid_catalogs(catalogs: CatalogSet) -> None:
    errors = validate_catalog_references(catalogs)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"catalog validation failed:\n{formatted}")
