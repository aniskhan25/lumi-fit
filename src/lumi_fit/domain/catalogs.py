"""Versioned catalog contracts for LUMI infrastructure and serving runtimes."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, HttpUrl, model_validator

from .base import ContractModel
from .units import Duration, MemoryAmount


class SourceReference(ContractModel):
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_one_location(self) -> SourceReference:
        if (self.url is None) == (self.path is None):
            raise ValueError("source must provide exactly one of url or path")
        return self


class CatalogMetadata(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    profile_version: str = Field(min_length=1)
    valid_from: date
    valid_until: date | None = None
    last_verified_at: datetime
    sources: tuple[SourceReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self) -> CatalogMetadata:
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("valid_until cannot precede valid_from")
        return self


class GcdProfile(ContractModel):
    accelerator_architecture: str = Field(min_length=1)
    model: str = Field(min_length=1)
    gcds_per_module: int = Field(ge=1)
    modules_per_node: int = Field(ge=1)
    hbm_per_gcd: MemoryAmount

    @property
    def gcds_per_node(self) -> int:
        return self.gcds_per_module * self.modules_per_node


class CpuProfile(ContractModel):
    model: str = Field(min_length=1)
    physical_cores: int = Field(ge=1)
    schedulable_cores: int = Field(ge=1)
    numa_domains: int = Field(ge=1)
    memory_per_node: MemoryAmount

    @model_validator(mode="after")
    def validate_cores(self) -> CpuProfile:
        if self.schedulable_cores > self.physical_cores:
            raise ValueError("schedulable cores cannot exceed physical cores")
        return self


class TopologyLink(ContractModel):
    scope: str = Field(min_length=1)
    peak_bidirectional_bandwidth_gb_s: Decimal = Field(gt=0)
    notes: str = Field(min_length=1)


class NetworkProfile(ContractModel):
    fabric: str = Field(min_length=1)
    endpoints_per_node: int = Field(ge=1)
    peak_bidirectional_bandwidth_per_endpoint_gb_s: Decimal = Field(gt=0)


class HardwareProfile(ContractModel):
    kind: Literal["hardware"] = "hardware"
    metadata: CatalogMetadata
    system_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    node_count: int = Field(ge=1)
    gcd: GcdProfile
    cpu: CpuProfile
    topology_links: tuple[TopologyLink, ...] = ()
    network: NetworkProfile


class AllocationGranularity(StrEnum):
    FULL_NODE = "full-node"
    RESOURCE = "resource"


class PartitionProfile(ContractModel):
    partition_id: str = Field(min_length=1)
    hardware_profile: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    allocation_granularity: AllocationGranularity
    exclusive: bool
    max_nodes: int = Field(ge=1)
    max_walltime: Duration
    max_node_hours: Decimal | None = Field(default=None, gt=0)
    default_enabled: bool = True
    special_access_required: bool = False
    billing_rule: str = Field(min_length=1)
    notes: tuple[str, ...] = ()


class PartitionCatalog(ContractModel):
    kind: Literal["partitions"] = "partitions"
    metadata: CatalogMetadata
    system_id: str = Field(min_length=1)
    partitions: tuple[PartitionProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> PartitionCatalog:
        ids = [partition.partition_id for partition in self.partitions]
        if len(ids) != len(set(ids)):
            raise ValueError("partition IDs must be unique")
        return self


class BillingRuleType(StrEnum):
    FULL_NODE_GPU = "full-node-gpu"
    SUBNODE_GPU = "subnode-gpu"


class BillingRule(ContractModel):
    rule_id: str = Field(min_length=1)
    type: BillingRuleType
    billed_gpu_hours_per_node_hour: Decimal | None = Field(default=None, gt=0)
    billed_gpu_hours_per_gcd_hour: Decimal | None = Field(default=None, gt=0)
    cpu_cores_per_billing_slice: int | None = Field(default=None, ge=1)
    memory_per_billing_slice: MemoryAmount | None = None
    formula: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rule_fields(self) -> BillingRule:
        if self.type is BillingRuleType.FULL_NODE_GPU:
            if self.billed_gpu_hours_per_node_hour is None:
                raise ValueError("full-node rule requires billed_gpu_hours_per_node_hour")
        if self.type is BillingRuleType.SUBNODE_GPU:
            required = (
                self.billed_gpu_hours_per_gcd_hour,
                self.cpu_cores_per_billing_slice,
                self.memory_per_billing_slice,
            )
            if any(value is None for value in required):
                raise ValueError("subnode rule requires GCD, CPU, and memory slice fields")
        return self


class BillingCatalog(ContractModel):
    kind: Literal["billing"] = "billing"
    metadata: CatalogMetadata
    system_id: str = Field(min_length=1)
    rules: tuple[BillingRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> BillingCatalog:
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("billing rule IDs must be unique")
        return self


class SupportLevel(StrEnum):
    PROVEN = "proven"
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class QuantizationSupport(ContractModel):
    name: str = Field(min_length=1)
    support: SupportLevel
    notes: str = Field(min_length=1)


class RuntimeProfile(ContractModel):
    runtime_id: str = Field(min_length=1)
    runtime_name: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    container_family: str = Field(min_length=1)
    container_path_pattern: str = Field(min_length=1)
    verified_container_path: str | None = Field(default=None, min_length=1)
    accelerator_architectures: tuple[str, ...] = Field(min_length=1)
    serving_modes: tuple[str, ...] = Field(min_length=1)
    tensor_parallel: SupportLevel
    pipeline_parallel: SupportLevel
    multi_node: SupportLevel
    quantizations: tuple[QuantizationSupport, ...] = Field(min_length=1)
    notes: tuple[str, ...] = ()


class RuntimeCatalog(ContractModel):
    kind: Literal["runtimes"] = "runtimes"
    metadata: CatalogMetadata
    system_id: str = Field(min_length=1)
    runtimes: tuple[RuntimeProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RuntimeCatalog:
        ids = [runtime.runtime_id for runtime in self.runtimes]
        if len(ids) != len(set(ids)):
            raise ValueError("runtime IDs must be unique")
        return self


class ScoreWeights(ContractModel):
    model_suitability: Decimal = Field(ge=0, le=1)
    service_performance: Decimal = Field(ge=0, le=1)
    memory_safety: Decimal = Field(ge=0, le=1)
    allocation_efficiency: Decimal = Field(ge=0, le=1)
    billing_efficiency: Decimal = Field(ge=0, le=1)
    topology_efficiency: Decimal = Field(ge=0, le=1)
    runtime_compatibility: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> ScoreWeights:
        total = sum(
            (
                self.model_suitability,
                self.service_performance,
                self.memory_safety,
                self.allocation_efficiency,
                self.billing_efficiency,
                self.topology_efficiency,
                self.runtime_compatibility,
            ),
            Decimal(0),
        )
        if total != Decimal(1):
            raise ValueError(f"score weights must sum to 1.0, got {total}")
        return self


class ScoringProfile(ContractModel):
    profile_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weights: ScoreWeights


class ScoringCatalog(ContractModel):
    kind: Literal["scoring"] = "scoring"
    metadata: CatalogMetadata
    profiles: tuple[ScoringProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ScoringCatalog:
        ids = [profile.profile_id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("scoring profile IDs must be unique")
        return self
