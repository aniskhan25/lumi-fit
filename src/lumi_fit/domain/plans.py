"""Immutable serving-plan contract shared by both planning directions."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .units import Duration, MemoryAmount


class ContainerIdentity(ContractModel):
    family: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str | None = Field(default=None, min_length=1)


class RuntimeOptions(ContractModel):
    max_model_length: int = Field(ge=1)
    max_concurrent_sequences: int = Field(ge=1)
    max_batched_tokens: int = Field(ge=1)
    gpu_memory_utilization: Decimal = Field(gt=0, le=1)
    kv_cache_dtype: str = Field(min_length=1)
    enable_prefix_caching: bool = False
    enforce_eager: bool = False


class ReplicaRouting(StrEnum):
    SINGLE = "single"
    ROUND_ROBIN = "round-robin"


class ServingPlan(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str = Field(pattern=r"^.+/.+$")
    model_revision: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)
    container: ContainerIdentity
    runtime_options: RuntimeOptions
    partition: str = Field(min_length=1)
    nodes: int = Field(ge=1)
    gcds_per_node: int = Field(ge=1)
    cpu_cores_per_node: int = Field(ge=1)
    cpu_memory_per_node: MemoryAmount
    tensor_parallel: int = Field(ge=1)
    pipeline_parallel: int = Field(ge=1)
    replicas: int = Field(ge=1)
    request_routing: ReplicaRouting = ReplicaRouting.SINGLE
    ranks_per_node: int = Field(ge=1)
    threads_per_rank: int = Field(ge=1)
    walltime: Duration

    @property
    def allocated_gcds(self) -> int:
        return self.nodes * self.gcds_per_node

    @property
    def gcds_per_replica(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel

    @property
    def used_gcds(self) -> int:
        return self.replicas * self.gcds_per_replica

    @model_validator(mode="after")
    def validate_layout(self) -> ServingPlan:
        if self.used_gcds > self.allocated_gcds:
            raise ValueError("replica and parallelism layout exceeds allocated GCDs")
        if self.ranks_per_node > self.gcds_per_node:
            raise ValueError("ranks per node cannot exceed allocated GCDs per node")
        if self.ranks_per_node * self.nodes != self.used_gcds:
            raise ValueError("total ranks must equal the GCDs used by replicas")
        if self.replicas == 1 and self.request_routing is not ReplicaRouting.SINGLE:
            raise ValueError("a single replica requires single request routing")
        if self.replicas > 1 and self.request_routing is not ReplicaRouting.ROUND_ROBIN:
            raise ValueError("multiple replicas require round-robin request routing")
        return self
