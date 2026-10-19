"""Model and serveable-artifact catalog contracts."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .catalogs import CatalogMetadata


class ModelCapability(StrEnum):
    TEXT_GENERATION = "text-generation"
    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    TOOL_USE = "tool-use"


class ModelEvaluation(ContractModel):
    evaluation_id: str = Field(min_length=1)
    use_case: str = Field(min_length=1)
    score: Decimal
    score_minimum: Decimal
    score_maximum: Decimal
    source_url: str = Field(min_length=1)
    evaluated_at: date | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_score(self) -> ModelEvaluation:
        if self.score_maximum <= self.score_minimum:
            raise ValueError("evaluation score maximum must exceed minimum")
        if not self.score_minimum <= self.score <= self.score_maximum:
            raise ValueError("evaluation score must lie within its declared range")
        return self


class TransformerArchitecture(ContractModel):
    architecture_name: str = Field(min_length=1)
    total_parameters: int = Field(ge=1)
    active_parameters: int | None = Field(default=None, ge=1)
    layers: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    intermediate_size: int = Field(ge=1)
    attention_heads: int = Field(ge=1)
    kv_heads: int = Field(ge=1)
    head_dimension: int = Field(ge=1)
    vocabulary_size: int = Field(ge=1)
    context_length: int = Field(ge=1)
    tie_word_embeddings: bool
    moe_experts: int | None = Field(default=None, ge=1)
    active_experts_per_token: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_architecture(self) -> TransformerArchitecture:
        if self.active_parameters is not None and self.active_parameters > self.total_parameters:
            raise ValueError("active parameters cannot exceed total parameters")
        if self.kv_heads > self.attention_heads:
            raise ValueError("KV heads cannot exceed attention heads")
        if self.active_experts_per_token is not None:
            if self.moe_experts is None:
                raise ValueError("active experts require total MoE expert metadata")
            if self.active_experts_per_token > self.moe_experts:
                raise ValueError("active experts per token cannot exceed total experts")
        return self


class ModelRecord(ContractModel):
    model_id: str = Field(pattern=r"^.+/.+$")
    revision: str = Field(min_length=1)
    family: str = Field(min_length=1)
    release_date: date | None = None
    license: str | None = None
    gated: bool = False
    capabilities: tuple[ModelCapability, ...] = (ModelCapability.TEXT_GENERATION,)
    languages: tuple[str, ...] = ()
    evaluations: tuple[ModelEvaluation, ...] = ()
    architecture: TransformerArchitecture
    source_url: str = Field(min_length=1)


class ArtifactFormat(StrEnum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    AWQ = "awq"
    GPTQ = "gptq"
    OTHER = "other"


class WeightDtype(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    OTHER = "other"


class ArtifactRecord(ContractModel):
    artifact_id: str = Field(min_length=1)
    model_id: str = Field(pattern=r"^.+/.+$")
    revision: str = Field(min_length=1)
    format: ArtifactFormat
    weight_dtype: WeightDtype
    quantization: str = Field(min_length=1)
    total_bytes: int | None = Field(default=None, ge=1)
    shard_bytes: tuple[int, ...] = ()
    compatible_runtime_ids: tuple[str, ...] = Field(min_length=1)
    kernel_requirements: tuple[str, ...] = ()
    source_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_size(self) -> ArtifactRecord:
        if self.shard_bytes and any(size < 1 for size in self.shard_bytes):
            raise ValueError("artifact shard sizes must be positive")
        if self.total_bytes is not None and self.shard_bytes:
            if sum(self.shard_bytes) != self.total_bytes:
                raise ValueError("artifact shard sizes must sum to total_bytes")
        return self


class ModelCatalog(ContractModel):
    kind: Literal["models"] = "models"
    metadata: CatalogMetadata
    models: tuple[ModelRecord, ...]
    artifacts: tuple[ArtifactRecord, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> ModelCatalog:
        model_keys = [(model.model_id, model.revision) for model in self.models]
        if len(model_keys) != len(set(model_keys)):
            raise ValueError("model ID/revision pairs must be unique")

        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique")

        known_models = set(model_keys)
        for artifact in self.artifacts:
            if (artifact.model_id, artifact.revision) not in known_models:
                raise ValueError(
                    f"artifact {artifact.artifact_id!r} references an unknown model revision"
                )
        return self


class ParameterCoverageBand(ContractModel):
    band_id: str = Field(min_length=1)
    minimum_parameters: int = Field(ge=0)
    maximum_parameters: int | None = Field(default=None, ge=1)
    minimum_models: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> ParameterCoverageBand:
        if (
            self.maximum_parameters is not None
            and self.maximum_parameters < self.minimum_parameters
        ):
            raise ValueError("coverage-band maximum cannot precede minimum")
        return self


class ModelCoverageTargets(ContractModel):
    kind: Literal["model-coverage"] = "model-coverage"
    metadata: CatalogMetadata
    minimum_models: int = Field(ge=1)
    minimum_families: int = Field(ge=1)
    required_capabilities: tuple[ModelCapability, ...] = ()
    parameter_bands: tuple[ParameterCoverageBand, ...] = Field(min_length=1)


class ModelCoverageReport(ContractModel):
    catalog_profile_version: str = Field(min_length=1)
    target_profile_version: str = Field(min_length=1)
    catalog_source_count: int = Field(ge=0)
    model_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    gated_model_count: int = Field(ge=0)
    family_counts: dict[str, int]
    capability_counts: dict[str, int]
    license_counts: dict[str, int]
    quantization_counts: dict[str, int]
    parameter_band_counts: dict[str, int]
    provenance_issues: tuple[str, ...] = ()
    provenance_complete: bool
    unmet_targets: tuple[str, ...] = ()
    complete: bool

    @model_validator(mode="after")
    def validate_completion(self) -> ModelCoverageReport:
        if self.provenance_complete != (not self.provenance_issues):
            raise ValueError("provenance completion must agree with provenance issues")
        if self.complete != (not self.unmet_targets and self.provenance_complete):
            raise ValueError("coverage completion must agree with targets and provenance")
        return self
