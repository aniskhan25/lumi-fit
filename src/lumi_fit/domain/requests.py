"""User-facing resource, serving, and model-selection contracts."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .models import ModelCapability
from .units import CountRange, Duration, MemoryAmount


class ServingMode(StrEnum):
    ONLINE = "online"
    BATCH = "batch"


class UseCase(StrEnum):
    GENERAL = "general"
    CHAT = "chat"
    CODING = "coding"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class LatencyTargets(ContractModel):
    max_ttft_seconds: Decimal | None = Field(default=None, gt=0)
    max_itl_seconds: Decimal | None = Field(default=None, gt=0)
    max_end_to_end_seconds: Decimal | None = Field(default=None, gt=0)


class ThroughputTargets(ContractModel):
    min_input_tokens_per_second: Decimal | None = Field(default=None, gt=0)
    min_output_tokens_per_second: Decimal | None = Field(default=None, gt=0)
    min_requests_per_second: Decimal | None = Field(default=None, gt=0)


class TokenLengthPoint(ContractModel):
    """One weighted point in a declared prompt/output-length distribution."""

    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    probability: Decimal = Field(gt=0, le=1)


class ServingRequirements(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    mode: ServingMode
    use_case: UseCase = UseCase.GENERAL
    input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_model_length: int = Field(ge=1)
    concurrent_sequences: int = Field(ge=1)
    request_arrival_rate_per_second: Decimal | None = Field(default=None, gt=0)
    batch_request_count: int | None = Field(default=None, ge=1)
    service_duration: Duration | None = None
    completion_deadline: Duration | None = None
    latency_targets: LatencyTargets = LatencyTargets()
    throughput_targets: ThroughputTargets = ThroughputTargets()
    token_length_distribution: tuple[TokenLengthPoint, ...] = ()
    minimum_context_coverage: Decimal = Field(default=Decimal(1), gt=0, le=1)
    required_slo_attainment: Decimal | None = Field(default=None, gt=0, le=1)
    maximum_failure_rate: Decimal | None = Field(default=None, ge=0, lt=1)
    minimum_quality_score: Decimal | None = Field(default=None, ge=0, le=100)
    minimum_hbm_headroom_fraction: Decimal = Field(default=Decimal("0.15"), ge=0, lt=1)
    allowed_runtimes: tuple[str, ...] = ("vllm",)
    allowed_quantizations: tuple[str, ...] = ("bf16", "fp16")
    @model_validator(mode="before")
    @classmethod
    def default_token_length_distribution(cls, value: object) -> object:
        """Use the declared input/output point when no richer distribution is given."""

        if not isinstance(value, dict) or value.get("token_length_distribution"):
            return value
        normalized = dict(value)
        input_tokens = normalized.get("input_tokens")
        output_tokens = normalized.get("max_output_tokens")
        if input_tokens is not None and output_tokens is not None:
            normalized["token_length_distribution"] = (
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "probability": Decimal(1),
                },
            )
        return normalized

    @model_validator(mode="after")
    def validate_serving_shape(self) -> ServingRequirements:
        required_tokens = self.input_tokens + self.max_output_tokens
        if self.max_model_length < required_tokens:
            raise ValueError("max_model_length must cover input_tokens + max_output_tokens")
        if self.mode is ServingMode.ONLINE:
            if self.service_duration is None:
                raise ValueError("online serving requires service_duration")
            if self.completion_deadline is not None or self.batch_request_count is not None:
                raise ValueError(
                    "online serving cannot set completion_deadline or batch_request_count"
                )
        if self.mode is ServingMode.BATCH:
            if self.batch_request_count is None:
                raise ValueError("batch serving requires batch_request_count")
            if self.service_duration is not None:
                raise ValueError("batch serving cannot set service_duration")
        if self.token_length_distribution:
            probability = sum(
                (point.probability for point in self.token_length_distribution),
                Decimal(0),
            )
            if probability != Decimal(1):
                raise ValueError("token length distribution probabilities must sum to 1")
            if (
                max(
                    point.input_tokens + point.output_tokens
                    for point in self.token_length_distribution
                )
                > self.max_model_length
            ):
                # This can still be a valid non-strict distribution, but only when the
                # declared coverage threshold permits the out-of-context probability.
                covered = sum(
                    (
                        point.probability
                        for point in self.token_length_distribution
                        if point.input_tokens + point.output_tokens <= self.max_model_length
                    ),
                    Decimal(0),
                )
                if covered < self.minimum_context_coverage:
                    raise ValueError(
                        "token length distribution does not meet minimum context coverage"
                    )
        return self


class ResourceEnvelope(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    partition: str = "auto"
    nodes: CountRange = CountRange(minimum=1, maximum=1)
    gcds: CountRange
    cpu_cores_per_node: int | None = Field(default=None, ge=1)
    cpu_memory_per_node: MemoryAmount | None = None
    walltime: Duration | None = None
    maximum_gpu_hours: Decimal | None = Field(default=None, gt=0)
    allow_partial_node: bool = True
    allow_multi_node: bool = False
    enabled_partitions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_resource_bounds(self) -> ResourceEnvelope:
        if self.nodes.minimum < 1:
            raise ValueError("at least one node is required for LLM serving")
        if not self.allow_multi_node and self.nodes.maximum > 1:
            raise ValueError("nodes cannot exceed one when allow_multi_node is false")
        if self.gcds.minimum < 1:
            raise ValueError("at least one GCD is required for LLM serving")
        return self


class ModelSelector(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    exact_model: str | None = None
    models: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    use_case: UseCase | None = None
    capabilities: tuple[ModelCapability, ...] = ()
    families: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    minimum_parameters: int | None = Field(default=None, ge=1)
    maximum_parameters: int | None = Field(default=None, ge=1)
    allow_gated: bool = False

    @model_validator(mode="after")
    def require_selection(self) -> ModelSelector:
        filters = (
            self.capabilities,
            self.families,
            self.licenses,
            self.languages,
            self.minimum_parameters,
            self.maximum_parameters,
        )
        if not (
            self.exact_model
            or self.models
            or self.include_patterns
            or self.use_case
            or any(filters)
        ):
            raise ValueError("model selector must provide a model, pattern, or use case")
        if (
            self.minimum_parameters is not None
            and self.maximum_parameters is not None
            and self.minimum_parameters > self.maximum_parameters
        ):
            raise ValueError("minimum_parameters cannot exceed maximum_parameters")
        return self
