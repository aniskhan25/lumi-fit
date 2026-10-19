"""Per-rank HBM estimates for inference serving."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING

from lumi_fit.domain.catalogs import HardwareProfile
from lumi_fit.domain.models import ArtifactRecord, ModelRecord, WeightDtype
from lumi_fit.domain.plans import ServingPlan
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.domain.results import (
    ConfidenceClass,
    EstimateRange,
    MemoryAnalysis,
    MemoryComponent,
)

MIB = 1_048_576
GIB = 1_073_741_824


def _ceil_decimal(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_CEILING)


def _estimate(
    lower: Decimal,
    expected: Decimal,
    upper: Decimal,
    *,
    method_id: str,
    confidence: ConfidenceClass = ConfidenceClass.ANALYTICAL,
) -> EstimateRange:
    return EstimateRange(
        lower=_ceil_decimal(lower),
        expected=_ceil_decimal(expected),
        upper=_ceil_decimal(upper),
        unit="bytes",
        method_id=method_id,
        confidence=confidence,
    )


def _weight_bytes(artifact: ArtifactRecord, model: ModelRecord) -> tuple[int, ConfidenceClass]:
    if artifact.total_bytes is not None:
        return artifact.total_bytes, ConfidenceClass.ANALYTICAL
    bytes_per_parameter = {
        WeightDtype.BF16: 2,
        WeightDtype.FP16: 2,
        WeightDtype.FP32: 4,
        WeightDtype.FP8: 1,
        WeightDtype.INT8: 1,
        WeightDtype.INT4: Decimal("0.5"),
    }.get(artifact.weight_dtype)
    if bytes_per_parameter is None:
        raise ValueError("artifact size is required for unknown weight dtype")
    estimated = Decimal(model.architecture.total_parameters) * Decimal(bytes_per_parameter)
    return int(_ceil_decimal(estimated)), ConfidenceClass.HEURISTIC


def _kv_element_bytes(kv_dtype: str, weight_dtype: WeightDtype) -> Decimal:
    normalized = kv_dtype.lower()
    if normalized == "auto":
        return Decimal(1) if weight_dtype is WeightDtype.FP8 else Decimal(2)
    mapping = {
        "bf16": Decimal(2),
        "fp16": Decimal(2),
        "fp8": Decimal(1),
        "fp8_e4m3": Decimal(1),
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported KV cache dtype {kv_dtype!r}")
    return mapping[normalized]


def estimate_serving_memory(
    *,
    model: ModelRecord,
    artifact: ArtifactRecord,
    plan: ServingPlan,
    serving: ServingRequirements,
    hardware: HardwareProfile,
) -> MemoryAnalysis:
    """Estimate peak HBM for one rank/GCD of a serving replica.

    The formula deliberately separates weights, KV cache, activations,
    workspaces, communication, and allocator reserve. Tensor-parallel behavior
    is applied per component instead of treating node HBM as a flat pool.
    """

    architecture = model.architecture
    tensor_parallel = plan.tensor_parallel
    total_weight_bytes, weight_confidence = _weight_bytes(artifact, model)

    ideal_weight_shard = Decimal(total_weight_bytes) / Decimal(tensor_parallel)
    weights = _estimate(
        ideal_weight_shard,
        ideal_weight_shard * Decimal("1.01"),
        ideal_weight_shard * Decimal("1.03"),
        method_id="weights.v1.artifact-tp-shard",
        confidence=weight_confidence,
    )

    kv_heads_per_rank = max(1, math.ceil(architecture.kv_heads / tensor_parallel))
    kv_element_bytes = _kv_element_bytes(
        plan.runtime_options.kv_cache_dtype,
        artifact.weight_dtype,
    )
    expected_sequence_tokens = min(
        plan.runtime_options.max_model_length,
        serving.input_tokens + serving.max_output_tokens,
    )
    per_replica_concurrency = plan.runtime_options.max_concurrent_sequences
    expected_reserved_tokens = expected_sequence_tokens * per_replica_concurrency
    upper_reserved_tokens = (
        plan.runtime_options.max_model_length * plan.runtime_options.max_concurrent_sequences
    )
    kv_constant = (
        Decimal(2)
        * Decimal(architecture.layers)
        * Decimal(kv_heads_per_rank)
        * Decimal(architecture.head_dimension)
        * kv_element_bytes
    )
    kv_cache = _estimate(
        kv_constant * Decimal(expected_reserved_tokens) * Decimal("0.90"),
        kv_constant * Decimal(expected_reserved_tokens),
        kv_constant * Decimal(upper_reserved_tokens),
        method_id="kv.v1.architecture-per-rank",
    )

    expected_active_tokens = min(
        plan.runtime_options.max_batched_tokens,
        serving.input_tokens * per_replica_concurrency,
    )
    upper_active_tokens = plan.runtime_options.max_batched_tokens
    activation_base = (
        Decimal(architecture.hidden_size) * Decimal(max(1, expected_active_tokens)) * Decimal(2)
    )
    activation_upper_base = (
        Decimal(architecture.hidden_size) * Decimal(max(1, upper_active_tokens)) * Decimal(2)
    )
    activations = _estimate(
        activation_base * Decimal(2),
        activation_base * Decimal(4),
        activation_upper_base * Decimal(8),
        method_id="activations.v1.serving-token-envelope",
        confidence=ConfidenceClass.HEURISTIC,
    )

    workspace_expected = max(Decimal(2 * GIB), weights.expected * Decimal("0.03"))
    workspace_upper = max(Decimal(4 * GIB), weights.upper * Decimal("0.06"))
    workspace = _estimate(
        Decimal(1 * GIB),
        workspace_expected,
        workspace_upper,
        method_id="workspace.v1.vllm-prior",
        confidence=ConfidenceClass.HEURISTIC,
    )

    if tensor_parallel == 1:
        communication = _estimate(
            Decimal(0),
            Decimal(0),
            Decimal(0),
            method_id="communication.v1.none",
        )
    else:
        stages = max(1, math.ceil(math.log2(tensor_parallel)))
        communication = _estimate(
            Decimal(128 * MIB),
            Decimal(256 * MIB * stages),
            Decimal(512 * MIB * stages),
            method_id="communication.v1.tp-buffer-prior",
            confidence=ConfidenceClass.HEURISTIC,
        )

    pre_reserve_expected = sum(
        component.expected
        for component in (weights, kv_cache, activations, workspace, communication)
    )
    pre_reserve_upper = sum(
        component.upper for component in (weights, kv_cache, activations, workspace, communication)
    )
    fragmentation = _estimate(
        pre_reserve_expected * Decimal("0.03"),
        pre_reserve_expected * Decimal("0.05"),
        pre_reserve_upper * Decimal("0.08"),
        method_id="allocator.v1.fragmentation-prior",
        confidence=ConfidenceClass.HEURISTIC,
    )

    component_ranges = (
        ("weights", weights, f"tensor-parallel/{tensor_parallel}"),
        ("kv-cache", kv_cache, f"kv-heads-per-rank/{kv_heads_per_rank}"),
        ("activations", activations, "per-rank serving envelope"),
        ("runtime-workspace", workspace, "per-rank runtime prior"),
        ("communication", communication, f"tensor-parallel/{tensor_parallel}"),
        ("allocator-reserve", fragmentation, "per-rank subtotal reserve"),
    )
    components = tuple(
        MemoryComponent(component=name, per_rank=estimate, sharding_rule=rule)
        for name, estimate, rule in component_ranges
    )
    peak = _estimate(
        sum(component.per_rank.lower for component in components),
        sum(component.per_rank.expected for component in components),
        sum(component.per_rank.upper for component in components),
        method_id="peak-hbm.v1.component-sum",
        confidence=ConfidenceClass.HEURISTIC,
    )
    physical_bytes = hardware.gcd.hbm_per_gcd.bytes
    return MemoryAnalysis(
        components=components,
        peak_per_rank=peak,
        physical_hbm_per_rank_bytes=physical_bytes,
        expected_utilization_fraction=peak.expected / Decimal(physical_bytes),
        conservative_utilization_fraction=peak.upper / Decimal(physical_bytes),
    )
