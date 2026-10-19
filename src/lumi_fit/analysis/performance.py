"""Transparent analytical serving-capacity priors.

These estimates are deliberately heuristic. They provide a stable Phase 3
ranking signal and must be replaced by calibrated benchmark evidence before
being interpreted as an SLA.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING

from lumi_fit.domain.models import ArtifactRecord, ModelRecord
from lumi_fit.domain.plans import ServingPlan
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.domain.results import ConfidenceClass, EstimateRange, PerformanceAnalysis


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _estimate(
    lower: Decimal,
    expected: Decimal,
    upper: Decimal,
    *,
    unit: str,
    method_id: str,
) -> EstimateRange:
    quantum = Decimal("0.000001")
    return EstimateRange(
        lower=max(Decimal(0), lower.quantize(quantum)),
        expected=max(Decimal(0), expected.quantize(quantum)),
        upper=max(Decimal(0), upper.quantize(quantum, rounding=ROUND_CEILING)),
        unit=unit,
        method_id=method_id,
        confidence=ConfidenceClass.HEURISTIC,
        evidence_ids=("performance-prior.mi250x-gcd.v1",),
    )


def _parallel_efficiency(plan: ServingPlan) -> Decimal:
    tensor_parallel = plan.tensor_parallel
    if tensor_parallel == 1:
        return Decimal("1")
    if tensor_parallel == 2:
        return Decimal("0.95")
    if tensor_parallel == 4:
        return Decimal("0.85")
    if tensor_parallel == 8:
        return Decimal("0.72")
    return Decimal("0.50")


def estimate_serving_performance(
    *,
    model: ModelRecord,
    artifact: ArtifactRecord,
    plan: ServingPlan,
    serving: ServingRequirements,
) -> PerformanceAnalysis:
    """Estimate latency and aggregate decode capacity for one serving plan."""

    parameters = model.architecture.active_parameters or model.architecture.total_parameters
    artifact_bytes = artifact.total_bytes or model.architecture.total_parameters * 2
    parallel_efficiency = _parallel_efficiency(plan)
    effective_ranks = Decimal(plan.tensor_parallel) * parallel_efficiency

    # Conservative effective priors, not vendor peaks: 22 TFLOP/s for prefill
    # and 850 GB/s for decode weight streaming per MI250X GCD.
    prefill_flops = Decimal(2 * parameters * serving.input_tokens)
    prefill_rate = Decimal("22e12") * effective_ranks
    expected_ttft = prefill_flops / prefill_rate + Decimal("0.08")
    ttft = _estimate(
        expected_ttft * Decimal("0.65"),
        expected_ttft,
        expected_ttft * Decimal("1.70"),
        unit="seconds",
        method_id="performance.v1.prefill-effective-flops",
    )

    expected_itl = Decimal(artifact_bytes) / (Decimal("850e9") * effective_ranks)
    expected_itl += Decimal("0.003")
    itl = _estimate(
        expected_itl * Decimal("0.75"),
        expected_itl,
        expected_itl * Decimal("1.55"),
        unit="seconds/token",
        method_id="performance.v1.decode-weight-streaming",
    )

    per_replica_concurrency = plan.runtime_options.max_concurrent_sequences
    batching_gain = Decimal(1) + Decimal("0.55") * _decimal(math.log2(1 + per_replica_concurrency))
    aggregate_expected = Decimal(1) / expected_itl * batching_gain * Decimal(plan.replicas)
    aggregate = _estimate(
        aggregate_expected * Decimal("0.60"),
        aggregate_expected,
        aggregate_expected * Decimal("1.30"),
        unit="tokens/second",
        method_id="performance.v1.decode-capacity-prior",
    )
    requests_expected = aggregate_expected / Decimal(serving.max_output_tokens)
    requests = _estimate(
        requests_expected * Decimal("0.60"),
        requests_expected,
        requests_expected * Decimal("1.30"),
        unit="requests/second",
        method_id="performance.v1.output-size-capacity",
    )
    return PerformanceAnalysis(
        ttft_seconds=ttft,
        inter_token_latency_seconds=itl,
        aggregate_output_tokens_per_second=aggregate,
        requests_per_second=requests,
    )
