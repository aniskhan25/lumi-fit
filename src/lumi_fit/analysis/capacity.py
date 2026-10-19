"""Bounded memory-safe serving-capacity searches for recommendation plans."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal

from lumi_fit.analysis.memory import estimate_serving_memory
from lumi_fit.catalog.validation import CatalogSet
from lumi_fit.domain.models import ArtifactRecord, ModelRecord
from lumi_fit.domain.plans import ServingPlan
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.domain.results import (
    AllocationNeighbor,
    CandidateResult,
    CapacityEnvelope,
    CapacityProbe,
    FeasibilityStatus,
)


CONCURRENCY_SEARCH_UPPER_BOUND = 65_536


def _probe_plan(
    plan: ServingPlan,
    *,
    context_tokens: int,
    total_concurrent_sequences: int,
) -> ServingPlan:
    per_replica = math.ceil(total_concurrent_sequences / plan.replicas)
    maximum_batched_tokens = max(
        context_tokens,
        min(context_tokens * per_replica, 65_536),
    )
    options = plan.runtime_options.model_copy(
        update={
            "max_model_length": context_tokens,
            "max_concurrent_sequences": per_replica,
            "max_batched_tokens": maximum_batched_tokens,
        }
    )
    return plan.model_copy(update={"runtime_options": options})


def _probe_serving(
    serving: ServingRequirements,
    *,
    context_tokens: int,
    total_concurrent_sequences: int,
) -> ServingRequirements:
    payload = serving.model_dump(mode="json")
    payload["max_model_length"] = context_tokens
    payload["concurrent_sequences"] = total_concurrent_sequences
    return ServingRequirements.model_validate(payload)


def _capacity_probe(
    *,
    plan: ServingPlan,
    model: ModelRecord,
    artifact: ArtifactRecord,
    serving: ServingRequirements,
    catalogs: CatalogSet,
    context_tokens: int,
    total_concurrent_sequences: int,
) -> CapacityProbe:
    probe_plan = _probe_plan(
        plan,
        context_tokens=context_tokens,
        total_concurrent_sequences=total_concurrent_sequences,
    )
    probe_serving = _probe_serving(
        serving,
        context_tokens=context_tokens,
        total_concurrent_sequences=total_concurrent_sequences,
    )
    memory = estimate_serving_memory(
        model=model,
        artifact=artifact,
        plan=probe_plan,
        serving=probe_serving,
        hardware=catalogs.hardware,
    )
    safe_fraction = Decimal(1) - serving.minimum_hbm_headroom_fraction
    safe_bytes = Decimal(memory.physical_hbm_per_rank_bytes) * safe_fraction
    return CapacityProbe(
        context_tokens=context_tokens,
        total_concurrent_sequences=total_concurrent_sequences,
        concurrent_sequences_per_replica=math.ceil(total_concurrent_sequences / plan.replicas),
        memory_safe=memory.peak_per_rank.upper <= safe_bytes,
        conservative_peak_per_rank_bytes=int(memory.peak_per_rank.upper),
        conservative_hbm_utilization_fraction=(memory.conservative_utilization_fraction),
    )


def estimate_capacity_envelope(
    *,
    candidate: CandidateResult,
    model: ModelRecord,
    artifact: ArtifactRecord,
    serving: ServingRequirements,
    catalogs: CatalogSet,
) -> CapacityEnvelope:
    """Find memory-safe integer boundaries and retain every evaluated point."""

    probes: dict[tuple[int, int], CapacityProbe] = {}

    def probe(context_tokens: int, total_concurrent_sequences: int) -> CapacityProbe:
        key = (context_tokens, total_concurrent_sequences)
        if key not in probes:
            probes[key] = _capacity_probe(
                plan=candidate.plan,
                model=model,
                artifact=artifact,
                serving=serving,
                catalogs=catalogs,
                context_tokens=context_tokens,
                total_concurrent_sequences=total_concurrent_sequences,
            )
        return probes[key]

    requested = probe(serving.max_model_length, serving.concurrent_sequences)

    best_concurrency: CapacityProbe | None = None
    left = 1
    right = CONCURRENCY_SEARCH_UPPER_BOUND
    while left <= right:
        midpoint = (left + right) // 2
        point = probe(serving.max_model_length, midpoint)
        if point.memory_safe:
            best_concurrency = point
            left = midpoint + 1
        else:
            right = midpoint - 1

    minimum_context = serving.input_tokens + serving.max_output_tokens
    best_context: CapacityProbe | None = None
    left = minimum_context
    right = model.architecture.context_length
    while left <= right:
        midpoint = (left + right) // 2
        point = probe(midpoint, serving.concurrent_sequences)
        if point.memory_safe:
            best_context = point
            left = midpoint + 1
        else:
            right = midpoint - 1

    evaluated = tuple(probes[key] for key in sorted(probes, key=lambda item: (item[0], item[1])))
    return CapacityEnvelope(
        requested=requested,
        memory_safe_maximum_concurrency=best_concurrency,
        memory_safe_maximum_context=best_context,
        catalog_context_limit=model.architecture.context_length,
        safe_hbm_limit_fraction=(Decimal(1) - serving.minimum_hbm_headroom_fraction),
        concurrency_search_upper_bound=CONCURRENCY_SEARCH_UPPER_BOUND,
        concurrency_search_capped=(
            best_concurrency is not None
            and best_concurrency.total_concurrent_sequences == CONCURRENCY_SEARCH_UPPER_BOUND
        ),
        evaluated_points=evaluated,
        analytical_performance_maximum_concurrency=None,
        analytical_performance_explanation=(
            "Not estimated: analytical performance at the requested point is a "
            "heuristic prior, not evidence for a maximum sustainable concurrency."
        ),
        assumptions=(
            "Safety uses the conservative per-rank HBM estimate and requested headroom.",
            "Concurrency is workload-wide and divided across replicas with ceiling rounding.",
            "Context search never exceeds the model's cataloged context limit.",
        ),
    )


def _neighbor(
    candidate: CandidateResult,
    relationship: Literal["lower-cost", "higher-capacity"],
) -> AllocationNeighbor:
    envelope = candidate.capacity_envelope
    billing = candidate.analysis.billing
    if (
        envelope is None
        or billing is None
        or envelope.memory_safe_maximum_concurrency is None
        or envelope.memory_safe_maximum_context is None
    ):
        raise ValueError("capacity neighbor requires complete capacity and billing data")
    return AllocationNeighbor(
        relationship=relationship,
        plan_id=candidate.plan_id,
        nodes=candidate.plan.nodes,
        allocated_gcds=candidate.plan.allocated_gcds,
        tensor_parallel=candidate.plan.tensor_parallel,
        replicas=candidate.plan.replicas,
        gpu_hours_per_wallclock_hour=billing.gpu_hours_per_wallclock_hour,
        memory_safe_maximum_concurrency=(
            envelope.memory_safe_maximum_concurrency.total_concurrent_sequences
        ),
        memory_safe_maximum_context_tokens=(envelope.memory_safe_maximum_context.context_tokens),
    )


def _has_neighbor_data(candidate: CandidateResult) -> bool:
    envelope = candidate.capacity_envelope
    return (
        candidate.analysis.feasibility is FeasibilityStatus.FEASIBLE
        and candidate.analysis.billing is not None
        and envelope is not None
        and envelope.requested.memory_safe
        and envelope.memory_safe_maximum_concurrency is not None
        and envelope.memory_safe_maximum_context is not None
    )


def _billing_rate(candidate: CandidateResult) -> Decimal:
    billing = candidate.analysis.billing
    if billing is None:
        raise ValueError("capacity neighbor requires billing data")
    return billing.gpu_hours_per_wallclock_hour


def _maximum_concurrency(candidate: CandidateResult) -> int:
    envelope = candidate.capacity_envelope
    if envelope is None or envelope.memory_safe_maximum_concurrency is None:
        raise ValueError("capacity neighbor requires a maximum concurrency")
    return envelope.memory_safe_maximum_concurrency.total_concurrent_sequences


def attach_capacity_envelopes(
    *,
    candidates: tuple[CandidateResult, ...],
    serving: ServingRequirements,
    catalogs: CatalogSet,
) -> tuple[CandidateResult, ...]:
    """Attach capacity boundaries and references to useful neighboring plans."""

    models = {model.model_id: model for model in catalogs.models.models}
    artifacts = {artifact.artifact_id: artifact for artifact in catalogs.models.artifacts}
    enriched: list[CandidateResult] = []
    for candidate in candidates:
        if candidate.analysis.feasibility is FeasibilityStatus.INFEASIBLE:
            enriched.append(candidate)
            continue
        envelope = estimate_capacity_envelope(
            candidate=candidate,
            model=models[candidate.plan.model_id],
            artifact=artifacts[candidate.plan.artifact_id],
            serving=serving,
            catalogs=catalogs,
        )
        enriched.append(candidate.model_copy(update={"capacity_envelope": envelope}))

    base = tuple(enriched)
    result: list[CandidateResult] = []
    for candidate in base:
        envelope = candidate.capacity_envelope
        billing = candidate.analysis.billing
        if envelope is None or billing is None:
            result.append(candidate)
            continue
        maximum = envelope.memory_safe_maximum_concurrency
        if maximum is None:
            result.append(candidate)
            continue

        eligible = tuple(
            other
            for other in base
            if other.plan_id != candidate.plan_id
            and other.plan.model_id == candidate.plan.model_id
            and _has_neighbor_data(other)
        )
        cheaper = tuple(
            other
            for other in eligible
            if _billing_rate(other) < billing.gpu_hours_per_wallclock_hour
        )
        lower_cost = (
            max(
                cheaper,
                key=lambda other: (
                    _billing_rate(other),
                    _maximum_concurrency(other),
                    other.plan_id,
                ),
            )
            if cheaper
            else None
        )
        higher = tuple(
            other
            for other in eligible
            if _maximum_concurrency(other) > maximum.total_concurrent_sequences
        )
        higher_capacity = (
            min(
                higher,
                key=lambda other: (
                    _billing_rate(other),
                    other.plan.allocated_gcds,
                    _maximum_concurrency(other),
                    other.plan_id,
                ),
            )
            if higher
            else None
        )
        updated = envelope.model_copy(
            update={
                "lower_cost_neighbor": (
                    _neighbor(lower_cost, "lower-cost") if lower_cost else None
                ),
                "higher_capacity_neighbor": (
                    _neighbor(higher_capacity, "higher-capacity") if higher_capacity else None
                ),
            }
        )
        result.append(candidate.model_copy(update={"capacity_envelope": updated}))
    return tuple(result)
