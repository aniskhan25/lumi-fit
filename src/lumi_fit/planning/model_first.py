"""Model-first allocation search for inference serving on LUMI-G."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

from lumi_fit import __version__
from lumi_fit.analysis.capacity import attach_capacity_envelopes
from lumi_fit.catalog.validation import CatalogSet
from lumi_fit.domain.catalogs import AllocationGranularity, PartitionProfile, RuntimeProfile
from lumi_fit.domain.models import ArtifactRecord, ModelRecord
from lumi_fit.domain.plans import (
    ContainerIdentity,
    ReplicaRouting,
    RuntimeOptions,
    ServingPlan,
)
from lumi_fit.domain.requests import ModelSelector, ResourceEnvelope, ServingRequirements
from lumi_fit.domain.results import (
    CandidateResult,
    CatalogVersions,
    FeasibilityStatus,
    RecommendationBundle,
)
from lumi_fit.domain.units import Duration, MemoryAmount
from lumi_fit.planning._shared import (
    analyze_plan,
    eligible_artifacts,
    plan_id,
    powers_of_two,
    runtime_for_artifact,
    stable_hash,
)
from lumi_fit.scoring.engine import enrich_candidates, rank_candidates
from lumi_fit.scoring.quality import model_suitability


def _find_model(catalogs: CatalogSet, model_id: str) -> ModelRecord:
    matches = [model for model in catalogs.models.models if model.model_id == model_id]
    if not matches:
        raise ValueError(f"model {model_id!r} is not present in the curated catalog")
    if len(matches) > 1:
        raise ValueError(f"model {model_id!r} has multiple revisions; select one explicitly")
    return matches[0]


def _eligible_partitions(
    catalogs: CatalogSet,
    resources: ResourceEnvelope,
) -> tuple[PartitionProfile, ...]:
    by_id = {partition.partition_id: partition for partition in catalogs.partitions.partitions}
    if resources.partition != "auto":
        partition = by_id.get(resources.partition)
        if partition is None:
            raise ValueError(f"unknown partition {resources.partition!r}")
        return (partition,)
    requested = resources.enabled_partitions or ("small-g", "standard-g")
    missing = [partition for partition in requested if partition not in by_id]
    if missing:
        raise ValueError(f"unknown enabled partitions: {', '.join(missing)}")
    return tuple(
        by_id[partition]
        for partition in requested
        if by_id[partition].default_enabled
        and (
            resources.allow_partial_node
            or by_id[partition].allocation_granularity is AllocationGranularity.FULL_NODE
        )
    )


def _walltime(resources: ResourceEnvelope, serving: ServingRequirements) -> Duration:
    if resources.walltime is not None:
        return resources.walltime
    if serving.service_duration is not None:
        return serving.service_duration
    if serving.completion_deadline is not None:
        return serving.completion_deadline
    return Duration.model_validate("1 hour")


def _allocation_shape(
    *,
    tensor_parallel: int,
    partition: PartitionProfile,
    resources: ResourceEnvelope,
) -> tuple[int, int] | None:
    nodes = max(1, math.ceil(tensor_parallel / 8), resources.nodes.minimum)
    if (
        nodes > resources.nodes.maximum
        or nodes > partition.max_nodes
        or tensor_parallel % nodes != 0
    ):
        return None

    if partition.allocation_granularity is AllocationGranularity.FULL_NODE:
        gcds_per_node = 8
    else:
        minimum_total = max(tensor_parallel, resources.gcds.minimum)
        gcds_per_node = math.ceil(minimum_total / nodes)
    if gcds_per_node > 8:
        return None
    allocated_gcds = nodes * gcds_per_node
    if allocated_gcds > resources.gcds.maximum:
        return None
    if tensor_parallel > allocated_gcds:
        return None
    return nodes, gcds_per_node


def _build_plan(
    *,
    model: ModelRecord,
    artifact: ArtifactRecord,
    runtime: RuntimeProfile,
    partition: PartitionProfile,
    resources: ResourceEnvelope,
    serving: ServingRequirements,
    tensor_parallel: int,
    nodes: int,
    gcds_per_node: int,
    catalogs: CatalogSet,
) -> ServingPlan:
    ranks_per_node = tensor_parallel // nodes
    if resources.cpu_cores_per_node is not None:
        cpu_cores = resources.cpu_cores_per_node
    elif partition.allocation_granularity is AllocationGranularity.FULL_NODE:
        cpu_cores = catalogs.hardware.cpu.schedulable_cores
    else:
        cpu_cores = min(catalogs.hardware.cpu.schedulable_cores, 8 * gcds_per_node)

    if resources.cpu_memory_per_node is not None:
        cpu_memory = resources.cpu_memory_per_node
    elif partition.allocation_granularity is AllocationGranularity.FULL_NODE:
        cpu_memory = catalogs.hardware.cpu.memory_per_node
    else:
        cpu_memory = MemoryAmount.model_validate(f"{64 * gcds_per_node} GB")

    container_path = runtime.verified_container_path or runtime.container_path_pattern
    maximum_batched_tokens = max(
        serving.max_model_length,
        min(serving.max_model_length * serving.concurrent_sequences, 65_536),
    )
    return ServingPlan(
        model_id=model.model_id,
        model_revision=model.revision,
        artifact_id=artifact.artifact_id,
        quantization=artifact.quantization,
        runtime_id=runtime.runtime_id,
        container=ContainerIdentity(
            family=runtime.container_family,
            path=container_path,
        ),
        runtime_options=RuntimeOptions(
            max_model_length=serving.max_model_length,
            max_concurrent_sequences=serving.concurrent_sequences,
            max_batched_tokens=maximum_batched_tokens,
            gpu_memory_utilization=Decimal(1) - serving.minimum_hbm_headroom_fraction,
            kv_cache_dtype="auto",
        ),
        partition=partition.partition_id,
        nodes=nodes,
        gcds_per_node=gcds_per_node,
        cpu_cores_per_node=cpu_cores,
        cpu_memory_per_node=cpu_memory,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=1,
        replicas=1,
        request_routing=ReplicaRouting.SINGLE,
        ranks_per_node=ranks_per_node,
        threads_per_rank=max(1, cpu_cores // ranks_per_node),
        walltime=_walltime(resources, serving),
    )


def _rank_key(candidate: CandidateResult) -> tuple[int, Decimal, int, Decimal, str]:
    status_order = {
        FeasibilityStatus.FEASIBLE: 0,
        FeasibilityStatus.CONDITIONAL: 1,
        FeasibilityStatus.INFEASIBLE: 2,
    }
    billing = candidate.analysis.billing
    memory = candidate.analysis.memory
    return (
        status_order[candidate.analysis.feasibility],
        billing.gpu_hours_per_wallclock_hour if billing else Decimal("Infinity"),
        candidate.plan.allocated_gcds,
        memory.conservative_utilization_fraction if memory else Decimal("Infinity"),
        candidate.plan_id,
    )


def _assign_tiers(candidates: tuple[CandidateResult, ...]) -> tuple[CandidateResult, ...]:
    feasible = [
        candidate
        for candidate in candidates
        if candidate.analysis.feasibility is FeasibilityStatus.FEASIBLE
    ]
    if not feasible:
        conditional = [
            candidate
            for candidate in candidates
            if candidate.analysis.feasibility is FeasibilityStatus.CONDITIONAL
        ]
        if not conditional:
            return candidates
        closest = min(conditional, key=_rank_key)
        return tuple(
            candidate.model_copy(
                update={"tiers": ("closest-conditional",) if candidate is closest else ()}
            )
            for candidate in candidates
        )
    minimum = min(feasible, key=_rank_key)
    recommended = feasible[0]
    throughput_optimized = min(
        feasible,
        key=lambda candidate: (
            -(
                candidate.analysis.performance.aggregate_output_tokens_per_second.expected
                if candidate.analysis.performance
                and candidate.analysis.performance.aggregate_output_tokens_per_second
                else Decimal(0)
            ),
            candidate.analysis.billing.total_gpu_hours
            if candidate.analysis.billing
            else Decimal("Infinity"),
            candidate.plan_id,
        ),
    )
    tier_map: dict[str, list[str]] = {minimum.plan_id: ["minimum-runnable"]}
    tier_map.setdefault(recommended.plan_id, []).append("recommended")
    tier_map.setdefault(throughput_optimized.plan_id, []).append("throughput-optimized")
    return tuple(
        candidate.model_copy(update={"tiers": tuple(tier_map.get(candidate.plan_id, []))})
        for candidate in candidates
    )


def generate_model_first_candidates(
    *,
    model_id: str,
    serving: ServingRequirements,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
) -> tuple[CandidateResult, ...]:
    model = _find_model(catalogs, model_id)
    artifacts = eligible_artifacts(catalogs, model, serving)
    if not artifacts:
        raise ValueError("no artifact matches the allowed runtime and quantization policy")
    partitions = _eligible_partitions(catalogs, resources)

    candidates: list[CandidateResult] = []
    for artifact in artifacts:
        runtime = runtime_for_artifact(catalogs, artifact)
        for tensor_parallel in powers_of_two(resources.gcds.maximum):
            for partition in partitions:
                shape = _allocation_shape(
                    tensor_parallel=tensor_parallel,
                    partition=partition,
                    resources=resources,
                )
                if shape is None:
                    continue
                nodes, gcds_per_node = shape
                plan = _build_plan(
                    model=model,
                    artifact=artifact,
                    runtime=runtime,
                    partition=partition,
                    resources=resources,
                    serving=serving,
                    tensor_parallel=tensor_parallel,
                    nodes=nodes,
                    gcds_per_node=gcds_per_node,
                    catalogs=catalogs,
                )
                analysis = analyze_plan(
                    plan=plan,
                    model=model,
                    artifact=artifact,
                    runtime=runtime,
                    partition=partition,
                    serving=serving,
                    resources=resources,
                    catalogs=catalogs,
                )
                candidates.append(
                    CandidateResult(
                        plan_id=plan_id(plan),
                        plan=plan,
                        model_parameters=model.architecture.total_parameters,
                        analysis=analysis,
                    )
                )

    scored = enrich_candidates(
        candidates=tuple(candidates),
        serving=serving,
        catalogs=catalogs,
        suitability_by_model={model.model_id: model_suitability(model, serving.use_case)},
    )
    ranked = rank_candidates(scored, serving)
    return attach_capacity_envelopes(
        candidates=_assign_tiers(ranked),
        serving=serving,
        catalogs=catalogs,
    )


def build_model_first_bundle(
    *,
    model_selector: ModelSelector,
    serving: ServingRequirements,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
    generated_at: datetime | None = None,
) -> RecommendationBundle:
    if model_selector.exact_model is None:
        raise ValueError("model-first planning requires exact_model")
    selected_model = _find_model(catalogs, model_selector.exact_model)
    if selected_model.gated and not model_selector.allow_gated:
        raise ValueError("selected model is gated but allow_gated is false")
    candidates = generate_model_first_candidates(
        model_id=model_selector.exact_model,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    versions = CatalogVersions(
        hardware=catalogs.hardware.metadata.profile_version,
        partitions=catalogs.partitions.metadata.profile_version,
        billing=catalogs.billing.metadata.profile_version,
        runtimes=catalogs.runtimes.metadata.profile_version,
        models=catalogs.models.metadata.profile_version,
        scoring=catalogs.scoring.metadata.profile_version,
    )
    fingerprint_input = {
        "tool_version": __version__,
        "resources": resources.model_dump(mode="json"),
        "serving": serving.model_dump(mode="json"),
        "model_selector": model_selector.model_dump(mode="json"),
        "catalog_versions": versions.model_dump(mode="json"),
        "candidate_plan_ids": [candidate.plan_id for candidate in candidates],
    }
    return RecommendationBundle(
        tool_version=__version__,
        generated_at=generated_at or datetime.now(UTC),
        analysis_fingerprint=stable_hash(fingerprint_input),
        resources=resources,
        serving=serving,
        model_selector=model_selector,
        catalog_versions=versions,
        candidates=candidates,
        objective_sensitivity=(),
        comparisons=(),
        search_complete=True,
        search_notes=(
            "Phase 1 enumerates power-of-two tensor parallelism with one replica.",
            "Phase 4B capacity envelopes use bounded memory-safe searches.",
            "Performance capacity remains unknown without comparable evidence.",
            "Task quality and LUMI-vLLM fit are separate facts; no weighted composite is used.",
            "Numeric fit requires an exact workload, SLO, demand, and measured goodput LCB.",
        ),
    )
