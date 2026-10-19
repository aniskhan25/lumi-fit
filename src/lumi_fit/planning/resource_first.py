"""Resource-first model discovery inside a fixed LUMI allocation."""

from __future__ import annotations

import fnmatch
import math
from datetime import UTC, datetime
from decimal import Decimal

from lumi_fit import __version__
from lumi_fit.analysis.capacity import attach_capacity_envelopes
from lumi_fit.catalog.validation import CatalogSet
from lumi_fit.domain.catalogs import AllocationGranularity, PartitionProfile
from lumi_fit.domain.models import ModelCapability, ModelRecord
from lumi_fit.domain.plans import (
    ContainerIdentity,
    ReplicaRouting,
    RuntimeOptions,
    ServingPlan,
)
from lumi_fit.domain.requests import (
    ModelSelector,
    ResourceEnvelope,
    ServingRequirements,
    UseCase,
)
from lumi_fit.domain.results import (
    CandidateResult,
    CatalogVersions,
    FeasibilityStatus,
    FitBand,
    ModelPlanGroup,
    RecommendationBundle,
)
from lumi_fit.domain.units import MemoryAmount
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

_USE_CASE_CAPABILITY = {
    UseCase.GENERAL: ModelCapability.TEXT_GENERATION,
    UseCase.CHAT: ModelCapability.CHAT,
    UseCase.CODING: ModelCapability.CODING,
    UseCase.REASONING: ModelCapability.REASONING,
    UseCase.EMBEDDING: ModelCapability.EMBEDDING,
}


def select_models(
    *,
    selector: ModelSelector,
    catalogs: CatalogSet,
) -> tuple[ModelRecord, ...]:
    """Apply explicit IDs, glob patterns, use case, gating, and exclusions."""

    models = catalogs.models.models
    by_id = {model.model_id: model for model in models}
    requested_ids = ({selector.exact_model} if selector.exact_model else set()) | set(
        selector.models
    )
    missing = sorted(model_id for model_id in requested_ids if model_id not in by_id)
    if missing:
        raise ValueError(f"models are not present in the curated catalog: {', '.join(missing)}")

    if requested_ids or selector.include_patterns:
        selected_ids = set(requested_ids)
        selected_ids.update(
            model.model_id
            for model in models
            if any(
                fnmatch.fnmatchcase(model.model_id, pattern)
                for pattern in selector.include_patterns
            )
        )
    else:
        selected_ids = {model.model_id for model in models}

    capability = (
        _USE_CASE_CAPABILITY.get(selector.use_case) if selector.use_case is not None else None
    )
    requested_capabilities = set(selector.capabilities)
    requested_families = {family.casefold() for family in selector.families}
    requested_licenses = {license_id.casefold() for license_id in selector.licenses}
    requested_languages = {language.casefold() for language in selector.languages}
    selected = tuple(
        model
        for model in models
        if model.model_id in selected_ids
        and (selector.allow_gated or not model.gated)
        and (capability is None or capability in model.capabilities)
        and requested_capabilities <= set(model.capabilities)
        and (not requested_families or model.family.casefold() in requested_families)
        and (
            not requested_licenses or (model.license or "unknown").casefold() in requested_licenses
        )
        and (
            not requested_languages
            or bool(requested_languages & {language.casefold() for language in model.languages})
        )
        and (
            selector.minimum_parameters is None
            or model.architecture.total_parameters >= selector.minimum_parameters
        )
        and (
            selector.maximum_parameters is None
            or model.architecture.total_parameters <= selector.maximum_parameters
        )
        and not any(
            fnmatch.fnmatchcase(model.model_id, pattern) for pattern in selector.exclude_patterns
        )
    )
    if not selected:
        raise ValueError("model selector did not match any eligible cataloged models")
    return tuple(sorted(selected, key=lambda model: model.model_id))


def _fixed_allocation(resources: ResourceEnvelope) -> tuple[int, int, int]:
    if not resources.nodes.is_fixed or not resources.gcds.is_fixed:
        raise ValueError("resource-first planning requires fixed nodes and fixed GCDs")
    nodes = resources.nodes.minimum
    total_gcds = resources.gcds.minimum
    if total_gcds % nodes != 0:
        raise ValueError("fixed GCD count must divide evenly across allocated nodes")
    return nodes, total_gcds, total_gcds // nodes


def _compatible_partitions(
    *,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
    nodes: int,
    gcds_per_node: int,
) -> tuple[PartitionProfile, ...]:
    requested = (
        (resources.partition,)
        if resources.partition != "auto"
        else resources.enabled_partitions or ("small-g", "standard-g")
    )
    by_id = {partition.partition_id: partition for partition in catalogs.partitions.partitions}
    missing = [partition_id for partition_id in requested if partition_id not in by_id]
    if missing:
        raise ValueError(f"unknown enabled partitions: {', '.join(missing)}")

    compatible: list[PartitionProfile] = []
    for partition_id in requested:
        partition = by_id[partition_id]
        if nodes > partition.max_nodes:
            continue
        if partition.allocation_granularity is AllocationGranularity.FULL_NODE:
            if gcds_per_node != catalogs.hardware.gcd.gcds_per_node:
                continue
        elif not resources.allow_partial_node:
            continue
        compatible.append(partition)
    if not compatible:
        raise ValueError("fixed allocation is not compatible with any enabled partition")
    return tuple(compatible)


def _cpu_resources(
    *,
    resources: ResourceEnvelope,
    partition: PartitionProfile,
    gcds_per_node: int,
    catalogs: CatalogSet,
) -> tuple[int, MemoryAmount]:
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
    return cpu_cores, cpu_memory


def _build_plan(
    *,
    model: ModelRecord,
    artifact_id: str,
    quantization: str,
    runtime_id: str,
    container_family: str,
    container_path: str,
    partition: PartitionProfile,
    resources: ResourceEnvelope,
    serving: ServingRequirements,
    catalogs: CatalogSet,
    nodes: int,
    gcds_per_node: int,
    tensor_parallel: int,
    replicas: int,
) -> ServingPlan:
    used_gcds = tensor_parallel * replicas
    ranks_per_node = used_gcds // nodes
    cpu_cores, cpu_memory = _cpu_resources(
        resources=resources,
        partition=partition,
        gcds_per_node=gcds_per_node,
        catalogs=catalogs,
    )
    per_replica_concurrency = math.ceil(serving.concurrent_sequences / replicas)
    maximum_batched_tokens = max(
        serving.max_model_length,
        min(serving.max_model_length * per_replica_concurrency, 65_536),
    )
    walltime = resources.walltime or serving.service_duration or serving.completion_deadline
    if walltime is None:
        raise ValueError("serving plan requires a resource or service duration")
    return ServingPlan(
        model_id=model.model_id,
        model_revision=model.revision,
        artifact_id=artifact_id,
        quantization=quantization,
        runtime_id=runtime_id,
        container=ContainerIdentity(family=container_family, path=container_path),
        runtime_options=RuntimeOptions(
            max_model_length=serving.max_model_length,
            max_concurrent_sequences=per_replica_concurrency,
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
        replicas=replicas,
        request_routing=(ReplicaRouting.SINGLE if replicas == 1 else ReplicaRouting.ROUND_ROBIN),
        ranks_per_node=ranks_per_node,
        threads_per_rank=max(1, cpu_cores // ranks_per_node),
        walltime=walltime,
    )


def _rank_key(candidate: CandidateResult) -> tuple[int, int, int, int, Decimal, str]:
    status_rank = {
        FeasibilityStatus.FEASIBLE: 0,
        FeasibilityStatus.CONDITIONAL: 1,
        FeasibilityStatus.INFEASIBLE: 2,
    }[candidate.analysis.feasibility]
    fit_rank = {
        FitBand.EXCELLENT: 0,
        FitBand.GOOD: 1,
        FitBand.MARGINAL: 2,
        None: 3,
    }[candidate.analysis.fit_band]
    unused_gcds = candidate.plan.allocated_gcds - candidate.plan.used_gcds
    memory = candidate.analysis.memory
    utilization = (
        memory.conservative_utilization_fraction if memory is not None else Decimal("Infinity")
    )
    return (
        status_rank,
        fit_rank,
        unused_gcds,
        -candidate.plan.replicas,
        utilization,
        candidate.plan_id,
    )


def _tier_candidates(
    candidates: tuple[CandidateResult, ...],
    serving: ServingRequirements,
) -> tuple[tuple[CandidateResult, ...], str | None, str | None, tuple[str, ...], int]:
    ranked = rank_candidates(candidates, serving)
    feasible = [
        candidate
        for candidate in ranked
        if candidate.analysis.feasibility is FeasibilityStatus.FEASIBLE
    ]
    if feasible:
        best = feasible[0]
        alternatives = tuple(candidate.plan_id for candidate in feasible[1:])
        tiered = tuple(
            candidate.model_copy(
                update={
                    "tiers": (
                        ("best-for-model",)
                        if candidate.plan_id == best.plan_id
                        else ("alternative",)
                        if candidate.plan_id in alternatives
                        else ()
                    )
                }
            )
            for candidate in ranked
        )
        return tiered, best.plan_id, None, alternatives, len(feasible)

    diagnostic = ranked[0] if ranked else None
    tiered = tuple(
        candidate.model_copy(
            update={
                "tiers": (
                    ("closest-conditional",)
                    if candidate is diagnostic
                    and candidate.analysis.feasibility is FeasibilityStatus.CONDITIONAL
                    else ("closest-infeasible",)
                    if candidate is diagnostic
                    else ()
                )
            }
        )
        for candidate in ranked
    )
    return tiered, None, diagnostic.plan_id if diagnostic else None, (), 0


def generate_resource_first_candidates(
    *,
    model: ModelRecord,
    serving: ServingRequirements,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
) -> tuple[CandidateResult, ...]:
    nodes, total_gcds, gcds_per_node = _fixed_allocation(resources)
    if gcds_per_node > catalogs.hardware.gcd.gcds_per_node:
        raise ValueError("fixed allocation exceeds the GCDs available per LUMI-G node")
    partitions = _compatible_partitions(
        resources=resources,
        catalogs=catalogs,
        nodes=nodes,
        gcds_per_node=gcds_per_node,
    )
    artifacts = eligible_artifacts(catalogs, model, serving)
    if not artifacts:
        return ()

    candidates: list[CandidateResult] = []
    for artifact in artifacts:
        runtime = runtime_for_artifact(catalogs, artifact)
        container_path = runtime.verified_container_path or runtime.container_path_pattern
        for partition in partitions:
            for tensor_parallel in powers_of_two(total_gcds):
                for replicas in range(1, total_gcds // tensor_parallel + 1):
                    used_gcds = tensor_parallel * replicas
                    if used_gcds % nodes != 0:
                        continue
                    plan = _build_plan(
                        model=model,
                        artifact_id=artifact.artifact_id,
                        quantization=artifact.quantization,
                        runtime_id=runtime.runtime_id,
                        container_family=runtime.container_family,
                        container_path=container_path,
                        partition=partition,
                        resources=resources,
                        serving=serving,
                        catalogs=catalogs,
                        nodes=nodes,
                        gcds_per_node=gcds_per_node,
                        tensor_parallel=tensor_parallel,
                        replicas=replicas,
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
    return tuple(candidates)


def build_resource_first_bundle(
    *,
    model_selector: ModelSelector,
    serving: ServingRequirements,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
    generated_at: datetime | None = None,
) -> RecommendationBundle:
    """Find the best serving layout for each model in a fixed allocation."""

    _fixed_allocation(resources)
    selected_models = select_models(selector=model_selector, catalogs=catalogs)
    use_case = model_selector.use_case or serving.use_case
    raw_by_model: dict[str, tuple[CandidateResult, ...]] = {}
    suitability_by_model = {
        model.model_id: model_suitability(model, use_case) for model in selected_models
    }
    for model in selected_models:
        raw_by_model[model.model_id] = generate_resource_first_candidates(
            model=model,
            serving=serving,
            resources=resources,
            catalogs=catalogs,
        )

    scored = enrich_candidates(
        candidates=tuple(
            candidate for model in selected_models for candidate in raw_by_model[model.model_id]
        ),
        serving=serving,
        catalogs=catalogs,
        suitability_by_model=suitability_by_model,
    )
    scored_by_model = {
        model.model_id: tuple(
            candidate for candidate in scored if candidate.plan.model_id == model.model_id
        )
        for model in selected_models
    }
    all_candidates: list[CandidateResult] = []
    groups: list[ModelPlanGroup] = []
    for model in selected_models:
        tiered, best_id, diagnostic_id, alternative_ids, feasible_count = _tier_candidates(
            scored_by_model[model.model_id], serving
        )
        all_candidates.extend(tiered)
        groups.append(
            ModelPlanGroup(
                model_id=model.model_id,
                best_plan_id=best_id,
                diagnostic_plan_id=diagnostic_id,
                alternative_plan_ids=alternative_ids,
                feasible_plan_count=feasible_count,
                suitability=suitability_by_model[model.model_id],
            )
        )

    versions = CatalogVersions(
        hardware=catalogs.hardware.metadata.profile_version,
        partitions=catalogs.partitions.metadata.profile_version,
        billing=catalogs.billing.metadata.profile_version,
        runtimes=catalogs.runtimes.metadata.profile_version,
        models=catalogs.models.metadata.profile_version,
        scoring=catalogs.scoring.metadata.profile_version,
    )
    candidates_with_capacity = attach_capacity_envelopes(
        candidates=tuple(all_candidates),
        serving=serving,
        catalogs=catalogs,
    )
    fingerprint_input = {
        "tool_version": __version__,
        "resources": resources.model_dump(mode="json"),
        "serving": serving.model_dump(mode="json"),
        "model_selector": model_selector.model_dump(mode="json"),
        "catalog_versions": versions.model_dump(mode="json"),
        "candidate_plan_ids": [candidate.plan_id for candidate in candidates_with_capacity],
        "model_groups": [group.model_dump(mode="json") for group in groups],
    }
    return RecommendationBundle(
        tool_version=__version__,
        generated_at=generated_at or datetime.now(UTC),
        analysis_fingerprint=stable_hash(fingerprint_input),
        resources=resources,
        serving=serving,
        model_selector=model_selector,
        catalog_versions=versions,
        candidates=candidates_with_capacity,
        model_groups=tuple(groups),
        objective_sensitivity=(),
        comparisons=(),
        search_complete=True,
        search_notes=(
            "Phase 2 searches tensor-parallel and replica layouts inside a fixed allocation.",
            "Best-for-model ranking prioritizes feasibility, fit band, allocation use, "
            "replica count, then conservative HBM use.",
            "Task-quality scores remain null unless comparable evaluation evidence exists.",
            "Phase 4B capacity envelopes use bounded memory-safe searches.",
            "Performance capacity remains unknown without comparable evidence.",
            "Task quality and LUMI-vLLM fit are separate facts; no weighted composite is used.",
            "Numeric fit requires an exact workload, SLO, demand, and measured goodput LCB.",
        ),
    )
