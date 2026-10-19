"""Helpers shared by the resource-first and model-first planning searches."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from lumi_fit.analysis.billing import calculate_gpu_billing
from lumi_fit.analysis.memory import estimate_serving_memory
from lumi_fit.catalog.validation import CatalogSet
from lumi_fit.domain.catalogs import PartitionProfile, RuntimeProfile, SupportLevel
from lumi_fit.domain.models import ArtifactRecord, ModelRecord
from lumi_fit.domain.plans import ServingPlan
from lumi_fit.domain.requests import ResourceEnvelope, ServingRequirements
from lumi_fit.domain.results import (
    BillingAnalysis,
    ConfidenceClass,
    ConstraintResult,
    ConstraintSeverity,
    FeasibilityStatus,
    FitBand,
    PlanAnalysis,
)
from lumi_fit.domain.units import MemoryAmount


def powers_of_two(maximum: int) -> tuple[int, ...]:
    values: list[int] = []
    value = 1
    while value <= maximum:
        values.append(value)
        value *= 2
    return tuple(values)


def stable_hash(value: object, length: int | None = None) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def plan_id(plan: ServingPlan) -> str:
    return stable_hash(plan.model_dump(mode="json"), 16)


def eligible_artifacts(
    catalogs: CatalogSet,
    model: ModelRecord,
    serving: ServingRequirements,
) -> tuple[ArtifactRecord, ...]:
    allowed_runtime_ids = {
        runtime.runtime_id
        for runtime in catalogs.runtimes.runtimes
        if runtime.engine in serving.allowed_runtimes
        or runtime.runtime_id in serving.allowed_runtimes
    }
    return tuple(
        artifact
        for artifact in catalogs.models.artifacts
        if artifact.model_id == model.model_id
        and artifact.revision == model.revision
        and artifact.quantization in serving.allowed_quantizations
        and bool(set(artifact.compatible_runtime_ids) & allowed_runtime_ids)
    )


def runtime_for_artifact(catalogs: CatalogSet, artifact: ArtifactRecord) -> RuntimeProfile:
    allowed = set(artifact.compatible_runtime_ids)
    for runtime in catalogs.runtimes.runtimes:
        if runtime.runtime_id in allowed:
            return runtime
    raise ValueError(f"artifact {artifact.artifact_id!r} has no cataloged runtime")


def _support_constraint(
    constraint_id: str,
    support: SupportLevel,
    message: str,
) -> tuple[ConstraintResult, bool, bool]:
    if support is SupportLevel.UNSUPPORTED:
        return (
            ConstraintResult(
                constraint_id=constraint_id,
                severity=ConstraintSeverity.ERROR,
                passed=False,
                message=message,
            ),
            True,
            False,
        )
    if support in {SupportLevel.CONDITIONAL, SupportLevel.EXPERIMENTAL, SupportLevel.UNKNOWN}:
        return (
            ConstraintResult(
                constraint_id=constraint_id,
                severity=ConstraintSeverity.WARNING,
                passed=False,
                message=message,
                remediation=(
                    "Treat this layout as conditional until comparable external "
                    "evidence is available."
                ),
            ),
            False,
            True,
        )
    return (
        ConstraintResult(
            constraint_id=constraint_id,
            severity=ConstraintSeverity.INFO,
            passed=True,
            message=message,
        ),
        False,
        False,
    )


def analyze_plan(
    *,
    plan: ServingPlan,
    model: ModelRecord,
    artifact: ArtifactRecord,
    runtime: RuntimeProfile,
    partition: PartitionProfile,
    serving: ServingRequirements,
    resources: ResourceEnvelope,
    catalogs: CatalogSet,
) -> PlanAnalysis:
    constraints: list[ConstraintResult] = []
    infeasible = False
    conditional = False

    context_ok = serving.max_model_length <= model.architecture.context_length
    constraints.append(
        ConstraintResult(
            constraint_id="model-context",
            severity=ConstraintSeverity.INFO if context_ok else ConstraintSeverity.ERROR,
            passed=context_ok,
            message=(
                "Requested context is within the model limit"
                if context_ok
                else "Requested context exceeds the model's cataloged limit"
            ),
            actual=str(serving.max_model_length),
            required=f"<= {model.architecture.context_length}",
            remediation=None if context_ok else "Reduce max_model_length or choose another model.",
        )
    )
    infeasible |= not context_ok

    divisible = model.architecture.attention_heads % plan.tensor_parallel == 0
    constraints.append(
        ConstraintResult(
            constraint_id="tensor-parallel-divisibility",
            severity=ConstraintSeverity.INFO if divisible else ConstraintSeverity.ERROR,
            passed=divisible,
            message=(
                "Attention heads divide evenly across tensor-parallel ranks"
                if divisible
                else "Attention heads do not divide across tensor-parallel ranks"
            ),
            actual=str(plan.tensor_parallel),
            required=f"divisor of {model.architecture.attention_heads}",
        )
    )
    infeasible |= not divisible

    if plan.tensor_parallel > 1:
        result, failed, needs_validation = _support_constraint(
            "runtime-tensor-parallel",
            runtime.tensor_parallel,
            f"Runtime tensor-parallel support is {runtime.tensor_parallel.value}",
        )
        constraints.append(result)
        infeasible |= failed
        conditional |= needs_validation

    if plan.nodes > 1:
        result, failed, needs_validation = _support_constraint(
            "runtime-multi-node",
            runtime.multi_node,
            f"Runtime multi-node support is {runtime.multi_node.value}",
        )
        constraints.append(result)
        infeasible |= failed
        conditional |= needs_validation

    quantization = next(
        (item for item in runtime.quantizations if item.name == artifact.quantization),
        None,
    )
    quant_support = quantization.support if quantization else SupportLevel.UNSUPPORTED
    result, failed, needs_validation = _support_constraint(
        "runtime-quantization",
        quant_support,
        f"Runtime support for {artifact.quantization} is {quant_support.value}",
    )
    constraints.append(result)
    infeasible |= failed
    conditional |= needs_validation

    walltime_ok = plan.walltime.seconds <= partition.max_walltime.seconds
    node_hours_ok = (
        partition.max_node_hours is None
        or Decimal(plan.nodes) * plan.walltime.hours <= partition.max_node_hours
    )
    partition_ok = plan.nodes <= partition.max_nodes and walltime_ok and node_hours_ok
    constraints.append(
        ConstraintResult(
            constraint_id="partition-limits",
            severity=ConstraintSeverity.INFO if partition_ok else ConstraintSeverity.ERROR,
            passed=partition_ok,
            message=(
                "Plan is within partition node and walltime limits"
                if partition_ok
                else "Plan exceeds partition node, walltime, or node-hour limits"
            ),
        )
    )
    infeasible |= not partition_ok

    node_resources_ok = (
        plan.gcds_per_node <= catalogs.hardware.gcd.gcds_per_node
        and plan.cpu_cores_per_node <= catalogs.hardware.cpu.schedulable_cores
        and plan.cpu_memory_per_node.bytes <= catalogs.hardware.cpu.memory_per_node.bytes
    )
    constraints.append(
        ConstraintResult(
            constraint_id="node-resources",
            severity=ConstraintSeverity.INFO if node_resources_ok else ConstraintSeverity.ERROR,
            passed=node_resources_ok,
            message=(
                "Per-node resources fit the LUMI-G hardware profile"
                if node_resources_ok
                else "Per-node resources exceed the LUMI-G hardware profile"
            ),
        )
    )
    infeasible |= not node_resources_ok

    memory = estimate_serving_memory(
        model=model,
        artifact=artifact,
        plan=plan,
        serving=serving,
        hardware=catalogs.hardware,
    )
    physical_fit = memory.peak_per_rank.upper <= memory.physical_hbm_per_rank_bytes
    safe_limit = Decimal(1) - serving.minimum_hbm_headroom_fraction
    headroom_ok = memory.conservative_utilization_fraction <= safe_limit
    constraints.append(
        ConstraintResult(
            constraint_id="physical-hbm",
            severity=ConstraintSeverity.INFO if physical_fit else ConstraintSeverity.ERROR,
            passed=physical_fit,
            message=(
                "Conservative peak fits physical HBM on every rank"
                if physical_fit
                else "Conservative peak exceeds physical HBM on at least one rank"
            ),
            actual=f"{memory.peak_per_rank.upper} bytes",
            required=f"<= {memory.physical_hbm_per_rank_bytes} bytes",
            remediation=(
                None
                if physical_fit
                else "Increase tensor parallelism or reduce context/concurrency."
            ),
        )
    )
    infeasible |= not physical_fit
    if physical_fit:
        constraints.append(
            ConstraintResult(
                constraint_id="operational-hbm-headroom",
                severity=ConstraintSeverity.INFO if headroom_ok else ConstraintSeverity.WARNING,
                passed=headroom_ok,
                message=(
                    "Conservative peak preserves requested HBM headroom"
                    if headroom_ok
                    else "Plan physically fits but misses requested HBM headroom"
                ),
                actual=f"{memory.conservative_utilization_fraction:.3f}",
                required=f"<= {safe_limit:.3f}",
                remediation=None if headroom_ok else "Add GCDs or reduce context/concurrency.",
            )
        )

    total_memory = MemoryAmount(
        value=plan.cpu_memory_per_node.value * plan.nodes,
        unit=plan.cpu_memory_per_node.unit,
    )
    billing_rule = next(
        rule for rule in catalogs.billing.rules if rule.rule_id == partition.billing_rule
    )
    billing_result = calculate_gpu_billing(
        billing_rule,
        allocated_nodes=plan.nodes,
        allocated_gcds=plan.allocated_gcds,
        allocated_cpu_cores=plan.cpu_cores_per_node * plan.nodes,
        allocated_cpu_memory=total_memory,
        duration=plan.walltime,
    )
    billing = BillingAnalysis(
        gpu_hours_per_wallclock_hour=billing_result.gpu_hours_per_wallclock_hour,
        duration_hours=billing_result.duration_hours,
        total_gpu_hours=billing_result.total_gpu_hours,
        billing_rule_id=billing_result.rule_id,
    )
    budget_ok = (
        resources.maximum_gpu_hours is None
        or billing.total_gpu_hours <= resources.maximum_gpu_hours
    )
    constraints.append(
        ConstraintResult(
            constraint_id="gpu-hour-budget",
            severity=ConstraintSeverity.INFO if budget_ok else ConstraintSeverity.ERROR,
            passed=budget_ok,
            message=(
                "Plan is within the configured GPU-hour budget"
                if budget_ok
                else "Plan exceeds the configured GPU-hour budget"
            ),
            actual=str(billing.total_gpu_hours),
            required=(
                None if resources.maximum_gpu_hours is None else f"<= {resources.maximum_gpu_hours}"
            ),
        )
    )
    infeasible |= not budget_ok

    if infeasible:
        status = FeasibilityStatus.INFEASIBLE
        fit_band = None
    elif conditional:
        status = FeasibilityStatus.CONDITIONAL
        fit_band = None
    else:
        status = FeasibilityStatus.FEASIBLE
        utilization = memory.conservative_utilization_fraction
        if utilization <= Decimal("0.80"):
            fit_band = FitBand.EXCELLENT
        elif headroom_ok:
            fit_band = FitBand.GOOD
        else:
            fit_band = FitBand.MARGINAL

    return PlanAnalysis(
        feasibility=status,
        fit_band=fit_band,
        constraints=tuple(constraints),
        memory=memory,
        billing=billing,
        overall_confidence=ConfidenceClass.HEURISTIC,
        assumptions=(
            "Artifact bytes are tensor-sharded with a conservative imbalance allowance.",
            "KV capacity reserves the configured context for every concurrent sequence.",
            "Runtime workspace, activation, communication, and fragmentation "
            "are priors until benchmarked.",
        ),
    )
