"""Versioned recommendation and explanation result contracts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from .base import ContractModel
from .plans import ServingPlan
from .requests import (
    ModelSelector,
    ObjectiveProfile,
    ResourceEnvelope,
    ServingRequirements,
    TokenLengthPoint,
    UseCase,
)


class FeasibilityStatus(StrEnum):
    FEASIBLE = "feasible"
    CONDITIONAL = "conditional"
    INFEASIBLE = "infeasible"


class FitBand(StrEnum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"


class ConfidenceClass(StrEnum):
    MEASURED_EXACT = "measured-exact"
    MEASURED_NEAR = "measured-near"
    CALIBRATED = "calibrated"
    ANALYTICAL = "analytical"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class ConstraintSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ConstraintResult(ContractModel):
    constraint_id: str = Field(min_length=1)
    severity: ConstraintSeverity
    passed: bool
    message: str = Field(min_length=1)
    actual: str | None = None
    required: str | None = None
    remediation: str | None = None


class EstimateRange(ContractModel):
    lower: Decimal = Field(ge=0)
    expected: Decimal = Field(ge=0)
    upper: Decimal = Field(ge=0)
    unit: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    confidence: ConfidenceClass
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_order(self) -> EstimateRange:
        if not self.lower <= self.expected <= self.upper:
            raise ValueError("estimate range must satisfy lower <= expected <= upper")
        return self


class MemoryComponent(ContractModel):
    component: str = Field(min_length=1)
    per_rank: EstimateRange
    sharding_rule: str = Field(min_length=1)


class MemoryAnalysis(ContractModel):
    components: tuple[MemoryComponent, ...]
    peak_per_rank: EstimateRange
    physical_hbm_per_rank_bytes: int = Field(ge=1)
    expected_utilization_fraction: Decimal = Field(ge=0)
    conservative_utilization_fraction: Decimal = Field(ge=0)


class PerformanceAnalysis(ContractModel):
    ttft_seconds: EstimateRange | None = None
    inter_token_latency_seconds: EstimateRange | None = None
    aggregate_output_tokens_per_second: EstimateRange | None = None
    requests_per_second: EstimateRange | None = None


class BillingAnalysis(ContractModel):
    gpu_hours_per_wallclock_hour: Decimal = Field(ge=0)
    duration_hours: Decimal = Field(gt=0)
    total_gpu_hours: Decimal = Field(ge=0)
    billing_rule_id: str = Field(min_length=1)


class ScoreComponent(ContractModel):
    component_id: str = Field(min_length=1)
    value: Decimal | None = Field(default=None, ge=0, le=100)
    weight: Decimal = Field(ge=0, le=1)
    weighted_contribution: Decimal = Field(ge=0, le=100)
    explanation: str = Field(min_length=1)
    confidence: ConfidenceClass
    formula_id: str = Field(default="score.unspecified", min_length=1)
    evidence_ids: tuple[str, ...] = ()
    raw_inputs: dict[str, str] = Field(default_factory=dict)


class PlanAnalysis(ContractModel):
    feasibility: FeasibilityStatus
    fit_band: FitBand | None = None
    constraints: tuple[ConstraintResult, ...]
    memory: MemoryAnalysis | None = None
    performance: PerformanceAnalysis | None = None
    billing: BillingAnalysis | None = None
    scores: tuple[ScoreComponent, ...] = ()
    composite_score: Decimal | None = Field(default=None, ge=0, le=100)
    overall_confidence: ConfidenceClass
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> PlanAnalysis:
        if self.feasibility is FeasibilityStatus.FEASIBLE and self.fit_band is None:
            raise ValueError("feasible analysis requires a fit band")
        if self.feasibility is not FeasibilityStatus.FEASIBLE and self.fit_band is not None:
            raise ValueError("conditional or infeasible analysis cannot have a fit band")
        if self.feasibility is FeasibilityStatus.INFEASIBLE and not any(
            not result.passed for result in self.constraints
        ):
            raise ValueError("infeasible analysis requires a failed constraint")
        return self


class FitEligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"
    # Retained only so bundles written before ADR 0007's status clarification load.
    UNSCORED = "unscored"


class QualityRequirementStatus(StrEnum):
    NOT_REQUIRED = "not-required"
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class EligibilityGate(ContractModel):
    gate_id: Literal["support", "memory", "context", "quality"]
    passed: bool | None
    explanation: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class SloGoodputEvidence(ContractModel):
    """Measured goodput evidence for one exact vLLM deployment and workload."""

    model_id: str = Field(pattern=r"^.+/.+$")
    model_revision: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    context_tokens: int = Field(ge=1)
    gcd_count: int = Field(ge=1)
    tensor_parallel: int = Field(ge=1)
    concurrent_sequences: int = Field(ge=1)
    token_length_distribution: tuple[TokenLengthPoint, ...] = Field(min_length=1)
    offered_request_rate_per_second: Decimal = Field(gt=0)
    request_goodput_per_second: Decimal = Field(ge=0)
    request_goodput_lcb95_per_second: Decimal = Field(ge=0)
    slo_attainment_fraction: Decimal = Field(ge=0, le=1)
    failure_rate: Decimal = Field(ge=0, le=1)
    sample_size: int = Field(ge=1)
    maximum_slo_constrained: Literal[True] = True
    confidence: ConfidenceClass
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_goodput(self) -> SloGoodputEvidence:
        probability = sum(
            (point.probability for point in self.token_length_distribution),
            Decimal(0),
        )
        if probability != Decimal(1):
            raise ValueError("goodput evidence length probabilities must sum to 1")
        if self.request_goodput_lcb95_per_second > self.request_goodput_per_second:
            raise ValueError("goodput LCB cannot exceed measured goodput")
        if self.request_goodput_per_second > self.offered_request_rate_per_second:
            raise ValueError("request goodput cannot exceed offered request rate")
        return self


class LumiVllmFitAnalysis(ContractModel):
    """Technical serveability, estimated coverage, and validated LUMI fit."""

    # ``eligibility`` is retained as the serialized field name for compatibility;
    # it now means technical serveability and excludes the optional quality gate.
    eligibility: FitEligibilityStatus
    gates: tuple[EligibilityGate, ...] = Field(min_length=4, max_length=4)
    quality_status: QualityRequirementStatus
    model_quality: Decimal | None = Field(default=None, ge=0, le=100)
    minimum_quality_score: Decimal | None = Field(default=None, ge=0, le=100)
    target_request_rate_per_second: Decimal | None = Field(default=None, gt=0)
    deployment_goodput_lcb95_per_second: Decimal | None = Field(default=None, ge=0)
    equivalent_replicas_per_lumi_node: int = Field(ge=0)
    full_node_goodput_lcb95_per_second: Decimal | None = Field(default=None, ge=0)
    raw_full_node_capacity_lower_per_second: Decimal | None = Field(default=None, ge=0)
    estimated_capacity_coverage_score: Decimal | None = Field(default=None, ge=0, le=100)
    estimated_capacity_confidence: Literal[ConfidenceClass.HEURISTIC] = ConfidenceClass.HEURISTIC
    estimated_formula_id: Literal["fit.lumi-vllm-analytical-coverage.v1"] = (
        "fit.lumi-vllm-analytical-coverage.v1"
    )
    fit_score: Decimal | None = Field(default=None, ge=0, le=100)
    confidence: ConfidenceClass
    formula_id: Literal["fit.lumi-vllm-goodput-lcb.v1"] = "fit.lumi-vllm-goodput-lcb.v1"
    evidence_ids: tuple[str, ...] = ()
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fit(self) -> LumiVllmFitAnalysis:
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("fit eligibility gates must be unique")
        serveability_gates = [gate for gate in self.gates if gate.gate_id != "quality"]
        passed = [gate.passed for gate in serveability_gates]
        if self.eligibility is FitEligibilityStatus.ELIGIBLE and not all(
            value is True for value in passed
        ):
            raise ValueError("eligible serveability requires support, memory, and context")
        if self.eligibility is FitEligibilityStatus.INELIGIBLE and not any(
            value is False for value in passed
        ):
            raise ValueError("ineligible serveability requires a failed technical gate")
        if self.quality_status is QualityRequirementStatus.NOT_REQUIRED:
            if self.minimum_quality_score is not None:
                raise ValueError("not-required quality cannot have a minimum")
        elif self.minimum_quality_score is None:
            raise ValueError("required quality status needs a minimum score")
        if self.estimated_capacity_coverage_score is not None and (
            self.eligibility is not FitEligibilityStatus.ELIGIBLE
            or self.target_request_rate_per_second is None
            or self.raw_full_node_capacity_lower_per_second is None
        ):
            raise ValueError("estimated coverage requires serveability, demand, and capacity")
        if self.fit_score is not None:
            required = (
                self.eligibility is FitEligibilityStatus.ELIGIBLE,
                self.quality_status
                in {QualityRequirementStatus.NOT_REQUIRED, QualityRequirementStatus.PASS},
                self.target_request_rate_per_second is not None,
                self.full_node_goodput_lcb95_per_second is not None,
                bool(self.evidence_ids),
            )
            if not all(required):
                raise ValueError("numeric fit requires eligibility, demand, and evidence")
        return self


class CapacityProbe(ContractModel):
    context_tokens: int = Field(ge=1)
    total_concurrent_sequences: int = Field(ge=1)
    concurrent_sequences_per_replica: int = Field(ge=1)
    memory_safe: bool
    conservative_peak_per_rank_bytes: int = Field(ge=0)
    conservative_hbm_utilization_fraction: Decimal = Field(ge=0)


class AllocationNeighbor(ContractModel):
    relationship: Literal["lower-cost", "higher-capacity"]
    plan_id: str = Field(min_length=1)
    nodes: int = Field(ge=1)
    allocated_gcds: int = Field(ge=1)
    tensor_parallel: int = Field(ge=1)
    replicas: int = Field(ge=1)
    gpu_hours_per_wallclock_hour: Decimal = Field(ge=0)
    memory_safe_maximum_concurrency: int = Field(ge=1)
    memory_safe_maximum_context_tokens: int = Field(ge=1)


class CapacityEnvelope(ContractModel):
    basis: Literal["memory-safe"] = "memory-safe"
    method_id: Literal["capacity.memory-binary-search.v1"] = "capacity.memory-binary-search.v1"
    requested: CapacityProbe
    memory_safe_maximum_concurrency: CapacityProbe | None = None
    memory_safe_maximum_context: CapacityProbe | None = None
    catalog_context_limit: int = Field(ge=1)
    safe_hbm_limit_fraction: Decimal = Field(gt=0, le=1)
    concurrency_search_upper_bound: int = Field(ge=1)
    concurrency_search_capped: bool = False
    evaluated_points: tuple[CapacityProbe, ...] = Field(min_length=1)
    analytical_performance_maximum_concurrency: int | None = Field(default=None, ge=1)
    analytical_performance_explanation: str = Field(min_length=1)
    lower_cost_neighbor: AllocationNeighbor | None = None
    higher_capacity_neighbor: AllocationNeighbor | None = None
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_envelope(self) -> CapacityEnvelope:
        points = {
            (point.context_tokens, point.total_concurrent_sequences)
            for point in self.evaluated_points
        }
        requested_key = (
            self.requested.context_tokens,
            self.requested.total_concurrent_sequences,
        )
        if requested_key not in points:
            raise ValueError("requested capacity point must be included in evaluated_points")
        for maximum in (
            self.memory_safe_maximum_concurrency,
            self.memory_safe_maximum_context,
        ):
            if maximum is not None and not maximum.memory_safe:
                raise ValueError("memory-safe maxima must reference safe probes")
        if self.requested.context_tokens > self.catalog_context_limit:
            raise ValueError("requested context cannot exceed the catalog context limit")
        if self.concurrency_search_capped:
            maximum = self.memory_safe_maximum_concurrency
            if (
                maximum is None
                or maximum.total_concurrent_sequences != self.concurrency_search_upper_bound
            ):
                raise ValueError("a capped concurrency search must reach its upper bound")
        neighbor_ids = {
            neighbor.plan_id
            for neighbor in (self.lower_cost_neighbor, self.higher_capacity_neighbor)
            if neighbor is not None
        }
        if len(neighbor_ids) != sum(
            neighbor is not None
            for neighbor in (self.lower_cost_neighbor, self.higher_capacity_neighbor)
        ):
            raise ValueError("capacity neighbors must reference distinct plans")
        return self


class CandidateResult(ContractModel):
    plan_id: str = Field(min_length=1)
    plan: ServingPlan
    model_parameters: int | None = Field(default=None, ge=1)
    analysis: PlanAnalysis
    lumi_vllm_fit: LumiVllmFitAnalysis | None = None
    capacity_envelope: CapacityEnvelope | None = None
    tiers: tuple[str, ...] = ()


class ObjectivePlanRank(ContractModel):
    rank: int = Field(ge=1)
    plan_id: str = Field(min_length=1)
    model_id: str = Field(pattern=r"^.+/.+$")
    composite_score: Decimal = Field(ge=0, le=100)
    serving_targets_satisfied: bool
    component_weights: dict[str, Decimal]
    component_contributions: dict[str, Decimal]
    rank_change_from_requested: int = 0


class ObjectiveSensitivity(ContractModel):
    objective: ObjectiveProfile
    winner_plan_id: str | None = Field(default=None, min_length=1)
    winner_changed_from_requested: bool
    feasible_candidate_count: int = Field(ge=0)
    rankings: tuple[ObjectivePlanRank, ...] = ()
    unavailable_components: tuple[str, ...] = ()
    unavailable_components_excluded: bool
    rank_stable_after_excluding_unavailable_components: bool

    @model_validator(mode="after")
    def validate_rankings(self) -> ObjectiveSensitivity:
        expected = tuple(range(1, len(self.rankings) + 1))
        actual = tuple(item.rank for item in self.rankings)
        if actual != expected:
            raise ValueError("objective sensitivity ranks must be contiguous")
        if self.feasible_candidate_count != len(self.rankings):
            raise ValueError("feasible candidate count must equal ranking count")
        expected_winner = self.rankings[0].plan_id if self.rankings else None
        if self.winner_plan_id != expected_winner:
            raise ValueError("objective winner must be the rank-one plan")
        return self


class ScoreComponentDelta(ContractModel):
    component_id: str = Field(min_length=1)
    winner_value: Decimal | None = Field(default=None, ge=0, le=100)
    compared_value: Decimal | None = Field(default=None, ge=0, le=100)
    value_delta: Decimal | None = None
    winner_contribution: Decimal = Field(ge=0, le=100)
    compared_contribution: Decimal = Field(ge=0, le=100)
    contribution_delta: Decimal
    comparable: bool
    explanation: str = Field(min_length=1)


class PlanComparison(ContractModel):
    objective: ObjectiveProfile
    winner_plan_id: str = Field(min_length=1)
    compared_plan_id: str = Field(min_length=1)
    winner_rank: int = Field(ge=1)
    compared_rank: int = Field(ge=2)
    winner_model_id: str = Field(pattern=r"^.+/.+$")
    compared_model_id: str = Field(pattern=r"^.+/.+$")
    winner_feasibility: FeasibilityStatus
    compared_feasibility: FeasibilityStatus
    winner_targets_satisfied: bool
    compared_targets_satisfied: bool
    component_deltas: tuple[ScoreComponentDelta, ...]
    hbm_headroom_delta_fraction: Decimal | None = None
    memory_safe_concurrency_delta: int | None = None
    gpu_hours_delta: Decimal | None = None
    runtime_compatibility_delta: Decimal | None = None
    quality_comparable: bool
    quality_explanation: str = Field(min_length=1)
    decisive_factor: Literal[
        "feasibility",
        "serving-targets",
        "composite-score",
        "confidence",
        "billing",
        "stable-plan-id",
    ]
    decisive_component_id: str | None = None
    decisive_explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_plans(self) -> PlanComparison:
        if self.winner_plan_id == self.compared_plan_id:
            raise ValueError("a comparison requires two distinct plans")
        component_ids = [delta.component_id for delta in self.component_deltas]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("comparison component deltas must be unique")
        if (
            self.decisive_component_id is not None
            and self.decisive_component_id not in component_ids
        ):
            raise ValueError("decisive component must be present in component deltas")
        if self.decisive_factor != "composite-score" and self.decisive_component_id is not None:
            raise ValueError("only a composite-score decision can name a component")
        return self


class ModelSuitabilityEvidence(ContractModel):
    use_case: UseCase
    score: Decimal | None = Field(default=None, ge=0, le=100)
    confidence: ConfidenceClass
    evidence_ids: tuple[str, ...] = ()
    explanation: str = Field(min_length=1)


class ModelPlanGroup(ContractModel):
    model_id: str = Field(pattern=r"^.+/.+$")
    best_plan_id: str | None = Field(default=None, min_length=1)
    diagnostic_plan_id: str | None = Field(default=None, min_length=1)
    alternative_plan_ids: tuple[str, ...] = ()
    feasible_plan_count: int = Field(ge=0)
    suitability: ModelSuitabilityEvidence

    @model_validator(mode="after")
    def validate_plan_roles(self) -> ModelPlanGroup:
        if self.best_plan_id is not None and self.diagnostic_plan_id is not None:
            raise ValueError("model group cannot have both best and diagnostic plans")
        if (self.best_plan_id is None) != (self.feasible_plan_count == 0):
            raise ValueError("best plan presence must agree with feasible plan count")
        referenced = tuple(
            plan_id
            for plan_id in (
                self.best_plan_id,
                self.diagnostic_plan_id,
                *self.alternative_plan_ids,
            )
            if plan_id is not None
        )
        if len(referenced) != len(set(referenced)):
            raise ValueError("model group plan roles must reference unique plans")
        return self


class CatalogVersions(ContractModel):
    hardware: str = Field(min_length=1)
    partitions: str = Field(min_length=1)
    billing: str = Field(min_length=1)
    runtimes: str = Field(min_length=1)
    models: str = Field(min_length=1)
    scoring: str = Field(min_length=1)


class RecommendationBundle(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    tool_version: str = Field(min_length=1)
    generated_at: datetime
    analysis_fingerprint: str = Field(min_length=1)
    resources: ResourceEnvelope
    serving: ServingRequirements
    model_selector: ModelSelector
    catalog_versions: CatalogVersions
    candidates: tuple[CandidateResult, ...]
    model_groups: tuple[ModelPlanGroup, ...] = ()
    objective_sensitivity: tuple[ObjectiveSensitivity, ...] = ()
    comparisons: tuple[PlanComparison, ...] = ()
    search_complete: bool
    search_notes: tuple[str, ...] = ()
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_groups(self) -> RecommendationBundle:
        plan_ids = [candidate.plan_id for candidate in self.candidates]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("candidate plan IDs must be unique")
        model_ids = [group.model_id for group in self.model_groups]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model groups must have unique model IDs")
        candidate_ids = {candidate.plan_id for candidate in self.candidates}
        candidates_by_id = {candidate.plan_id: candidate for candidate in self.candidates}
        for candidate in self.candidates:
            envelope = candidate.capacity_envelope
            if envelope is None:
                continue
            for neighbor in (
                envelope.lower_cost_neighbor,
                envelope.higher_capacity_neighbor,
            ):
                if neighbor is None:
                    continue
                if neighbor.plan_id not in candidate_ids:
                    raise ValueError("capacity neighbor references an unknown candidate plan")
                if neighbor.plan_id == candidate.plan_id:
                    raise ValueError("capacity neighbor cannot reference its own plan")
                if candidates_by_id[neighbor.plan_id].plan.model_id != candidate.plan.model_id:
                    raise ValueError("capacity neighbors must use the same model")
        objectives = [item.objective for item in self.objective_sensitivity]
        if len(objectives) != len(set(objectives)):
            raise ValueError("objective sensitivity profiles must be unique")
        if objectives and self.serving.objective not in objectives:
            raise ValueError("objective sensitivity must include the requested objective")
        ranked_sets: list[set[str]] = []
        for sensitivity in self.objective_sensitivity:
            ranked_ids = {item.plan_id for item in sensitivity.rankings}
            if not ranked_ids <= candidate_ids:
                raise ValueError("objective sensitivity references an unknown plan")
            ranked_sets.append(ranked_ids)
        if ranked_sets and any(plan_ids != ranked_sets[0] for plan_ids in ranked_sets[1:]):
            raise ValueError("objective profiles must rank the same feasible plan set")
        feasible_ids = {
            candidate.plan_id
            for candidate in self.candidates
            if candidate.analysis.feasibility is FeasibilityStatus.FEASIBLE
        }
        if ranked_sets and ranked_sets[0] != feasible_ids:
            raise ValueError("objective profiles must include every feasible plan")
        for comparison in self.comparisons:
            if {
                comparison.winner_plan_id,
                comparison.compared_plan_id,
            } - candidate_ids:
                raise ValueError("comparison references an unknown candidate plan")
            winner = candidates_by_id[comparison.winner_plan_id]
            compared = candidates_by_id[comparison.compared_plan_id]
            if (
                comparison.winner_model_id != winner.plan.model_id
                or comparison.compared_model_id != compared.plan.model_id
            ):
                raise ValueError("comparison model IDs must match their plans")
            if comparison.objective is not self.serving.objective:
                raise ValueError("comparison must use the bundle's requested objective")
        for group in self.model_groups:
            referenced = {
                plan_id
                for plan_id in (
                    group.best_plan_id,
                    group.diagnostic_plan_id,
                    *group.alternative_plan_ids,
                )
                if plan_id is not None
            }
            if not referenced <= candidate_ids:
                raise ValueError("model group references an unknown candidate plan")
            for candidate in self.candidates:
                if candidate.plan_id in referenced and candidate.plan.model_id != group.model_id:
                    raise ValueError("model group references a plan for another model")
        return self
