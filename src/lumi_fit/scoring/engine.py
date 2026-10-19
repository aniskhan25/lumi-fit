"""Workload-specific vLLM fit scoring for LUMI deployments."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from lumi_fit.analysis.performance import estimate_serving_performance
from lumi_fit.catalog.validation import CatalogSet
from lumi_fit.domain.models import ArtifactRecord, ModelRecord
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.domain.results import (
    CandidateResult,
    ConfidenceClass,
    EligibilityGate,
    FeasibilityStatus,
    FitEligibilityStatus,
    LumiVllmFitAnalysis,
    ModelSuitabilityEvidence,
    QualityRequirementStatus,
    SloGoodputEvidence,
)


def _models(catalogs: CatalogSet) -> dict[tuple[str, str], ModelRecord]:
    return {(model.model_id, model.revision): model for model in catalogs.models.models}


def _artifacts(catalogs: CatalogSet) -> dict[str, ArtifactRecord]:
    return {artifact.artifact_id: artifact for artifact in catalogs.models.artifacts}


def _constraint_passed(candidate: CandidateResult, constraint_id: str) -> bool | None:
    result = next(
        (
            constraint
            for constraint in candidate.analysis.constraints
            if constraint.constraint_id == constraint_id
        ),
        None,
    )
    return result.passed if result is not None else None


def _support_gate(candidate: CandidateResult) -> EligibilityGate:
    runtime_constraints = tuple(
        constraint
        for constraint in candidate.analysis.constraints
        if constraint.constraint_id.startswith("runtime-")
    )
    supported = candidate.plan.runtime_id.startswith("vllm") and all(
        constraint.passed for constraint in runtime_constraints
    )
    return EligibilityGate(
        gate_id="support",
        passed=supported,
        explanation=(
            "The exact architecture, precision, and layout are supported by the "
            "cataloged vLLM/ROCm stack."
            if supported
            else (
                "The exact architecture, precision, or layout is not proven supported "
                "by the selected vLLM/ROCm stack."
            )
        ),
        evidence_ids=(candidate.plan.runtime_id,),
    )


def _memory_gate(candidate: CandidateResult) -> EligibilityGate:
    passed = _constraint_passed(candidate, "physical-hbm")
    return EligibilityGate(
        gate_id="memory",
        passed=passed,
        explanation=(
            "Weights, KV cache, and runtime memory fit independently on every GCD."
            if passed
            else "Per-GCD weights, KV cache, and runtime memory exceed physical HBM."
            if passed is False
            else "Per-GCD memory eligibility could not be evaluated."
        ),
        evidence_ids=("memory.per-rank.v1",) if passed is not None else (),
    )


def _context_gate(
    candidate: CandidateResult,
    serving: ServingRequirements,
) -> EligibilityGate:
    if not serving.token_length_distribution:
        return EligibilityGate(
            gate_id="context",
            passed=None,
            explanation=(
                "No prompt/output-length distribution is declared; context coverage "
                "cannot be established."
            ),
        )
    limit = candidate.plan.runtime_options.max_model_length
    coverage = sum(
        (
            point.probability
            for point in serving.token_length_distribution
            if point.input_tokens + point.output_tokens <= limit
        ),
        Decimal(0),
    )
    passed = coverage >= serving.minimum_context_coverage
    return EligibilityGate(
        gate_id="context",
        passed=passed,
        explanation=(
            f"Declared length-distribution coverage is {coverage:.4f}; required "
            f"coverage is {serving.minimum_context_coverage:.4f}."
        ),
        evidence_ids=("workload.token-length-distribution",),
    )


def _quality_gate(
    serving: ServingRequirements,
    suitability: ModelSuitabilityEvidence | None,
) -> EligibilityGate:
    if serving.minimum_quality_score is None:
        return EligibilityGate(
            gate_id="quality",
            passed=None,
            explanation="No minimum task-quality threshold is declared.",
        )
    if suitability is None or suitability.score is None:
        return EligibilityGate(
            gate_id="quality",
            passed=None,
            explanation="No comparable task-quality evidence is available for this model.",
        )
    passed = suitability.score >= serving.minimum_quality_score
    return EligibilityGate(
        gate_id="quality",
        passed=passed,
        explanation=(
            f"Task quality is {suitability.score:.1f}; required minimum is "
            f"{serving.minimum_quality_score:.1f}."
        ),
        evidence_ids=suitability.evidence_ids,
    )


def _evidence_matches(
    candidate: CandidateResult,
    serving: ServingRequirements,
    evidence: SloGoodputEvidence,
) -> bool:
    plan = candidate.plan
    return (
        evidence.model_id == plan.model_id
        and evidence.model_revision == plan.model_revision
        and evidence.artifact_id == plan.artifact_id
        and evidence.runtime_id == plan.runtime_id
        and evidence.precision == plan.quantization
        and evidence.context_tokens == plan.runtime_options.max_model_length
        and evidence.gcd_count == plan.gcds_per_replica
        and evidence.tensor_parallel == plan.tensor_parallel
        and evidence.concurrent_sequences == plan.runtime_options.max_concurrent_sequences
        and evidence.token_length_distribution == serving.token_length_distribution
    )


def evaluate_lumi_vllm_fit(
    *,
    candidate: CandidateResult,
    serving: ServingRequirements,
    suitability: ModelSuitabilityEvidence | None,
    evidence: SloGoodputEvidence | None = None,
    gcds_per_lumi_node: int = 8,
) -> LumiVllmFitAnalysis:
    """Evaluate hard gates and, when justified, the goodput-coverage fit score."""

    gates = (
        _support_gate(candidate),
        _memory_gate(candidate),
        _context_gate(candidate, serving),
        _quality_gate(serving, suitability),
    )
    technical_gates = gates[:3]
    quality_gate = gates[3]
    if any(gate.passed is False for gate in technical_gates):
        eligibility = FitEligibilityStatus.INELIGIBLE
    elif all(gate.passed is True for gate in technical_gates):
        eligibility = FitEligibilityStatus.ELIGIBLE
    else:
        eligibility = FitEligibilityStatus.UNKNOWN
    if serving.minimum_quality_score is None:
        quality_status = QualityRequirementStatus.NOT_REQUIRED
    elif quality_gate.passed is True:
        quality_status = QualityRequirementStatus.PASS
    elif quality_gate.passed is False:
        quality_status = QualityRequirementStatus.FAIL
    else:
        quality_status = QualityRequirementStatus.UNKNOWN

    equivalent_replicas = (
        gcds_per_lumi_node // candidate.plan.gcds_per_replica
        if candidate.plan.gcds_per_replica <= gcds_per_lumi_node
        else 0
    )
    performance = candidate.analysis.performance
    raw_capacity = (
        performance.requests_per_second.lower
        / Decimal(candidate.plan.replicas)
        * Decimal(equivalent_replicas)
        if performance is not None and performance.requests_per_second is not None
        else None
    )
    target = serving.request_arrival_rate_per_second
    estimated_coverage_score = None
    if (
        eligibility is FitEligibilityStatus.ELIGIBLE
        and target is not None
        and raw_capacity is not None
    ):
        estimated_coverage_score = (Decimal(100) * min(Decimal(1), raw_capacity / target)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
    slo_declared = (
        target is not None
        and serving.required_slo_attainment is not None
        and serving.maximum_failure_rate is not None
        and serving.latency_targets.max_ttft_seconds is not None
        and serving.latency_targets.max_itl_seconds is not None
    )
    evidence_valid = (
        evidence is not None
        and _evidence_matches(candidate, serving, evidence)
        and evidence.confidence
        in {
            ConfidenceClass.MEASURED_EXACT,
            ConfidenceClass.MEASURED_NEAR,
            ConfidenceClass.CALIBRATED,
        }
        and serving.required_slo_attainment is not None
        and evidence.slo_attainment_fraction >= serving.required_slo_attainment
        and serving.maximum_failure_rate is not None
        and evidence.failure_rate <= serving.maximum_failure_rate
    )
    deployment_lcb = (
        evidence.request_goodput_lcb95_per_second
        if evidence_valid and evidence is not None
        else None
    )
    node_lcb = deployment_lcb * Decimal(equivalent_replicas) if deployment_lcb is not None else None
    fit_score = None
    if (
        eligibility is FitEligibilityStatus.ELIGIBLE
        and quality_status in {QualityRequirementStatus.NOT_REQUIRED, QualityRequirementStatus.PASS}
        and slo_declared
        and target is not None
        and node_lcb is not None
    ):
        fit_score = (Decimal(100) * min(Decimal(1), node_lcb / target)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

    if eligibility is FitEligibilityStatus.INELIGIBLE:
        explanation = "INELIGIBLE: at least one support, memory, or context gate failed."
    elif quality_status is QualityRequirementStatus.FAIL:
        explanation = "UNSCORED: the explicit task-quality requirement failed."
    elif quality_status is QualityRequirementStatus.UNKNOWN:
        explanation = "UNSCORED: the explicit task-quality requirement lacks evidence."
    elif fit_score is not None:
        explanation = "Numeric fit is full-node 95% LCB SLO-goodput divided by target request rate."
    elif not slo_declared:
        explanation = (
            "UNSCORED: the complete SLO declaration (request rate, TTFT/TPOT limits, "
            "attainment, and failure tolerance) is not present; estimated analytical "
            "coverage remains available."
        )
    elif not evidence_valid:
        explanation = (
            "UNSCORED: no exact measured SLO-goodput evidence matches this deployment "
            "and workload; heuristic throughput is not converted into a fit score."
        )
    else:
        explanation = (
            "UNSCORED: eligibility is UNKNOWN because at least one gate lacks "
            "a declaration or matching evidence."
        )

    return LumiVllmFitAnalysis(
        eligibility=eligibility,
        gates=gates,
        quality_status=quality_status,
        model_quality=suitability.score if suitability is not None else None,
        minimum_quality_score=serving.minimum_quality_score,
        target_request_rate_per_second=target,
        deployment_goodput_lcb95_per_second=deployment_lcb,
        equivalent_replicas_per_lumi_node=equivalent_replicas,
        full_node_goodput_lcb95_per_second=node_lcb,
        raw_full_node_capacity_lower_per_second=raw_capacity,
        estimated_capacity_coverage_score=estimated_coverage_score,
        fit_score=fit_score,
        confidence=(
            evidence.confidence if evidence_valid and evidence else ConfidenceClass.UNKNOWN
        ),
        evidence_ids=evidence.evidence_ids if evidence_valid and evidence else (),
        explanation=explanation,
    )


def model_suitability_is_comparable(
    candidates: tuple[CandidateResult, ...],
    suitability_by_model: dict[str, ModelSuitabilityEvidence] | None,
) -> bool:
    """Return whether task-quality evidence is directly comparable across models."""

    model_ids = {candidate.plan.model_id for candidate in candidates}
    if len(model_ids) <= 1:
        return True
    if not suitability_by_model:
        return False
    evidence = [suitability_by_model.get(model_id) for model_id in sorted(model_ids)]
    if any(item is None or item.score is None or not item.evidence_ids for item in evidence):
        return False
    evidence_ids = {item.evidence_ids for item in evidence if item is not None}
    return len(evidence_ids) == 1


def rescore_candidates(
    *,
    candidates: tuple[CandidateResult, ...],
    serving: ServingRequirements,
    catalogs: CatalogSet,
    suitability_by_model: dict[str, ModelSuitabilityEvidence] | None = None,
    exclude_model_suitability: bool = False,
    goodput_by_plan: dict[str, SloGoodputEvidence] | None = None,
) -> tuple[CandidateResult, ...]:
    """Attach non-composite LUMI-vLLM fit results to candidates."""

    del catalogs
    enriched: list[CandidateResult] = []
    for candidate in candidates:
        suitability = None
        if not exclude_model_suitability:
            suitability = (suitability_by_model or {}).get(candidate.plan.model_id)
        fit = evaluate_lumi_vllm_fit(
            candidate=candidate,
            serving=serving,
            suitability=suitability,
            evidence=(goodput_by_plan or {}).get(candidate.plan_id),
        )
        analysis = candidate.analysis.model_copy(update={"scores": (), "composite_score": None})
        enriched.append(candidate.model_copy(update={"analysis": analysis, "lumi_vllm_fit": fit}))
    return tuple(enriched)


def enrich_candidates(
    *,
    candidates: tuple[CandidateResult, ...],
    serving: ServingRequirements,
    catalogs: CatalogSet,
    suitability_by_model: dict[str, ModelSuitabilityEvidence] | None = None,
    exclude_model_suitability: bool = False,
    goodput_by_plan: dict[str, SloGoodputEvidence] | None = None,
) -> tuple[CandidateResult, ...]:
    """Attach analytical raw capacity and evidence-backed fit when available."""

    model_by_key = _models(catalogs)
    artifact_by_id = _artifacts(catalogs)
    with_performance: list[CandidateResult] = []
    for candidate in candidates:
        model = model_by_key[(candidate.plan.model_id, candidate.plan.model_revision)]
        artifact = artifact_by_id[candidate.plan.artifact_id]
        performance = estimate_serving_performance(
            model=model,
            artifact=artifact,
            plan=candidate.plan,
            serving=serving,
        )
        analysis = candidate.analysis.model_copy(update={"performance": performance})
        with_performance.append(candidate.model_copy(update={"analysis": analysis}))
    return rescore_candidates(
        candidates=tuple(with_performance),
        serving=serving,
        catalogs=catalogs,
        suitability_by_model=suitability_by_model,
        exclude_model_suitability=exclude_model_suitability,
        goodput_by_plan=goodput_by_plan,
    )


def serving_targets_satisfied(
    candidate: CandidateResult,
    serving: ServingRequirements,
) -> bool:
    """Only an evidence-backed score of 100 establishes full target satisfaction."""

    del serving
    fit = candidate.lumi_vllm_fit
    return fit is not None and fit.fit_score == Decimal(100)


def rank_candidates(
    candidates: tuple[CandidateResult, ...],
    serving: ServingRequirements,
) -> tuple[CandidateResult, ...]:
    """Rank by serveability, optional quality, validated fit, then estimates."""

    del serving
    feasibility_order = {
        FeasibilityStatus.FEASIBLE: 0,
        FeasibilityStatus.CONDITIONAL: 1,
        FeasibilityStatus.INFEASIBLE: 2,
    }
    eligibility_order = {
        FitEligibilityStatus.ELIGIBLE: 0,
        FitEligibilityStatus.UNKNOWN: 1,
        FitEligibilityStatus.UNSCORED: 1,
        FitEligibilityStatus.INELIGIBLE: 2,
    }
    quality_order = {
        QualityRequirementStatus.NOT_REQUIRED: 0,
        QualityRequirementStatus.PASS: 0,
        QualityRequirementStatus.UNKNOWN: 1,
        QualityRequirementStatus.FAIL: 2,
    }

    def key(
        candidate: CandidateResult,
    ) -> tuple[int, int, int, Decimal, Decimal, Decimal, Decimal, Decimal, str]:
        fit = candidate.lumi_vllm_fit
        score = fit.fit_score if fit and fit.fit_score is not None else Decimal(-1)
        estimated = (
            fit.estimated_capacity_coverage_score
            if fit and fit.estimated_capacity_coverage_score is not None
            else Decimal(-1)
        )
        quality = (
            fit.model_quality
            if fit and fit.fit_score == Decimal(100) and fit.model_quality is not None
            else Decimal(-1)
        )
        raw_capacity = (
            fit.raw_full_node_capacity_lower_per_second
            if fit and fit.raw_full_node_capacity_lower_per_second is not None
            else Decimal(-1)
        )
        billing = candidate.analysis.billing
        return (
            feasibility_order[candidate.analysis.feasibility],
            eligibility_order[fit.eligibility] if fit else 3,
            quality_order[fit.quality_status] if fit else 3,
            -score,
            -estimated,
            -quality,
            -raw_capacity,
            billing.total_gpu_hours if billing else Decimal("Infinity"),
            candidate.plan_id,
        )

    return tuple(sorted(candidates, key=key))
