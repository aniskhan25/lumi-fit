from decimal import Decimal
from pathlib import Path

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.domain.results import (
    ConfidenceClass,
    FeasibilityStatus,
    FitEligibilityStatus,
    ModelSuitabilityEvidence,
    QualityRequirementStatus,
    SloGoodputEvidence,
)
from lumi_fit.planning.resource_first import build_resource_first_bundle
from lumi_fit.scoring.engine import evaluate_lumi_vllm_fit, rank_candidates

ROOT = Path(__file__).resolve().parents[2]


def bundle():
    return build_resource_first_bundle(
        model_selector=load_model_selector(ROOT / "examples/models-coding.yaml"),
        serving=load_serving(ROOT / "examples/serving-online.yaml"),
        resources=load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml"),
        catalogs=load_default_catalogs(ROOT),
    )


def declared_serving() -> ServingRequirements:
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    payload = serving.model_dump(mode="json")
    payload.update(
        {
            "token_length_distribution": [
                {"input_tokens": 8192, "output_tokens": 2048, "probability": "1"}
            ],
            "required_slo_attainment": "0.99",
            "maximum_failure_rate": "0.01",
            "minimum_quality_score": "70",
        }
    )
    return ServingRequirements.model_validate(payload)


def suitable_quality(score: str = "80") -> ModelSuitabilityEvidence:
    return ModelSuitabilityEvidence(
        use_case="coding",
        score=score,
        confidence=ConfidenceClass.MEASURED_NEAR,
        evidence_ids=("coding-quality-v1",),
        explanation="Comparable coding-quality evaluation.",
    )


def test_without_slo_evidence_has_serveability_and_estimate_but_no_validated_fit() -> None:
    candidate = next(
        item
        for item in bundle().candidates
        if item.analysis.feasibility is FeasibilityStatus.FEASIBLE
    )

    assert candidate.lumi_vllm_fit is not None
    assert candidate.lumi_vllm_fit.eligibility is FitEligibilityStatus.ELIGIBLE
    assert candidate.lumi_vllm_fit.quality_status is QualityRequirementStatus.NOT_REQUIRED
    assert candidate.lumi_vllm_fit.fit_score is None
    assert candidate.lumi_vllm_fit.estimated_capacity_coverage_score is not None
    assert candidate.lumi_vllm_fit.raw_full_node_capacity_lower_per_second is not None
    expected_coverage = (
        Decimal(100)
        * min(
            Decimal(1),
            candidate.lumi_vllm_fit.raw_full_node_capacity_lower_per_second / Decimal("0.25"),
        )
    ).quantize(Decimal("0.1"))
    assert candidate.lumi_vllm_fit.estimated_capacity_coverage_score == expected_coverage


def test_exact_measured_goodput_produces_full_node_demand_coverage_score() -> None:
    candidate = next(
        item
        for item in bundle().candidates
        if item.plan.model_id == "Qwen/Qwen3-8B"
        and item.analysis.feasibility is FeasibilityStatus.FEASIBLE
        and item.plan.tensor_parallel == 1
    )
    serving = declared_serving()
    equivalent_replicas = 8 // candidate.plan.gcds_per_replica
    target_node_goodput = serving.request_arrival_rate_per_second * Decimal("0.8")
    deployment_lcb = target_node_goodput / Decimal(equivalent_replicas)
    evidence = SloGoodputEvidence(
        model_id=candidate.plan.model_id,
        model_revision=candidate.plan.model_revision,
        artifact_id=candidate.plan.artifact_id,
        runtime_id=candidate.plan.runtime_id,
        precision=candidate.plan.quantization,
        context_tokens=candidate.plan.runtime_options.max_model_length,
        gcd_count=candidate.plan.gcds_per_replica,
        tensor_parallel=candidate.plan.tensor_parallel,
        concurrent_sequences=candidate.plan.runtime_options.max_concurrent_sequences,
        token_length_distribution=serving.token_length_distribution,
        offered_request_rate_per_second="1",
        request_goodput_per_second=deployment_lcb * Decimal("1.1"),
        request_goodput_lcb95_per_second=deployment_lcb,
        slo_attainment_fraction="0.995",
        failure_rate="0.001",
        sample_size=10000,
        confidence=ConfidenceClass.MEASURED_EXACT,
        evidence_ids=("lumi-qwen3-8b-goodput-v1",),
    )

    fit = evaluate_lumi_vllm_fit(
        candidate=candidate,
        serving=serving,
        suitability=suitable_quality(),
        evidence=evidence,
    )

    assert fit.eligibility is FitEligibilityStatus.ELIGIBLE
    assert fit.quality_status is QualityRequirementStatus.PASS
    assert fit.fit_score == Decimal("80.0")
    assert fit.model_quality == Decimal(80)
    assert fit.full_node_goodput_lcb95_per_second == target_node_goodput


def test_quality_requirement_does_not_change_technical_serveability() -> None:
    candidate = next(
        item
        for item in bundle().candidates
        if item.analysis.feasibility is FeasibilityStatus.FEASIBLE
    )
    fit = evaluate_lumi_vllm_fit(
        candidate=candidate,
        serving=declared_serving(),
        suitability=suitable_quality("60"),
    )

    assert fit.eligibility is FitEligibilityStatus.ELIGIBLE
    assert fit.quality_status is QualityRequirementStatus.FAIL
    assert fit.fit_score is None
    quality_gate = next(gate for gate in fit.gates if gate.gate_id == "quality")
    assert quality_gate.passed is False


def test_more_tensor_parallelism_reduces_analytical_single_request_itl() -> None:
    result = bundle()
    one = next(
        candidate
        for candidate in result.candidates
        if candidate.plan.model_id.endswith("8B")
        and candidate.plan.tensor_parallel == 1
        and candidate.plan.replicas == 1
    )
    four = next(
        candidate
        for candidate in result.candidates
        if candidate.plan.model_id.endswith("8B") and candidate.plan.tensor_parallel == 4
    )

    assert one.analysis.performance is not None
    assert four.analysis.performance is not None
    assert (
        four.analysis.performance.inter_token_latency_seconds.expected
        < one.analysis.performance.inter_token_latency_seconds.expected
    )


def test_infeasible_candidate_never_outranks_feasible_candidate() -> None:
    result = bundle()
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    ranked = rank_candidates(tuple(reversed(result.candidates)), serving)
    statuses = [candidate.analysis.feasibility for candidate in ranked]

    last_feasible = max(
        index for index, status in enumerate(statuses) if status is FeasibilityStatus.FEASIBLE
    )
    first_infeasible = min(
        index for index, status in enumerate(statuses) if status is FeasibilityStatus.INFEASIBLE
    )
    assert last_feasible < first_infeasible
