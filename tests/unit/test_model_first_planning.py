from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.requests import ResourceEnvelope, ServingRequirements
from lumi_fit.domain.results import FeasibilityStatus
from lumi_fit.planning.model_first import (
    build_model_first_bundle,
    generate_model_first_candidates,
)


ROOT = Path(__file__).resolve().parents[2]


def inputs():
    return (
        load_default_catalogs(ROOT),
        load_resources(ROOT / "examples/resources-model-first.yaml"),
        load_serving(ROOT / "examples/serving-online.yaml"),
    )


def test_qwen3_8b_minimum_plan_uses_one_small_g_gcd() -> None:
    catalogs, resources, serving = inputs()
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-8B",
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    minimum = next(candidate for candidate in candidates if "minimum-runnable" in candidate.tiers)

    assert minimum.analysis.feasibility is FeasibilityStatus.FEASIBLE
    assert minimum.plan.partition == "small-g"
    assert minimum.plan.tensor_parallel == 1
    assert minimum.plan.allocated_gcds == 1


def test_qwen3_32b_requested_kv_capacity_requires_four_gcds() -> None:
    catalogs, resources, serving = inputs()
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-32B",
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    minimum = next(candidate for candidate in candidates if "minimum-runnable" in candidate.tiers)
    two_gcd = next(
        candidate
        for candidate in candidates
        if candidate.plan.partition == "small-g" and candidate.plan.tensor_parallel == 2
    )

    assert minimum.plan.tensor_parallel == 4
    assert minimum.plan.allocated_gcds == 4
    assert two_gcd.analysis.feasibility is FeasibilityStatus.INFEASIBLE
    assert any(
        constraint.constraint_id == "physical-hbm" and not constraint.passed
        for constraint in two_gcd.analysis.constraints
    )


def test_multi_node_plan_is_conditional_without_validation_benchmark() -> None:
    catalogs, resources, serving = inputs()
    resource_payload = resources.model_dump(mode="json")
    resource_payload["maximum_gpu_hours"] = "100"
    expanded_budget = ResourceEnvelope.model_validate(resource_payload)
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-32B",
        serving=serving,
        resources=expanded_budget,
        catalogs=catalogs,
    )
    multi_node = next(
        candidate
        for candidate in candidates
        if candidate.plan.partition == "small-g" and candidate.plan.tensor_parallel == 16
    )

    assert multi_node.plan.nodes == 2
    assert multi_node.analysis.feasibility is FeasibilityStatus.CONDITIONAL


def test_conditional_only_search_is_not_labeled_runnable() -> None:
    catalogs, resources, serving = inputs()
    resource_payload = resources.model_dump(mode="json")
    resource_payload["nodes"] = 2
    resource_payload["gcds"] = 16
    resource_payload["maximum_gpu_hours"] = "100"
    multi_node_only = ResourceEnvelope.model_validate(resource_payload)
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-32B",
        serving=serving,
        resources=multi_node_only,
        catalogs=catalogs,
    )

    assert not any("minimum-runnable" in candidate.tiers for candidate in candidates)
    closest = next(
        candidate for candidate in candidates if "closest-conditional" in candidate.tiers
    )
    assert closest.analysis.feasibility is FeasibilityStatus.CONDITIONAL


def test_bundle_fingerprint_is_stable_when_timestamp_changes() -> None:
    catalogs, resources, serving = inputs()
    selector = load_model_selector(ROOT / "examples/model-qwen3-32b.yaml")
    first = build_model_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
        generated_at=datetime(2026, 8, 9, 8, tzinfo=UTC),
    )
    second = build_model_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
        generated_at=datetime(2026, 8, 9, 9, tzinfo=UTC),
    )

    assert first.analysis_fingerprint == second.analysis_fingerprint
    assert first.generated_at != second.generated_at


def test_context_above_model_limit_is_infeasible() -> None:
    catalogs, resources, serving = inputs()
    serving_payload = serving.model_dump(mode="json")
    serving_payload["max_model_length"] = 65_536
    oversized = ServingRequirements.model_validate(serving_payload)
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-8B",
        serving=oversized,
        resources=resources,
        catalogs=catalogs,
    )

    assert candidates
    assert all(
        candidate.analysis.feasibility is FeasibilityStatus.INFEASIBLE
        for candidate in candidates
    )


def test_minimum_plan_billing_matches_small_g_rate() -> None:
    catalogs, resources, serving = inputs()
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-8B",
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    minimum = next(candidate for candidate in candidates if "minimum-runnable" in candidate.tiers)

    assert minimum.analysis.billing is not None
    assert minimum.analysis.billing.gpu_hours_per_wallclock_hour == Decimal("0.5")
    assert minimum.analysis.billing.total_gpu_hours == Decimal("6")
