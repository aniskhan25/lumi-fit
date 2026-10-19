from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.models import ModelEvaluation
from lumi_fit.domain.requests import ModelSelector, ResourceEnvelope
from lumi_fit.domain.results import ConfidenceClass, FeasibilityStatus
from lumi_fit.planning.resource_first import (
    build_resource_first_bundle,
    select_models,
)

ROOT = Path(__file__).resolve().parents[2]


def inputs():
    return (
        load_default_catalogs(ROOT),
        load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml"),
        load_serving(ROOT / "examples/serving-online.yaml"),
        load_model_selector(ROOT / "examples/models-coding.yaml"),
    )


def test_selector_applies_use_case_include_and_exclude_patterns() -> None:
    catalogs, _, _, _ = inputs()
    selector = ModelSelector(
        include_patterns=("Qwen/*",),
        exclude_patterns=("*-32B",),
        use_case="coding",
    )

    selected = select_models(selector=selector, catalogs=catalogs)

    assert [model.model_id for model in selected] == [
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-8B",
    ]


def test_resource_first_requires_a_fixed_allocation() -> None:
    catalogs, resources, serving, selector = inputs()
    payload = resources.model_dump(mode="json")
    payload["gcds"] = {"minimum": 1, "maximum": 4}
    ranged = ResourceEnvelope.model_validate(payload)

    with pytest.raises(ValueError, match="fixed nodes and fixed GCDs"):
        build_resource_first_bundle(
            model_selector=selector,
            serving=serving,
            resources=ranged,
            catalogs=catalogs,
        )


def test_four_gcds_returns_a_result_per_cataloged_coding_model() -> None:
    catalogs, resources, serving, selector = inputs()

    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )

    assert {group.model_id for group in bundle.model_groups} == {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
    }
    assert all(
        group.best_plan_id is not None or group.diagnostic_plan_id is not None
        for group in bundle.model_groups
    )
    assert all(
        group.feasible_plan_count > 0
        for group in bundle.model_groups
        if group.model_id != "Qwen/Qwen3-235B-A22B"
    )


def test_selector_applies_model_metadata_filters() -> None:
    catalogs, _, _, _ = inputs()

    selected = select_models(
        selector=ModelSelector(
            capabilities=("coding",),
            families=("phi3",),
            licenses=("MIT",),
            languages=("fi",),
            maximum_parameters=4_000_000_000,
        ),
        catalogs=catalogs,
    )

    assert [model.model_id for model in selected] == ["microsoft/Phi-4-mini-instruct"]


@pytest.mark.parametrize("gcd_count", (1, 2, 4, 8))
def test_common_single_node_allocations_produce_a_useful_recommendation(
    gcd_count: int,
) -> None:
    catalogs, resources, serving, selector = inputs()
    payload = resources.model_dump(mode="json")
    payload["gcds"] = gcd_count
    payload["maximum_gpu_hours"] = 32

    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=ResourceEnvelope.model_validate(payload),
        catalogs=catalogs,
    )

    best_ids = {group.best_plan_id for group in bundle.model_groups if group.best_plan_id}
    assert best_ids
    assert all(
        candidate.capacity_envelope is not None
        for candidate in bundle.candidates
        if candidate.plan_id in best_ids
    )


def test_legacy_objective_does_not_change_goodput_fit_ranking() -> None:
    catalogs, resources, serving, selector = inputs()
    interactive = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    payload = serving.model_dump(mode="json")
    payload["objective"] = "throughput"
    payload["latency_targets"] = {}
    throughput_serving = type(serving).model_validate(payload)
    throughput = build_resource_first_bundle(
        model_selector=selector,
        serving=throughput_serving,
        resources=resources,
        catalogs=catalogs,
    )

    assert [group.best_plan_id for group in interactive.model_groups] == [
        group.best_plan_id for group in throughput.model_groups
    ]


def test_large_model_prefers_safe_tp4_over_marginal_two_replica_layout() -> None:
    catalogs, resources, serving, selector = inputs()
    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    group = next(group for group in bundle.model_groups if group.model_id.endswith("32B"))
    best = next(
        candidate for candidate in bundle.candidates if candidate.plan_id == group.best_plan_id
    )
    two_replicas = next(
        candidate
        for candidate in bundle.candidates
        if candidate.plan.model_id.endswith("32B")
        and candidate.plan.tensor_parallel == 2
        and candidate.plan.replicas == 2
    )

    assert best.plan.tensor_parallel == 4
    assert best.plan.replicas == 1
    assert best.analysis.fit_band.value == "excellent"
    assert two_replicas.analysis.fit_band.value == "marginal"
    assert two_replicas.plan.runtime_options.max_concurrent_sequences == 4
    assert two_replicas.plan_id in group.alternative_plan_ids


def test_unrunnable_model_has_diagnostic_but_no_best_plan() -> None:
    catalogs, resources, serving, _ = inputs()
    payload = resources.model_dump(mode="json")
    payload["gcds"] = 1
    payload["cpu_cores_per_node"] = 8
    payload["cpu_memory_per_node"] = "64 GB"
    payload["maximum_gpu_hours"] = "4"
    one_gcd = ResourceEnvelope.model_validate(payload)
    selector = ModelSelector(exact_model="Qwen/Qwen3-32B")

    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=one_gcd,
        catalogs=catalogs,
    )
    group = bundle.model_groups[0]
    diagnostic = next(
        candidate
        for candidate in bundle.candidates
        if candidate.plan_id == group.diagnostic_plan_id
    )

    assert group.best_plan_id is None
    assert group.feasible_plan_count == 0
    assert diagnostic.analysis.feasibility is FeasibilityStatus.INFEASIBLE
    assert "closest-infeasible" in diagnostic.tiers


def test_missing_quality_evidence_is_unknown_not_zero() -> None:
    catalogs, resources, serving, selector = inputs()
    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )

    assert all(group.suitability.score is None for group in bundle.model_groups)
    assert all(
        group.suitability.confidence is ConfidenceClass.UNKNOWN for group in bundle.model_groups
    )


def test_cataloged_quality_evidence_is_normalized_and_linked() -> None:
    catalogs, resources, serving, _ = inputs()
    models = tuple(
        model.model_copy(
            update={
                "evaluations": (
                    ModelEvaluation(
                        evaluation_id="coding-eval-v1",
                        use_case="coding",
                        score="0.75",
                        score_minimum="0",
                        score_maximum="1",
                        source_url="https://example.invalid/coding-eval-v1",
                    ),
                )
            }
        )
        if model.model_id == "Qwen/Qwen3-8B"
        else model
        for model in catalogs.models.models
    )
    catalogs = replace(
        catalogs,
        models=catalogs.models.model_copy(update={"models": models}),
    )

    bundle = build_resource_first_bundle(
        model_selector=ModelSelector(exact_model="Qwen/Qwen3-8B", use_case="coding"),
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    suitability = bundle.model_groups[0].suitability

    assert suitability.score == Decimal("75")
    assert suitability.evidence_ids == ("coding-eval-v1",)
    assert suitability.confidence is ConfidenceClass.MEASURED_NEAR


def test_bundle_fingerprint_does_not_include_timestamp() -> None:
    catalogs, resources, serving, selector = inputs()
    first = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
        generated_at=datetime(2026, 8, 9, 8, tzinfo=UTC),
    )
    second = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=resources,
        catalogs=catalogs,
        generated_at=datetime(2026, 8, 9, 9, tzinfo=UTC),
    )

    assert first.analysis_fingerprint == second.analysis_fingerprint
