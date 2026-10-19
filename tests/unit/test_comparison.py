from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from lumi_fit.catalog.loader import load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.models import ModelEvaluation
from lumi_fit.domain.requests import ModelSelector, ServingRequirements
from lumi_fit.planning.resource_first import build_resource_first_bundle

ROOT = Path(__file__).resolve().parents[2]


def _bundle(serving: ServingRequirements | None = None):
    return build_resource_first_bundle(
        model_selector=ModelSelector(include_patterns=("Qwen/*",), use_case="coding"),
        serving=serving or load_serving(ROOT / "examples/serving-online.yaml"),
        resources=load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml"),
        catalogs=load_default_catalogs(ROOT),
    )


def test_weighted_objective_sensitivity_is_not_part_of_fit_scoring() -> None:
    bundle = _bundle()

    assert bundle.objective_sensitivity == ()
    assert bundle.comparisons == ()
    assert all(candidate.analysis.composite_score is None for candidate in bundle.candidates)
    assert all(candidate.analysis.scores == () for candidate in bundle.candidates)


def test_legacy_objective_profile_does_not_change_recommendation_order() -> None:
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    payload = serving.model_dump(mode="json")
    payload["objective"] = "budget"
    budget = ServingRequirements.model_validate(payload)

    original = _bundle(serving)
    changed = _bundle(budget)

    assert [group.best_plan_id for group in original.model_groups] == [
        group.best_plan_id for group in changed.model_groups
    ]


def test_task_quality_is_reported_separately_from_lumi_fit() -> None:
    catalogs = load_default_catalogs(ROOT)
    model_id = "Qwen/Qwen3-8B"
    models = tuple(
        model.model_copy(
            update={
                "evaluations": (
                    ModelEvaluation(
                        evaluation_id="coding-quality-v1",
                        use_case="coding",
                        score=Decimal("0.80"),
                        score_minimum=0,
                        score_maximum=1,
                        source_url="https://example.invalid/coding-quality-v1",
                    ),
                )
            }
        )
        if model.model_id == model_id
        else model
        for model in catalogs.models.models
    )
    catalogs = replace(catalogs, models=catalogs.models.model_copy(update={"models": models}))
    bundle = build_resource_first_bundle(
        model_selector=ModelSelector(exact_model=model_id, use_case="coding"),
        serving=load_serving(ROOT / "examples/serving-online.yaml"),
        resources=load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml"),
        catalogs=catalogs,
    )
    candidate = bundle.candidates[0]

    assert candidate.lumi_vllm_fit is not None
    assert candidate.lumi_vllm_fit.model_quality == Decimal(80)
    assert candidate.lumi_vllm_fit.fit_score is None
    assert candidate.analysis.composite_score is None
