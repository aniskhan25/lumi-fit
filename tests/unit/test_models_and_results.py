from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lumi_fit.domain.models import (
    ArtifactFormat,
    ArtifactRecord,
    ModelCatalog,
    ModelRecord,
    TransformerArchitecture,
    WeightDtype,
)
from lumi_fit.domain.plans import ContainerIdentity, RuntimeOptions, ServingPlan
from lumi_fit.domain.results import (
    ConfidenceClass,
    ConstraintResult,
    ConstraintSeverity,
    FeasibilityStatus,
    FitBand,
    PlanAnalysis,
)


def architecture() -> TransformerArchitecture:
    return TransformerArchitecture(
        architecture_name="ExampleForCausalLM",
        total_parameters=7_000_000_000,
        layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        attention_heads=32,
        kv_heads=8,
        head_dimension=128,
        vocabulary_size=128_000,
        context_length=32_768,
        tie_word_embeddings=False,
    )


def serving_plan() -> ServingPlan:
    return ServingPlan(
        model_id="example/model-7b",
        model_revision="abc123",
        artifact_id="example-model-7b-bf16",
        quantization="bf16",
        runtime_id="vllm-laif",
        container=ContainerIdentity(
            family="lumi-multitorch-full",
            path="/appl/local/laifs/containers/example.sif",
        ),
        runtime_options=RuntimeOptions(
            max_model_length=8192,
            max_concurrent_sequences=8,
            max_batched_tokens=8192,
            gpu_memory_utilization="0.85",
            kv_cache_dtype="auto",
        ),
        partition="small-g",
        nodes=1,
        gcds_per_node=2,
        cpu_cores_per_node=16,
        cpu_memory_per_node="128 GB",
        tensor_parallel=2,
        pipeline_parallel=1,
        replicas=1,
        ranks_per_node=2,
        threads_per_rank=8,
        walltime="8 hours",
    )


def test_model_catalog_links_artifacts_to_model_revision() -> None:
    model = ModelRecord(
        model_id="example/model-7b",
        revision="abc123",
        family="example",
        architecture=architecture(),
        source_url="https://huggingface.co/example/model-7b",
    )
    artifact = ArtifactRecord(
        artifact_id="example-model-7b-bf16",
        model_id=model.model_id,
        revision=model.revision,
        format=ArtifactFormat.SAFETENSORS,
        weight_dtype=WeightDtype.BF16,
        quantization="bf16",
        total_bytes=14_000_000_000,
        shard_bytes=(7_000_000_000, 7_000_000_000),
        compatible_runtime_ids=("vllm-laif",),
        source_url="https://huggingface.co/example/model-7b/tree/abc123",
    )

    catalog = ModelCatalog.model_validate(
        {
            "metadata": {
                "profile_version": "test.1",
                "valid_from": "2026-08-09",
                "last_verified_at": datetime.now(UTC).isoformat(),
                "sources": [
                    {
                        "title": "fixture",
                        "path": "tests/unit/test_models_and_results.py",
                    }
                ],
            },
            "models": [model],
            "artifacts": [artifact],
        }
    )

    assert catalog.artifacts[0].total_bytes == 14_000_000_000


def test_artifact_rejects_inconsistent_shard_sizes() -> None:
    with pytest.raises(ValidationError, match="must sum"):
        ArtifactRecord(
            artifact_id="bad",
            model_id="example/model",
            revision="abc",
            format=ArtifactFormat.SAFETENSORS,
            weight_dtype=WeightDtype.BF16,
            quantization="bf16",
            total_bytes=10,
            shard_bytes=(4, 4),
            compatible_runtime_ids=("vllm-laif",),
            source_url="https://example.invalid",
        )


def test_serving_plan_validates_parallelism_layout() -> None:
    plan = serving_plan()

    assert plan.allocated_gcds == 2
    assert plan.used_gcds == 2


def test_serving_plan_rejects_layout_larger_than_allocation() -> None:
    payload = serving_plan().model_dump()
    payload["replicas"] = 2

    with pytest.raises(ValidationError, match="exceeds allocated GCDs"):
        ServingPlan.model_validate(payload)


def test_feasible_analysis_requires_fit_band() -> None:
    with pytest.raises(ValidationError, match="requires a fit band"):
        PlanAnalysis(
            feasibility=FeasibilityStatus.FEASIBLE,
            constraints=(),
            overall_confidence=ConfidenceClass.ANALYTICAL,
        )


def test_infeasible_analysis_requires_failed_constraint() -> None:
    passed = ConstraintResult(
        constraint_id="memory",
        severity=ConstraintSeverity.ERROR,
        passed=True,
        message="memory fits",
    )

    with pytest.raises(ValidationError, match="requires a failed constraint"):
        PlanAnalysis(
            feasibility=FeasibilityStatus.INFEASIBLE,
            constraints=(passed,),
            overall_confidence=ConfidenceClass.ANALYTICAL,
        )


def test_feasible_analysis_accepts_fit_band() -> None:
    analysis = PlanAnalysis(
        feasibility=FeasibilityStatus.FEASIBLE,
        fit_band=FitBand.GOOD,
        constraints=(),
        overall_confidence=ConfidenceClass.ANALYTICAL,
    )

    assert analysis.fit_band is FitBand.GOOD
