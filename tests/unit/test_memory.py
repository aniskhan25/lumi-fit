from pathlib import Path

from lumi_fit.analysis.memory import estimate_serving_memory
from lumi_fit.catalog.loader import load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.planning.model_first import generate_model_first_candidates

ROOT = Path(__file__).resolve().parents[2]


def test_memory_estimate_is_componentized_and_per_rank() -> None:
    catalogs = load_default_catalogs(ROOT)
    resources = load_resources(ROOT / "examples/resources-model-first.yaml")
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    candidate = generate_model_first_candidates(
        model_id="Qwen/Qwen3-8B",
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )[0]
    model = next(
        model for model in catalogs.models.models if model.model_id == candidate.plan.model_id
    )
    artifact = next(
        artifact
        for artifact in catalogs.models.artifacts
        if artifact.artifact_id == candidate.plan.artifact_id
    )

    memory = estimate_serving_memory(
        model=model,
        artifact=artifact,
        plan=candidate.plan,
        serving=serving,
        hardware=catalogs.hardware,
    )

    assert {component.component for component in memory.components} == {
        "weights",
        "kv-cache",
        "activations",
        "runtime-workspace",
        "communication",
        "allocator-reserve",
    }
    assert memory.peak_per_rank.upper <= memory.physical_hbm_per_rank_bytes
    assert memory.conservative_utilization_fraction > memory.expected_utilization_fraction


def test_more_tensor_parallelism_reduces_weight_shard() -> None:
    catalogs = load_default_catalogs(ROOT)
    resources = load_resources(ROOT / "examples/resources-model-first.yaml")
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    candidates = generate_model_first_candidates(
        model_id="Qwen/Qwen3-32B",
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    small = next(
        candidate
        for candidate in candidates
        if candidate.plan.partition == "small-g" and candidate.plan.tensor_parallel == 2
    )
    large = next(
        candidate
        for candidate in candidates
        if candidate.plan.partition == "small-g" and candidate.plan.tensor_parallel == 4
    )

    def weight_bytes(candidate):
        return next(
            component.per_rank.upper
            for component in candidate.analysis.memory.components
            if component.component == "weights"
        )

    assert weight_bytes(large) < weight_bytes(small)
