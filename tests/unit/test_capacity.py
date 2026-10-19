from pathlib import Path

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.requests import ServingRequirements
from lumi_fit.planning.model_first import build_model_first_bundle

ROOT = Path(__file__).resolve().parents[2]


def _bundle(*, concurrency: int = 8):
    catalogs = load_default_catalogs(ROOT)
    serving = load_serving(ROOT / "examples/serving-online.yaml")
    if concurrency != serving.concurrent_sequences:
        payload = serving.model_dump(mode="json")
        payload["concurrent_sequences"] = concurrency
        serving = ServingRequirements.model_validate(payload)
    return build_model_first_bundle(
        model_selector=load_model_selector(ROOT / "examples/model-qwen3-32b.yaml"),
        serving=serving,
        resources=load_resources(ROOT / "examples/resources-model-first.yaml"),
        catalogs=catalogs,
    )


def _recommended(bundle):
    return next(candidate for candidate in bundle.candidates if "recommended" in candidate.tiers)


def test_capacity_envelope_exposes_requested_and_memory_safe_boundaries() -> None:
    candidate = _recommended(_bundle())
    envelope = candidate.capacity_envelope

    assert envelope is not None
    assert envelope.requested.context_tokens == 16_384
    assert envelope.requested.total_concurrent_sequences == 8
    assert envelope.requested.concurrent_sequences_per_replica == 8
    assert envelope.requested.memory_safe
    assert envelope.memory_safe_maximum_concurrency is not None
    assert envelope.memory_safe_maximum_concurrency.total_concurrent_sequences == 21
    assert envelope.memory_safe_maximum_context is not None
    assert envelope.memory_safe_maximum_context.context_tokens == 40_960
    assert envelope.catalog_context_limit == 40_960
    assert envelope.analytical_performance_maximum_concurrency is None
    assert "Not estimated" in envelope.analytical_performance_explanation


def test_concurrency_binary_search_persists_the_first_unsafe_boundary() -> None:
    envelope = _recommended(_bundle()).capacity_envelope
    assert envelope is not None
    maximum = envelope.memory_safe_maximum_concurrency
    assert maximum is not None
    probes = {
        (point.context_tokens, point.total_concurrent_sequences): point
        for point in envelope.evaluated_points
    }
    first_unsafe = probes[(maximum.context_tokens, maximum.total_concurrent_sequences + 1)]

    assert maximum.memory_safe
    assert not first_unsafe.memory_safe


def test_context_binary_search_persists_the_first_unsafe_boundary() -> None:
    envelope = _recommended(_bundle(concurrency=12)).capacity_envelope
    assert envelope is not None
    maximum = envelope.memory_safe_maximum_context
    assert maximum is not None
    assert maximum.context_tokens < envelope.catalog_context_limit
    probes = {
        (point.context_tokens, point.total_concurrent_sequences): point
        for point in envelope.evaluated_points
    }
    first_unsafe = probes[(maximum.context_tokens + 1, maximum.total_concurrent_sequences)]

    assert maximum.memory_safe
    assert not first_unsafe.memory_safe


def test_evaluated_capacity_points_are_monotonic_on_each_search_axis() -> None:
    envelope = _recommended(_bundle(concurrency=12)).capacity_envelope
    assert envelope is not None

    by_context = sorted(
        (
            point
            for point in envelope.evaluated_points
            if point.context_tokens == envelope.requested.context_tokens
        ),
        key=lambda point: point.total_concurrent_sequences,
    )
    by_concurrency = sorted(
        (
            point
            for point in envelope.evaluated_points
            if point.total_concurrent_sequences == envelope.requested.total_concurrent_sequences
        ),
        key=lambda point: point.context_tokens,
    )

    for points in (by_context, by_concurrency):
        safe_flags = [point.memory_safe for point in points]
        assert safe_flags == sorted(safe_flags, reverse=True)


def test_capacity_envelopes_reference_existing_lower_and_higher_neighbors() -> None:
    bundle = _bundle()
    recommended = _recommended(bundle)
    throughput = next(
        candidate for candidate in bundle.candidates if "throughput-optimized" in candidate.tiers
    )
    assert recommended.capacity_envelope is not None
    assert throughput.capacity_envelope is not None
    higher = recommended.capacity_envelope.higher_capacity_neighbor
    lower = throughput.capacity_envelope.lower_cost_neighbor

    assert higher is not None
    assert lower is not None
    assert higher.plan_id in {candidate.plan_id for candidate in bundle.candidates}
    assert lower.plan_id == recommended.plan_id
    assert (
        higher.memory_safe_maximum_concurrency
        > recommended.capacity_envelope.memory_safe_maximum_concurrency.total_concurrent_sequences
    )
    assert (
        lower.gpu_hours_per_wallclock_hour
        < throughput.analysis.billing.gpu_hours_per_wallclock_hour
    )
