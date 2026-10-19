import csv
import io
from pathlib import Path

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.planning.resource_first import build_resource_first_bundle
from lumi_fit.presentation import (
    PresentationOptions,
    filter_and_sort_candidates,
    format_candidates_csv,
)


ROOT = Path(__file__).resolve().parents[2]


def _bundle():
    return build_resource_first_bundle(
        model_selector=load_model_selector(ROOT / "examples/models-coding.yaml"),
        serving=load_serving(ROOT / "examples/serving-online.yaml"),
        resources=load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml"),
        catalogs=load_default_catalogs(ROOT),
    )


def test_result_filters_and_parameter_sort_are_deterministic() -> None:
    candidates = filter_and_sort_candidates(
        _bundle(),
        PresentationOptions(
            statuses=("feasible",),
            fit_bands=("excellent",),
            partitions=("small-g",),
            precisions=("bf16",),
            sort_by="parameters",
        ),
    )

    assert [candidate.plan.model_id for candidate in candidates] == [
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-8B",
    ]


def test_expanded_view_contains_more_plans_than_compact_view() -> None:
    bundle = _bundle()

    compact = filter_and_sort_candidates(bundle)
    expanded = filter_and_sort_candidates(
        bundle,
        PresentationOptions(include_alternatives=True),
    )

    assert len(compact) == len(bundle.model_groups)
    assert len(expanded) == len(bundle.candidates)
    assert len(expanded) > len(compact)


def test_csv_export_is_deterministic_and_has_a_stable_column_order() -> None:
    candidates = filter_and_sort_candidates(
        _bundle(),
        PresentationOptions(statuses=("feasible",), sort_by="model"),
    )

    first = format_candidates_csv(candidates)
    second = format_candidates_csv(candidates)
    rows = list(csv.DictReader(io.StringIO(first)))

    assert first == second
    assert first.splitlines()[0].startswith("model_id,parameters,plan_id,partition")
    assert [row["model_id"] for row in rows] == sorted(row["model_id"] for row in rows)
