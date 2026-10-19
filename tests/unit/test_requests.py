from pathlib import Path

import pytest
from pydantic import ValidationError

from lumi_fit.catalog.loader import load_model_selector, load_resources, load_serving
from lumi_fit.domain.requests import ModelSelector, ServingMode

ROOT = Path(__file__).resolve().parents[2]


def test_resource_first_example_is_fixed_and_single_node() -> None:
    resources = load_resources(ROOT / "examples/resources-one-node-four-gcds.yaml")

    assert resources.nodes.fixed == 1
    assert resources.gcds.fixed == 4
    assert resources.cpu_memory_per_node is not None
    assert resources.cpu_memory_per_node.bytes == 256_000_000_000


def test_model_first_resource_bounds_allow_two_nodes() -> None:
    resources = load_resources(ROOT / "examples/resources-model-first.yaml")

    assert resources.nodes.minimum == 1
    assert resources.nodes.maximum == 2
    assert resources.gcds.maximum == 16
    assert resources.allow_multi_node


def test_online_serving_example_has_duration() -> None:
    serving = load_serving(ROOT / "examples/serving-online.yaml")

    assert serving.mode is ServingMode.ONLINE
    assert serving.service_duration is not None
    assert serving.service_duration.hours == 8
    assert len(serving.token_length_distribution) == 1
    point = serving.token_length_distribution[0]
    assert point.input_tokens == serving.input_tokens
    assert point.output_tokens == serving.max_output_tokens
    assert point.probability == 1


def test_batch_serving_example_has_work_volume() -> None:
    serving = load_serving(ROOT / "examples/serving-batch.yaml")

    assert serving.mode is ServingMode.BATCH
    assert serving.batch_request_count == 100_000


def test_online_serving_requires_duration() -> None:
    with pytest.raises(ValidationError, match="requires service_duration"):
        load_serving_data = {
            "mode": "online",
            "input_tokens": 10,
            "max_output_tokens": 10,
            "max_model_length": 20,
            "concurrent_sequences": 1,
        }
        from lumi_fit.domain.requests import ServingRequirements

        ServingRequirements.model_validate(load_serving_data)


def test_model_selector_examples_validate() -> None:
    exact = load_model_selector(ROOT / "examples/model-qwen3-32b.yaml")
    filtered = load_model_selector(ROOT / "examples/models-coding.yaml")

    assert exact.exact_model == "Qwen/Qwen3-32B"
    assert filtered.use_case is not None


def test_model_selector_rejects_inverted_parameter_range() -> None:
    with pytest.raises(ValidationError, match="minimum_parameters cannot exceed"):
        ModelSelector(minimum_parameters=10, maximum_parameters=9)
