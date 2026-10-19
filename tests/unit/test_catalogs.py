from pathlib import Path

from lumi_fit.catalog.validation import (
    assert_valid_catalogs,
    load_default_catalogs,
    validate_catalog_references,
)
from lumi_fit.domain.catalogs import AllocationGranularity, SupportLevel
from lumi_fit.domain.units import MemoryUnit


ROOT = Path(__file__).resolve().parents[2]


def test_default_catalogs_validate_together() -> None:
    catalogs = load_default_catalogs(ROOT)

    assert validate_catalog_references(catalogs) == ()
    assert_valid_catalogs(catalogs)


def test_lumi_g_profile_uses_gcd_units() -> None:
    hardware = load_default_catalogs(ROOT).hardware

    assert hardware.gcd.gcds_per_module == 2
    assert hardware.gcd.modules_per_node == 4
    assert hardware.gcd.gcds_per_node == 8
    assert hardware.gcd.hbm_per_gcd.unit is MemoryUnit.GB
    assert hardware.cpu.schedulable_cores == 56


def test_partitions_reference_expected_allocation_granularity() -> None:
    partitions = {
        profile.partition_id: profile
        for profile in load_default_catalogs(ROOT).partitions.partitions
    }

    assert partitions["small-g"].allocation_granularity is AllocationGranularity.RESOURCE
    assert partitions["standard-g"].allocation_granularity is AllocationGranularity.FULL_NODE
    assert partitions["standard-g"].exclusive
    assert not partitions["bench"].default_enabled


def test_vllm_runtime_is_conservative_about_multi_node_support() -> None:
    runtime = load_default_catalogs(ROOT).runtimes.runtimes[0]

    assert runtime.runtime_id == "vllm-laif"
    assert runtime.tensor_parallel is SupportLevel.SUPPORTED
    assert runtime.multi_node is SupportLevel.CONDITIONAL


def test_scoring_profiles_sum_to_one() -> None:
    scoring = load_default_catalogs(ROOT).scoring

    assert {profile.profile_id for profile in scoring.profiles} == {
        "balanced",
        "interactive",
        "throughput",
        "budget",
        "quality-first",
    }


def test_phase4a_model_catalog_contains_representative_verified_artifacts() -> None:
    models = load_default_catalogs(ROOT).models

    assert {model.model_id for model in models.models} == {
        "microsoft/Phi-4-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "ibm-granite/granite-3.1-8b-instruct",
        "allenai/OLMo-2-1124-13B-Instruct",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    }
    assert all(artifact.total_bytes for artifact in models.artifacts)
