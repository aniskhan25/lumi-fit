from decimal import Decimal
from pathlib import Path

from lumi_fit.analysis.billing import calculate_gpu_billing
from lumi_fit.catalog.validation import load_default_catalogs
from lumi_fit.domain.units import Duration, MemoryAmount

ROOT = Path(__file__).resolve().parents[2]


def rules_by_id():
    return {rule.rule_id: rule for rule in load_default_catalogs(ROOT).billing.rules}


def test_standard_g_bills_four_modules_per_node_hour() -> None:
    result = calculate_gpu_billing(
        rules_by_id()["lumi-g-full-node"],
        allocated_nodes=4,
        allocated_gcds=32,
        allocated_cpu_cores=224,
        allocated_cpu_memory=MemoryAmount.model_validate("2 TiB"),
        duration=Duration.model_validate("24 hours"),
    )

    assert result.gpu_hours_per_wallclock_hour == Decimal(16)
    assert result.total_gpu_hours == Decimal(384)
    assert result.dominant_resource == "nodes"


def test_small_g_bills_half_gpu_hour_per_gcd_hour() -> None:
    result = calculate_gpu_billing(
        rules_by_id()["lumi-g-subnode"],
        allocated_nodes=1,
        allocated_gcds=2,
        allocated_cpu_cores=16,
        allocated_cpu_memory=MemoryAmount.model_validate("128 GB"),
        duration=Duration.model_validate("24 hours"),
    )

    assert result.gpu_hours_per_wallclock_hour == Decimal(1)
    assert result.total_gpu_hours == Decimal(24)


def test_subnode_memory_can_drive_billing() -> None:
    result = calculate_gpu_billing(
        rules_by_id()["lumi-g-subnode"],
        allocated_nodes=1,
        allocated_gcds=1,
        allocated_cpu_cores=8,
        allocated_cpu_memory=MemoryAmount.model_validate("128 GB"),
        duration=Duration.model_validate("24 hours"),
    )

    assert result.billed_slices == 2
    assert result.dominant_resource == "cpu-memory"
    assert result.total_gpu_hours == Decimal(24)


def test_subnode_cpu_rounds_up_to_complete_billing_slice() -> None:
    result = calculate_gpu_billing(
        rules_by_id()["lumi-g-subnode"],
        allocated_nodes=1,
        allocated_gcds=1,
        allocated_cpu_cores=17,
        allocated_cpu_memory=MemoryAmount.model_validate("64 GB"),
        duration=Duration.model_validate("1 hour"),
    )

    assert result.billed_slices == 3
    assert result.dominant_resource == "cpu-cores"
    assert result.total_gpu_hours == Decimal("1.5")


def test_subnode_duration_preserves_fractional_hours() -> None:
    result = calculate_gpu_billing(
        rules_by_id()["lumi-g-subnode"],
        allocated_nodes=1,
        allocated_gcds=1,
        allocated_cpu_cores=8,
        allocated_cpu_memory=MemoryAmount.model_validate("64 GB"),
        duration=Duration.model_validate("30 minutes"),
    )

    assert result.total_gpu_hours == Decimal("0.25")
