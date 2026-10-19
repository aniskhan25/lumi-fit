"""Exact evaluation of versioned LUMI GPU billing rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from lumi_fit.domain.catalogs import BillingRule, BillingRuleType
from lumi_fit.domain.units import Duration, MemoryAmount


@dataclass(frozen=True)
class BillingCalculation:
    rule_id: str
    gpu_hours_per_wallclock_hour: Decimal
    duration_hours: Decimal
    total_gpu_hours: Decimal
    dominant_resource: str
    billed_slices: int


def _ceiling_ratio(numerator: int, denominator: int) -> int:
    return int((Decimal(numerator) / Decimal(denominator)).to_integral_value(ROUND_CEILING))


def calculate_gpu_billing(
    rule: BillingRule,
    *,
    allocated_nodes: int,
    allocated_gcds: int,
    allocated_cpu_cores: int,
    allocated_cpu_memory: MemoryAmount,
    duration: Duration,
) -> BillingCalculation:
    """Calculate LUMI GPU-hours without rounding away fractional job duration.

    Resource counts and CPU memory are totals for the complete allocation, not
    per-node values. The caller is responsible for first validating partition
    limits and node-level physical capacity.
    """

    if min(allocated_nodes, allocated_gcds, allocated_cpu_cores) < 1:
        raise ValueError("allocated nodes, GCDs, and CPU cores must be positive")

    if rule.type is BillingRuleType.FULL_NODE_GPU:
        rate = rule.billed_gpu_hours_per_node_hour
        if rate is None:
            raise ValueError("invalid full-node billing rule")
        per_hour = rate * allocated_nodes
        billed_slices = allocated_nodes
        dominant_resource = "nodes"
    else:
        gcd_rate = rule.billed_gpu_hours_per_gcd_hour
        cpu_slice_size = rule.cpu_cores_per_billing_slice
        memory_slice_size = rule.memory_per_billing_slice
        if gcd_rate is None or cpu_slice_size is None or memory_slice_size is None:
            raise ValueError("invalid subnode billing rule")

        resource_slices = {
            "gcds": allocated_gcds,
            "cpu-cores": _ceiling_ratio(allocated_cpu_cores, cpu_slice_size),
            "cpu-memory": _ceiling_ratio(
                allocated_cpu_memory.bytes,
                memory_slice_size.bytes,
            ),
        }
        dominant_resource, billed_slices = max(
            resource_slices.items(), key=lambda item: (item[1], item[0])
        )
        per_hour = gcd_rate * billed_slices

    total = per_hour * duration.hours
    return BillingCalculation(
        rule_id=rule.rule_id,
        gpu_hours_per_wallclock_hour=per_hour,
        duration_hours=duration.hours,
        total_gpu_hours=total,
        dominant_resource=dominant_resource,
        billed_slices=billed_slices,
    )
