from decimal import Decimal

import pytest
from pydantic import ValidationError

from lumi_fit.domain.units import CountRange, Duration, MemoryAmount, MemoryUnit


def test_memory_preserves_decimal_gb_and_converts_to_bytes() -> None:
    amount = MemoryAmount.model_validate("64 GB")

    assert amount.unit is MemoryUnit.GB
    assert amount.bytes == 64_000_000_000
    assert amount.as_unit(MemoryUnit.GIB) < Decimal(64)


def test_memory_preserves_binary_gib() -> None:
    amount = MemoryAmount.model_validate("64 GiB")

    assert amount.unit is MemoryUnit.GIB
    assert amount.bytes == 68_719_476_736


def test_memory_rejects_ambiguous_unitless_input() -> None:
    with pytest.raises(ValidationError, match="explicit unit"):
        MemoryAmount.model_validate("64")


def test_duration_parses_short_alias() -> None:
    duration = Duration.model_validate("2h")

    assert duration.hours == Decimal(2)


def test_fixed_count_normalizes_to_range() -> None:
    count = CountRange.model_validate(4)

    assert count.minimum == 4
    assert count.maximum == 4
    assert count.fixed == 4


def test_count_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="maximum count"):
        CountRange(minimum=8, maximum=4)
