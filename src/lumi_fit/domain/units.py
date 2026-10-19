"""Explicit resource units and ranges.

LUMI documentation uses both decimal GB and binary GiB. These types preserve
the source unit while exposing an exact canonical byte count.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .base import ContractModel


class MemoryUnit(StrEnum):
    MB = "MB"
    GB = "GB"
    TB = "TB"
    MIB = "MiB"
    GIB = "GiB"
    TIB = "TiB"

    @property
    def bytes_per_unit(self) -> int:
        return {
            MemoryUnit.MB: 1_000_000,
            MemoryUnit.GB: 1_000_000_000,
            MemoryUnit.TB: 1_000_000_000_000,
            MemoryUnit.MIB: 1_048_576,
            MemoryUnit.GIB: 1_073_741_824,
            MemoryUnit.TIB: 1_099_511_627_776,
        }[self]


_MEMORY_PATTERN = re.compile(
    r"^\s*(?P<value>(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*"
    r"(?P<unit>MB|GB|TB|MiB|GiB|TiB)\s*$"
)


class MemoryAmount(ContractModel):
    value: Decimal = Field(gt=0)
    unit: MemoryUnit

    @model_validator(mode="before")
    @classmethod
    def parse_string(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        match = _MEMORY_PATTERN.match(value)
        if match is None:
            raise ValueError("memory must include an explicit unit, for example '64 GB'")
        return {"value": match.group("value"), "unit": match.group("unit")}

    @property
    def bytes(self) -> int:
        return int(self.value * self.unit.bytes_per_unit)

    def as_unit(self, unit: MemoryUnit) -> Decimal:
        return Decimal(self.bytes) / Decimal(unit.bytes_per_unit)

    def __str__(self) -> str:
        return f"{self.value} {self.unit.value}"


class DurationUnit(StrEnum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

    @property
    def seconds_per_unit(self) -> int:
        return {
            DurationUnit.SECONDS: 1,
            DurationUnit.MINUTES: 60,
            DurationUnit.HOURS: 3_600,
            DurationUnit.DAYS: 86_400,
        }[self]


_DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*"
    r"(?P<unit>seconds?|minutes?|hours?|days?|s|m|h|d)\s*$",
    re.IGNORECASE,
)


class Duration(ContractModel):
    value: Decimal = Field(gt=0)
    unit: DurationUnit

    @model_validator(mode="before")
    @classmethod
    def parse_string(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        match = _DURATION_PATTERN.match(value)
        if match is None:
            raise ValueError("duration must include a unit, for example '8 hours'")
        aliases = {
            "s": DurationUnit.SECONDS,
            "second": DurationUnit.SECONDS,
            "seconds": DurationUnit.SECONDS,
            "m": DurationUnit.MINUTES,
            "minute": DurationUnit.MINUTES,
            "minutes": DurationUnit.MINUTES,
            "h": DurationUnit.HOURS,
            "hour": DurationUnit.HOURS,
            "hours": DurationUnit.HOURS,
            "d": DurationUnit.DAYS,
            "day": DurationUnit.DAYS,
            "days": DurationUnit.DAYS,
        }
        return {
            "value": match.group("value"),
            "unit": aliases[match.group("unit").lower()],
        }

    @property
    def seconds(self) -> Decimal:
        return self.value * self.unit.seconds_per_unit

    @property
    def hours(self) -> Decimal:
        return self.seconds / Decimal(3_600)

    def __str__(self) -> str:
        return f"{self.value} {self.unit.value}"


class CountRange(ContractModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def parse_fixed_count(cls, value: Any) -> Any:
        if isinstance(value, int):
            return {"minimum": value, "maximum": value}
        return value

    @model_validator(mode="after")
    def validate_order(self) -> CountRange:
        if self.maximum < self.minimum:
            raise ValueError("maximum count must be greater than or equal to minimum")
        return self

    @property
    def is_fixed(self) -> bool:
        return self.minimum == self.maximum

    @property
    def fixed(self) -> int | None:
        return self.minimum if self.is_fixed else None
