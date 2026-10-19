"""Shared Pydantic configuration for strict, immutable contracts."""

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base class for all persisted public contracts.

    Extra fields are rejected so misspelled resource settings cannot silently
    alter a recommendation. Models are immutable after validation so their
    normalized representation is safe to fingerprint.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        validate_default=True,
    )
