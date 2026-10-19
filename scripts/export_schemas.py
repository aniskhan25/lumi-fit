"""Export the public Pydantic contracts as deterministic JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path

from lumi_fit.domain.catalogs import (
    BillingCatalog,
    HardwareProfile,
    PartitionCatalog,
    RuntimeCatalog,
)
from lumi_fit.domain.models import ModelCatalog, ModelCoverageReport, ModelCoverageTargets
from lumi_fit.domain.plans import ServingPlan
from lumi_fit.domain.requests import ModelSelector, ResourceEnvelope, ServingRequirements
from lumi_fit.domain.results import (
    CapacityEnvelope,
    LumiVllmFitAnalysis,
    RecommendationBundle,
    SloGoodputEvidence,
)

SCHEMAS = {
    "billing-catalog.schema.json": BillingCatalog,
    "capacity-envelope.schema.json": CapacityEnvelope,
    "hardware-profile.schema.json": HardwareProfile,
    "lumi-vllm-fit-analysis.schema.json": LumiVllmFitAnalysis,
    "model-selector.schema.json": ModelSelector,
    "model-catalog.schema.json": ModelCatalog,
    "model-coverage-report.schema.json": ModelCoverageReport,
    "model-coverage-targets.schema.json": ModelCoverageTargets,
    "partition-catalog.schema.json": PartitionCatalog,
    "resource-envelope.schema.json": ResourceEnvelope,
    "runtime-catalog.schema.json": RuntimeCatalog,
    "serving-requirements.schema.json": ServingRequirements,
    "serving-plan.schema.json": ServingPlan,
    "slo-goodput-evidence.schema.json": SloGoodputEvidence,
    "recommendation-bundle.schema.json": RecommendationBundle,
}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    output_directory = repository_root / "data" / "schemas"
    output_directory.mkdir(parents=True, exist_ok=True)

    for filename, model in sorted(SCHEMAS.items()):
        schema = model.model_json_schema(mode="validation")
        rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        (output_directory / filename).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
