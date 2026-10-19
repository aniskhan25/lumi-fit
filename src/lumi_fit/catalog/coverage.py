"""Catalog coverage reporting against versioned coverage targets."""

from __future__ import annotations

import re
from collections import Counter

from lumi_fit.domain.models import (
    ModelCatalog,
    ModelCoverageReport,
    ModelCoverageTargets,
)


def build_model_coverage_report(
    catalog: ModelCatalog,
    targets: ModelCoverageTargets,
) -> ModelCoverageReport:
    family_counts = Counter(model.family for model in catalog.models)
    capability_counts = Counter(
        capability.value for model in catalog.models for capability in model.capabilities
    )
    license_counts = Counter((model.license or "unknown") for model in catalog.models)
    quantization_counts = Counter(artifact.quantization for artifact in catalog.artifacts)
    band_counts = {
        band.band_id: sum(
            1
            for model in catalog.models
            if model.architecture.total_parameters >= band.minimum_parameters
            and (
                band.maximum_parameters is None
                or model.architecture.total_parameters <= band.maximum_parameters
            )
        )
        for band in targets.parameter_bands
    }

    unmet: list[str] = []
    if len(catalog.models) < targets.minimum_models:
        unmet.append(f"models: {len(catalog.models)} present, {targets.minimum_models} required")
    if len(family_counts) < targets.minimum_families:
        unmet.append(f"families: {len(family_counts)} present, {targets.minimum_families} required")
    for capability in targets.required_capabilities:
        if capability_counts[capability.value] == 0:
            unmet.append(f"capability {capability.value!r} is not represented")
    for band in targets.parameter_bands:
        count = band_counts[band.band_id]
        if count < band.minimum_models:
            unmet.append(f"band {band.band_id!r}: {count} present, {band.minimum_models} required")

    provenance_issues: list[str] = []
    sha_pattern = re.compile(r"^[0-9a-f]{40}$")
    for model in catalog.models:
        if not sha_pattern.fullmatch(model.revision):
            provenance_issues.append(
                f"model {model.model_id!r} does not use a 40-character revision SHA"
            )
        if not model.source_url.startswith("https://"):
            provenance_issues.append(f"model {model.model_id!r} does not use an HTTPS source URL")
    for artifact in catalog.artifacts:
        if artifact.revision not in artifact.source_url:
            provenance_issues.append(
                f"artifact {artifact.artifact_id!r} source is not revision pinned"
            )
        if artifact.total_bytes is None:
            provenance_issues.append(f"artifact {artifact.artifact_id!r} has no exact total size")

    return ModelCoverageReport(
        catalog_profile_version=catalog.metadata.profile_version,
        target_profile_version=targets.metadata.profile_version,
        catalog_source_count=len(catalog.metadata.sources),
        model_count=len(catalog.models),
        artifact_count=len(catalog.artifacts),
        gated_model_count=sum(model.gated for model in catalog.models),
        family_counts=dict(sorted(family_counts.items())),
        capability_counts=dict(sorted(capability_counts.items())),
        license_counts=dict(sorted(license_counts.items())),
        quantization_counts=dict(sorted(quantization_counts.items())),
        parameter_band_counts=band_counts,
        provenance_issues=tuple(provenance_issues),
        provenance_complete=not provenance_issues,
        unmet_targets=tuple(unmet),
        complete=not unmet and not provenance_issues,
    )
