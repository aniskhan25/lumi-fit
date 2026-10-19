from pathlib import Path

from lumi_fit.catalog.coverage import build_model_coverage_report
from lumi_fit.catalog.validation import load_default_catalogs


ROOT = Path(__file__).resolve().parents[2]


def test_phase4a_catalog_meets_versioned_coverage_targets() -> None:
    catalogs = load_default_catalogs(ROOT)

    report = build_model_coverage_report(catalogs.models, catalogs.model_coverage)

    assert report.complete
    assert report.unmet_targets == ()
    assert report.model_count == report.artifact_count == 11
    assert len(report.family_counts) == 9
    assert report.parameter_band_counts == {
        "compact": 2,
        "small": 3,
        "medium": 4,
        "large": 2,
    }


def test_every_artifact_has_revision_pinned_provenance() -> None:
    catalog = load_default_catalogs(ROOT).models

    for artifact in catalog.artifacts:
        assert len(artifact.revision) == 40
        assert artifact.revision in artifact.source_url
        assert artifact.model_id in artifact.source_url
        assert artifact.total_bytes is not None

