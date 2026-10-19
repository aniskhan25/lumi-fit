from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from lumi_fit.cli import main
from lumi_fit.domain.results import RecommendationBundle
from lumi_fit.presentation import fit_score_label, quality_label
from lumi_fit.tui import DashboardState

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden"


def test_models_command_saves_self_contained_bundle(tmp_path, capsys) -> None:
    output = tmp_path / "recommendation.json"
    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-one-node-four-gcds.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--model-selector",
            str(ROOT / "examples/models-coding.yaml"),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    bundle = RecommendationBundle.model_validate_json(output.read_text(encoding="utf-8"))

    assert "Qwen/Qwen3-8B" in captured.out
    assert "VALID FIT" in captured.out
    assert "EST COVER" in captured.out
    assert "Saved" in captured.err
    assert bundle.model_groups
    assert all(candidate.lumi_vllm_fit is not None for candidate in bundle.candidates)
    selected_ids = {group.best_plan_id for group in bundle.model_groups if group.best_plan_id}
    assert all(
        candidate.capacity_envelope is not None
        for candidate in bundle.candidates
        if candidate.plan_id in selected_ids
    )


def test_explain_reads_saved_analysis_without_replanning(tmp_path, capsys) -> None:
    output = tmp_path / "recommendation.json"
    main(
        [
            "plan",
            "Qwen/Qwen3-32B",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-model-first.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()
    bundle = RecommendationBundle.model_validate_json(output.read_text(encoding="utf-8"))
    plan_id = next(
        candidate.plan_id for candidate in bundle.candidates if "recommended" in candidate.tiers
    )

    main(["explain", str(output), "--plan", plan_id])
    explanation = capsys.readouterr().out

    assert f"Plan {plan_id}" in explanation
    assert "LUMI serving assessment" in explanation
    assert "SERVEABILITY=" in explanation
    assert "QUALITY=" in explanation
    assert "ESTIMATED_CAPACITY_COVERAGE=" in explanation
    assert "VALIDATED_LUMI_VLLM_FIT=" in explanation
    assert "Memory-safe capacity envelope" in explanation
    assert "Analytical performance capacity: unknown" in explanation
    assert bundle.analysis_fingerprint in explanation


def test_dashboard_opens_on_ranked_models_and_supports_live_exploration() -> None:
    dashboard = DashboardState.create(ROOT)

    assert len(dashboard.candidates) == 11
    assert dashboard.mode == "models"
    assert "LUMI-G" in dashboard.allocation_summary()
    selected = dashboard.selected
    assert selected is not None
    assert selected.lumi_vllm_fit is not None
    context_gate = next(gate for gate in selected.lumi_vllm_fit.gates if gate.gate_id == "context")
    assert context_gate.passed is True
    assert selected.lumi_vllm_fit.eligibility.value == "eligible"
    assert quality_label(selected.lumi_vllm_fit) == "NOT REQUIRED"
    assert selected.lumi_vllm_fit.estimated_capacity_coverage_score is not None
    assert fit_score_label(selected.lumi_vllm_fit) == "UNSCORED"

    dashboard.query = "qwen3-32b"
    assert [candidate.plan.model_id for candidate in dashboard.candidates] == ["Qwen/Qwen3-32B"]
    dashboard.query = ""
    dashboard.cycle_filter()
    assert dashboard.fit_filter == "feasible"
    assert all(
        candidate.analysis.feasibility.value == "feasible" for candidate in dashboard.candidates
    )


def test_dashboard_plan_mode_uses_selected_model_and_returns_to_models() -> None:
    dashboard = DashboardState.create(ROOT)
    selected = dashboard.selected
    assert selected is not None
    model_id = selected.plan.model_id

    dashboard.enter_plan_mode()

    assert dashboard.mode == "plan"
    assert dashboard.candidates
    assert {candidate.plan.model_id for candidate in dashboard.candidates} == {model_id}

    dashboard.leave_plan_mode()
    assert dashboard.mode == "models"
    assert len(dashboard.candidates) == 11


def test_dashboard_top_controls_recalculate_local_recommendations() -> None:
    dashboard = DashboardState.create(ROOT)

    dashboard.cycle_use_case()
    assert dashboard.serving.use_case.value == "chat"

    dashboard.cycle_allocation()
    assert dashboard.resources.partition == "standard-g"
    assert dashboard.resources.gcds.maximum == 8

    dashboard.cycle_context()
    assert dashboard.serving.max_model_length == 16384

    dashboard.cycle_concurrency()
    assert dashboard.serving.concurrent_sequences == 16

    dashboard.cycle_target_rate()
    assert dashboard.serving.request_arrival_rate_per_second == Decimal("0.5")

    dashboard.set_fit_filter("excellent")
    dashboard.set_sort("parameters")
    assert dashboard.fit_filter == "excellent"
    assert dashboard.sort_by == "parameters"
    assert dashboard.bundle.model_groups


def test_dashboard_task_control_skips_unsupported_catalog_tasks() -> None:
    dashboard = DashboardState.create(ROOT)

    assert dashboard.available_use_cases == ("general", "chat", "coding", "reasoning")
    for expected in ("chat", "coding", "reasoning", "general"):
        dashboard.cycle_use_case()
        assert dashboard.serving.use_case.value == expected
        assert dashboard.bundle.model_groups

    with pytest.raises(ValueError, match="unavailable in the model catalog"):
        dashboard.set_use_case("embedding")
    assert dashboard.serving.use_case.value == "general"


def test_catalog_coverage_command_reports_complete_matrix(capsys) -> None:
    main(["catalog", "--repository", str(ROOT), "coverage"])

    rendered = capsys.readouterr().out
    assert "Model catalog coverage: complete" in rendered
    assert "11 models, 11 artifacts, 9 families" in rendered


def test_catalog_coverage_command_supports_json(capsys) -> None:
    main(["catalog", "--repository", str(ROOT), "coverage", "--json"])

    rendered = capsys.readouterr().out
    assert '"complete": true' in rendered
    assert '"large": 2' in rendered


def test_direct_model_and_result_filters_need_no_selector_file(capsys) -> None:
    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-one-node-four-gcds.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--family",
            "phi3",
            "--language",
            "fi",
            "--capability",
            "coding",
            "--max-parameters",
            "4000000000",
            "--status",
            "feasible",
            "--format",
            "csv",
        ]
    )

    rendered = capsys.readouterr().out
    assert "microsoft/Phi-4-mini-instruct" in rendered
    assert "Qwen/" not in rendered


def test_detail_view_expands_selected_plan_explanations(capsys) -> None:
    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-one-node-four-gcds.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--family",
            "phi3",
            "--view",
            "detail",
        ]
    )

    rendered = capsys.readouterr().out
    assert "PARTITION" in rendered
    assert "CONFIDENCE" in rendered
    assert "Plan " in rendered
    assert "Memory-safe capacity envelope" in rendered


def test_csv_output_file_matches_stdout(tmp_path, capsys) -> None:
    output = tmp_path / "recommendations.csv"
    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-one-node-four-gcds.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--include",
            "Qwen/Qwen3-8B",
            "--format",
            "csv",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert output.read_text(encoding="utf-8") == captured.out
    assert "Saved" in captured.err


def test_json_keeps_complete_evidence_when_display_filters_are_present(capsys) -> None:
    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(ROOT / "examples/resources-one-node-four-gcds.yaml"),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--model-selector",
            str(ROOT / "examples/models-coding.yaml"),
            "--status",
            "feasible",
            "--format",
            "json",
        ]
    )

    bundle = RecommendationBundle.model_validate_json(capsys.readouterr().out)
    assert any(
        candidate.analysis.feasibility.value == "infeasible" for candidate in bundle.candidates
    )


@pytest.mark.parametrize("gcd_count", (1, 2, 4, 8))
def test_golden_csv_scenarios_for_common_allocations(
    gcd_count: int,
    tmp_path,
    capsys,
) -> None:
    resource_payload = yaml.safe_load(
        (ROOT / "examples/resources-one-node-four-gcds.yaml").read_text(encoding="utf-8")
    )
    resource_payload["gcds"] = gcd_count
    resource_payload["maximum_gpu_hours"] = 32
    resources = tmp_path / f"resources-{gcd_count}-gcd.yaml"
    resources.write_text(yaml.safe_dump(resource_payload), encoding="utf-8")

    main(
        [
            "models",
            "--repository",
            str(ROOT),
            "--resources",
            str(resources),
            "--serving",
            str(ROOT / "examples/serving-online.yaml"),
            "--include",
            "Qwen/Qwen3-8B",
            "--format",
            "csv",
        ]
    )

    rendered = capsys.readouterr().out
    expected = (GOLDEN / f"cli-{gcd_count}-gcd.csv").read_text(encoding="utf-8")
    assert rendered == expected
