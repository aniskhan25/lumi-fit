"""Presentation-independent summaries for CLI and interactive interfaces."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from lumi_fit.domain.results import (
    CandidateResult,
    LumiVllmFitAnalysis,
    QualityRequirementStatus,
    RecommendationBundle,
)
from lumi_fit.scoring.engine import rank_candidates

SortField = Literal[
    "model",
    "parameters",
    "fit",
    "hbm",
    "throughput",
    "gpu-hours",
    "safe-concurrency",
]


@dataclass(frozen=True)
class PresentationOptions:
    include_alternatives: bool = False
    statuses: tuple[str, ...] = ()
    fit_bands: tuple[str, ...] = ()
    partitions: tuple[str, ...] = ()
    precisions: tuple[str, ...] = ()
    sort_by: SortField = "fit"
    sort_direction: Literal["asc", "desc"] | None = None


DEFAULT_PRESENTATION_OPTIONS = PresentationOptions()


def quality_label(fit: LumiVllmFitAnalysis | None) -> str:
    """Render optional quality status without fabricating a quality value."""

    if fit is None or fit.quality_status is QualityRequirementStatus.NOT_REQUIRED:
        return "NOT REQUIRED"
    if fit.quality_status is QualityRequirementStatus.UNKNOWN:
        return "UNKNOWN"
    score = f" {fit.model_quality:.1f}" if fit.model_quality is not None else ""
    return f"{fit.quality_status.value.upper()}{score}"


def estimated_coverage_label(fit: LumiVllmFitAnalysis | None) -> str:
    """Render analytical target coverage separately from validated fit."""

    if fit is None or fit.target_request_rate_per_second is None:
        return "NO TARGET"
    score = fit.estimated_capacity_coverage_score
    return f"{score:.1f}" if score is not None else "N/A"


def fit_score_label(fit: LumiVllmFitAnalysis | None) -> str:
    """A missing validated fit is unscored, never a numeric zero or a dash."""

    return f"{fit.fit_score:.1f}" if fit and fit.fit_score is not None else "UNSCORED"


def selected_candidates(bundle: RecommendationBundle) -> tuple[CandidateResult, ...]:
    """Return tiered model-first plans or one selected plan per model group."""

    if bundle.model_groups:
        selected_ids = {
            group.best_plan_id or group.diagnostic_plan_id
            for group in bundle.model_groups
            if group.best_plan_id or group.diagnostic_plan_id
        }
        return tuple(
            candidate for candidate in bundle.candidates if candidate.plan_id in selected_ids
        )
    tiered = tuple(candidate for candidate in bundle.candidates if candidate.tiers)
    return tiered or bundle.candidates[:10]


def filter_and_sort_candidates(
    bundle: RecommendationBundle,
    options: PresentationOptions = DEFAULT_PRESENTATION_OPTIONS,
) -> tuple[CandidateResult, ...]:
    candidates = bundle.candidates if options.include_alternatives else selected_candidates(bundle)
    statuses = {value.casefold() for value in options.statuses}
    fit_bands = {value.casefold() for value in options.fit_bands}
    partitions = {value.casefold() for value in options.partitions}
    precisions = {value.casefold() for value in options.precisions}
    filtered = tuple(
        candidate
        for candidate in candidates
        if (not statuses or candidate.analysis.feasibility.value.casefold() in statuses)
        and (
            not fit_bands
            or (
                candidate.analysis.fit_band is not None
                and candidate.analysis.fit_band.value.casefold() in fit_bands
            )
        )
        and (not partitions or candidate.plan.partition.casefold() in partitions)
        and (not precisions or candidate.plan.quantization.casefold() in precisions)
    )

    def metric(candidate: CandidateResult) -> str | int | Decimal | None:
        if options.sort_by == "model":
            return candidate.plan.model_id.casefold()
        if options.sort_by == "parameters":
            return candidate.model_parameters
        if options.sort_by == "fit":
            fit = candidate.lumi_vllm_fit
            if fit and fit.fit_score is not None:
                return Decimal(1000) + fit.fit_score
            return fit.estimated_capacity_coverage_score if fit else None
        if options.sort_by == "hbm":
            memory = candidate.analysis.memory
            return memory.conservative_utilization_fraction if memory else None
        if options.sort_by == "throughput":
            performance = candidate.analysis.performance
            estimate = performance.aggregate_output_tokens_per_second if performance else None
            return estimate.expected if estimate else None
        if options.sort_by == "gpu-hours":
            billing = candidate.analysis.billing
            return billing.gpu_hours_per_wallclock_hour if billing else None
        envelope = candidate.capacity_envelope
        maximum = envelope.memory_safe_maximum_concurrency if envelope else None
        return maximum.total_concurrent_sequences if maximum else None

    default_direction = {
        "model": "asc",
        "parameters": "desc",
        "fit": "desc",
        "hbm": "asc",
        "throughput": "desc",
        "gpu-hours": "asc",
        "safe-concurrency": "desc",
    }[options.sort_by]
    descending = (options.sort_direction or default_direction) == "desc"
    available = [candidate for candidate in filtered if metric(candidate) is not None]
    missing = [candidate for candidate in filtered if metric(candidate) is None]
    available.sort(
        key=lambda candidate: (metric(candidate), candidate.plan_id),
        reverse=descending,
    )
    if options.sort_by == "fit":
        missing = list(rank_candidates(tuple(missing), bundle.serving))
    else:
        missing.sort(key=lambda candidate: candidate.plan_id)
    return tuple((*available, *missing))


def _parameter_label(parameters: int | None) -> str:
    if parameters is None:
        return "-"
    return f"{Decimal(parameters) / Decimal(1_000_000_000):.1f}B"


def format_bundle_table(
    bundle: RecommendationBundle,
    candidates: tuple[CandidateResult, ...] | None = None,
    *,
    detailed: bool = False,
) -> str:
    headers = [
        "MODEL",
        "PARAMS",
        "LAYOUT",
        "GCDS",
        "STATUS",
        "MEM FIT",
        "HBM",
        "SAFE CONC",
        "SAFE CTX",
        "TOK/S",
        "GPU-H/H",
        "SERVEABILITY",
        "EST COVER",
        "QUALITY",
        "VALID FIT",
    ]
    if detailed:
        headers.extend(("PARTITION", "PREC", "CTX/CONC", "CONFIDENCE"))
    headers.extend(("TIER", "PLAN"))
    rows: list[tuple[str, ...]] = []
    for candidate in candidates if candidates is not None else selected_candidates(bundle):
        plan = candidate.plan
        analysis = candidate.analysis
        memory = analysis.memory
        performance = analysis.performance
        billing = analysis.billing
        capacity = candidate.capacity_envelope
        fit = candidate.lumi_vllm_fit
        row = [
            plan.model_id,
            _parameter_label(candidate.model_parameters),
            f"TP{plan.tensor_parallel}x{plan.replicas}",
            f"{plan.used_gcds}/{plan.allocated_gcds}",
            analysis.feasibility.value,
            analysis.fit_band.value if analysis.fit_band else "-",
            (f"{memory.conservative_utilization_fraction * 100:.1f}%" if memory else "-"),
            (
                str(capacity.memory_safe_maximum_concurrency.total_concurrent_sequences)
                if capacity and capacity.memory_safe_maximum_concurrency
                else "-"
            ),
            (
                str(capacity.memory_safe_maximum_context.context_tokens)
                if capacity and capacity.memory_safe_maximum_context
                else "-"
            ),
            (
                f"{performance.aggregate_output_tokens_per_second.expected:.1f}"
                if performance and performance.aggregate_output_tokens_per_second
                else "-"
            ),
            f"{billing.gpu_hours_per_wallclock_hour:.1f}" if billing else "-",
            fit.eligibility.value if fit else "unknown",
            estimated_coverage_label(fit),
            quality_label(fit),
            fit_score_label(fit),
        ]
        if detailed:
            row.extend(
                (
                    plan.partition,
                    plan.quantization,
                    (
                        f"{plan.runtime_options.max_model_length}/"
                        f"{plan.runtime_options.max_concurrent_sequences}"
                    ),
                    analysis.overall_confidence.value,
                )
            )
        row.extend((",".join(candidate.tiers) or "-", candidate.plan_id))
        rows.append(tuple(row))
    widths = [len(header) for header in headers]
    for display_row in rows:
        for index, value in enumerate(display_row):
            widths[index] = max(widths[index], len(value))

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    header_row = tuple(headers)
    separator = render(tuple("-" * width for width in widths))
    return "\n".join((render(header_row), separator, *(render(row) for row in rows)))


def find_candidate(bundle: RecommendationBundle, plan_id: str) -> CandidateResult:
    candidate = next(
        (candidate for candidate in bundle.candidates if candidate.plan_id == plan_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"plan {plan_id!r} is not present in the recommendation bundle")
    return candidate


def format_candidates_csv(candidates: tuple[CandidateResult, ...]) -> str:
    """Render stable, locale-independent recommendation rows."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "model_id",
            "parameters",
            "plan_id",
            "partition",
            "nodes",
            "allocated_gcds",
            "used_gcds",
            "tensor_parallel",
            "pipeline_parallel",
            "replicas",
            "precision",
            "status",
            "fit_band",
            "hbm_upper_fraction",
            "safe_max_concurrency",
            "safe_max_context",
            "output_tokens_per_second",
            "gpu_hours_per_wallclock_hour",
            "total_gpu_hours",
            "serveability",
            "quality_status",
            "model_quality",
            "estimated_capacity_coverage_score",
            "validated_lumi_vllm_fit_score",
            "raw_full_node_capacity_lower_requests_per_second",
            "tiers",
        )
    )
    for candidate in candidates:
        plan = candidate.plan
        analysis = candidate.analysis
        memory = analysis.memory
        performance = analysis.performance
        billing = analysis.billing
        envelope = candidate.capacity_envelope
        fit = candidate.lumi_vllm_fit
        concurrency = envelope.memory_safe_maximum_concurrency if envelope else None
        context = envelope.memory_safe_maximum_context if envelope else None
        output_rate = performance.aggregate_output_tokens_per_second if performance else None
        writer.writerow(
            (
                plan.model_id,
                candidate.model_parameters or "",
                candidate.plan_id,
                plan.partition,
                plan.nodes,
                plan.allocated_gcds,
                plan.used_gcds,
                plan.tensor_parallel,
                plan.pipeline_parallel,
                plan.replicas,
                plan.quantization,
                analysis.feasibility.value,
                analysis.fit_band.value if analysis.fit_band else "",
                memory.conservative_utilization_fraction if memory else "",
                concurrency.total_concurrent_sequences if concurrency else "",
                context.context_tokens if context else "",
                output_rate.expected if output_rate else "",
                billing.gpu_hours_per_wallclock_hour if billing else "",
                billing.total_gpu_hours if billing else "",
                fit.eligibility.value if fit else "unknown",
                fit.quality_status.value if fit else "unknown",
                fit.model_quality if fit and fit.model_quality is not None else "",
                (
                    fit.estimated_capacity_coverage_score
                    if fit and fit.estimated_capacity_coverage_score is not None
                    else ""
                ),
                fit.fit_score if fit and fit.fit_score is not None else "",
                (
                    fit.raw_full_node_capacity_lower_per_second
                    if fit and fit.raw_full_node_capacity_lower_per_second is not None
                    else ""
                ),
                ";".join(candidate.tiers),
            )
        )
    return output.getvalue()


def format_plan_explanation(bundle: RecommendationBundle, plan_id: str) -> str:
    candidate = find_candidate(bundle, plan_id)
    plan = candidate.plan
    analysis = candidate.analysis
    lines = [
        f"Plan {candidate.plan_id}",
        f"Model: {plan.model_id}@{plan.model_revision}",
        f"Artifact/runtime: {plan.artifact_id} / {plan.runtime_id}",
        (
            f"Allocation: {plan.partition}, {plan.nodes} node(s), "
            f"{plan.allocated_gcds} allocated GCDs, {plan.used_gcds} used"
        ),
        (
            f"Layout: TP={plan.tensor_parallel}, PP={plan.pipeline_parallel}, "
            f"replicas={plan.replicas}, routing={plan.request_routing.value}"
        ),
        (
            f"Capacity: context={plan.runtime_options.max_model_length}, "
            f"concurrency/replica={plan.runtime_options.max_concurrent_sequences}"
        ),
        (
            f"Result: {analysis.feasibility.value}, "
            f"memory-fit={analysis.fit_band.value if analysis.fit_band else '-'}, "
            f"confidence={analysis.overall_confidence.value}"
        ),
        "",
        "Constraints",
    ]
    lines.extend(
        f"  [{'PASS' if constraint.passed else 'FAIL'}] {constraint.constraint_id}: "
        f"{constraint.message}"
        + (f" Remediation: {constraint.remediation}" if constraint.remediation else "")
        for constraint in analysis.constraints
    )
    fit = candidate.lumi_vllm_fit
    if fit:
        lines.extend(
            (
                "",
                "LUMI serving assessment",
                f"  SERVEABILITY={fit.eligibility.value.upper()}",
                (
                    f"  ESTIMATED_CAPACITY_COVERAGE={estimated_coverage_label(fit)} "
                    f"({fit.estimated_capacity_confidence.value})"
                ),
                f"  QUALITY={quality_label(fit)}",
                f"  VALIDATED_LUMI_VLLM_FIT={fit_score_label(fit)}",
                (
                    "  Raw full-node request capacity lower bound: "
                    + (
                        str(fit.raw_full_node_capacity_lower_per_second)
                        if fit.raw_full_node_capacity_lower_per_second is not None
                        else "unknown"
                    )
                    + " requests/second (analytical, not SLO goodput)"
                ),
                f"  {fit.explanation}",
                "  Technical and optional-quality gates:",
            )
        )
        for gate in fit.gates:
            if (
                gate.gate_id == "quality"
                and fit.quality_status is QualityRequirementStatus.NOT_REQUIRED
            ):
                status = "NOT REQUIRED"
            else:
                status = (
                    "PASS" if gate.passed is True else "FAIL" if gate.passed is False else "UNKNOWN"
                )
            lines.append(f"    {gate.gate_id}: {status}; {gate.explanation}")
    if analysis.memory:
        lines.extend(("", "Per-rank HBM"))
        lines.extend(
            f"  {component.component}: expected={component.per_rank.expected} bytes, "
            f"upper={component.per_rank.upper} bytes, rule={component.sharding_rule}"
            for component in analysis.memory.components
        )
        lines.append(
            "  Peak: "
            f"expected={analysis.memory.expected_utilization_fraction * 100:.1f}%, "
            f"upper={analysis.memory.conservative_utilization_fraction * 100:.1f}%"
        )
    capacity = candidate.capacity_envelope
    if capacity:
        maximum_concurrency = capacity.memory_safe_maximum_concurrency
        maximum_context = capacity.memory_safe_maximum_context
        lines.extend(
            (
                "",
                "Memory-safe capacity envelope",
                (
                    f"  Requested: context={capacity.requested.context_tokens}, "
                    f"total concurrency={capacity.requested.total_concurrent_sequences}, "
                    f"memory-safe={'yes' if capacity.requested.memory_safe else 'no'}"
                ),
                (
                    "  Maximum concurrency at requested context: "
                    + (
                        str(maximum_concurrency.total_concurrent_sequences)
                        if maximum_concurrency
                        else "none"
                    )
                    + (" (search cap reached)" if capacity.concurrency_search_capped else "")
                ),
                (
                    "  Maximum context at requested concurrency: "
                    + (str(maximum_context.context_tokens) if maximum_context else "none")
                    + f"; catalog limit={capacity.catalog_context_limit}"
                ),
                f"  Evaluated memory points: {len(capacity.evaluated_points)}",
                (
                    "  Analytical performance capacity: "
                    + (
                        str(capacity.analytical_performance_maximum_concurrency)
                        if capacity.analytical_performance_maximum_concurrency
                        else "unknown"
                    )
                    + f". {capacity.analytical_performance_explanation}"
                ),
            )
        )
        for neighbor in (
            capacity.lower_cost_neighbor,
            capacity.higher_capacity_neighbor,
        ):
            if neighbor:
                lines.append(
                    f"  {neighbor.relationship} neighbor: {neighbor.plan_id}, "
                    f"{neighbor.allocated_gcds} GCDs, "
                    f"{neighbor.gpu_hours_per_wallclock_hour} GPU-h/h, "
                    f"safe concurrency={neighbor.memory_safe_maximum_concurrency}"
                )
    if analysis.performance:
        lines.extend(("", "Analytical performance (heuristic, not an SLA)"))
        for label, estimate in (
            ("TTFT", analysis.performance.ttft_seconds),
            ("ITL", analysis.performance.inter_token_latency_seconds),
            ("Aggregate output", analysis.performance.aggregate_output_tokens_per_second),
            ("Requests", analysis.performance.requests_per_second),
        ):
            if estimate:
                lines.append(
                    f"  {label}: {estimate.expected} {estimate.unit} expected "
                    f"({estimate.lower}–{estimate.upper}), {estimate.method_id}"
                )
    if analysis.billing:
        lines.extend(
            (
                "",
                "Billing",
                (
                    f"  {analysis.billing.gpu_hours_per_wallclock_hour} GPU-h/h; "
                    f"{analysis.billing.total_gpu_hours} GPU-h over "
                    f"{analysis.billing.duration_hours} h; "
                    f"rule={analysis.billing.billing_rule_id}"
                ),
            )
        )
    if analysis.assumptions:
        lines.extend(("", "Assumptions"))
        lines.extend(f"  - {assumption}" for assumption in analysis.assumptions)
    lines.extend(
        (
            "",
            f"Analysis fingerprint: {bundle.analysis_fingerprint}",
            "Performance values are analytical priors until validated on LUMI.",
        )
    )
    return "\n".join(lines)
