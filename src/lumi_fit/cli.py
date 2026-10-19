"""Command-line interface for LUMI model-serving planning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from lumi_fit import __version__
from lumi_fit.catalog.coverage import build_model_coverage_report
from lumi_fit.catalog.loader import (
    load_contract,
    load_model_selector,
    load_resources,
    load_serving,
)
from lumi_fit.catalog.validation import (
    assert_valid_catalogs,
    load_default_catalogs,
)
from lumi_fit.domain.models import ModelCapability
from lumi_fit.domain.requests import ModelSelector, ResourceEnvelope, ServingRequirements
from lumi_fit.domain.results import RecommendationBundle
from lumi_fit.domain.units import CountRange
from lumi_fit.planning.model_first import build_model_first_bundle
from lumi_fit.planning.resource_first import build_resource_first_bundle
from lumi_fit.presentation import (
    PresentationOptions,
    filter_and_sort_candidates,
    format_bundle_table,
    format_candidates_csv,
    format_plan_explanation,
)


def _add_repository(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", type=Path, default=Path.cwd())


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        help="Save JSON evidence, or displayed rows when --format csv is used",
    )
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--view", choices=("compact", "detail"), default="compact")
    parser.add_argument("--all-plans", action="store_true", help="Include alternative layouts")
    parser.add_argument(
        "--sort",
        choices=(
            "model",
            "parameters",
            "fit",
            "hbm",
            "throughput",
            "gpu-hours",
            "safe-concurrency",
        ),
        default="fit",
    )
    parser.add_argument("--sort-direction", choices=("asc", "desc"))
    parser.add_argument(
        "--status",
        action="append",
        choices=("feasible", "conditional", "infeasible"),
        default=[],
    )
    parser.add_argument(
        "--fit-band",
        action="append",
        choices=("excellent", "good", "marginal"),
        default=[],
    )
    parser.add_argument("--partition", action="append", default=[])
    parser.add_argument("--precision", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumi-fit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"lumi-fit {__version__}")
    commands = parser.add_subparsers(dest="command")

    models = commands.add_parser("models", help="Find models for a fixed allocation")
    _add_repository(models)
    models.add_argument("--resources", type=Path, required=True)
    models.add_argument("--serving", type=Path, required=True)
    models.add_argument("--model-selector", type=Path)
    models.add_argument("--include", action="append", default=[])
    models.add_argument("--exclude", action="append", default=[])
    models.add_argument("--family", action="append", default=[])
    models.add_argument("--license", action="append", default=[])
    models.add_argument("--language", action="append", default=[])
    models.add_argument(
        "--capability",
        action="append",
        choices=tuple(capability.value for capability in ModelCapability),
        default=[],
    )
    models.add_argument("--min-parameters", type=int)
    models.add_argument("--max-parameters", type=int)
    models.add_argument("--allow-gated", action="store_true")
    _add_output(models)

    plan = commands.add_parser("plan", help="Find resources for one model")
    _add_repository(plan)
    plan.add_argument("model")
    plan.add_argument("--resources", type=Path)
    plan.add_argument("--serving", type=Path, required=True)
    _add_output(plan)

    compare = commands.add_parser("compare", help="Compare selected models")
    _add_repository(compare)
    compare.add_argument("models_file", type=Path)
    compare.add_argument("--resources", type=Path, required=True)
    compare.add_argument("--serving", type=Path, required=True)
    _add_output(compare)

    explain = commands.add_parser("explain", help="Explain a saved recommendation")
    explain.add_argument("recommendation", type=Path)
    explain.add_argument("--plan", required=True)

    validate = commands.add_parser("validate", help="Validate an input contract")
    validate.add_argument("path", type=Path)

    catalog = commands.add_parser("catalog", help="Inspect versioned catalogs")
    _add_repository(catalog)
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_commands.add_parser("validate", help="Validate all catalog references")
    model_catalog = catalog_commands.add_parser("models", help="List cataloged models")
    model_catalog.add_argument("query", nargs="*", default=())
    coverage = catalog_commands.add_parser(
        "coverage", help="Report model-catalog coverage and provenance targets"
    )
    coverage.add_argument("--json", action="store_true")

    system = commands.add_parser("system", help="Show the LUMI hardware profile")
    _add_repository(system)
    system.add_argument("system_command", nargs="?", choices=("show",), default="show")

    doctor = commands.add_parser("doctor", help="Check catalogs and optional UI support")
    _add_repository(doctor)
    return parser


def _default_model_first_resources(serving: ServingRequirements) -> ResourceEnvelope:
    return ResourceEnvelope(
        partition="auto",
        nodes=CountRange(minimum=1, maximum=2),
        gcds=CountRange(minimum=1, maximum=16),
        walltime=serving.service_duration or serving.completion_deadline,
        allow_partial_node=True,
        allow_multi_node=True,
        enabled_partitions=("small-g", "standard-g"),
    )


def _presentation_options(args: argparse.Namespace) -> PresentationOptions:
    return PresentationOptions(
        include_alternatives=args.all_plans,
        statuses=tuple(args.status),
        fit_bands=tuple(args.fit_band),
        partitions=tuple(args.partition),
        precisions=tuple(args.precision),
        sort_by=args.sort,
        sort_direction=args.sort_direction,
    )


def _emit_bundle(bundle: RecommendationBundle, args: argparse.Namespace) -> None:
    rendered = bundle.model_dump_json(indent=2)
    output_format = args.format
    candidates = filter_and_sort_candidates(bundle, _presentation_options(args))
    if output_format != "json" and not candidates:
        raise ValueError("result filters did not match any displayed candidates")
    csv_output = format_candidates_csv(candidates)
    if args.output is not None:
        saved = csv_output if output_format == "csv" else rendered + "\n"
        args.output.write_text(saved, encoding="utf-8")
        print(f"Saved {args.output}", file=sys.stderr)
    if output_format == "json":
        print(rendered)
    elif output_format == "csv":
        print(csv_output, end="")
    else:
        print(
            format_bundle_table(
                bundle,
                candidates,
                detailed=args.view == "detail",
            )
        )
        if args.view == "detail":
            for candidate in candidates:
                print("\n" + format_plan_explanation(bundle, candidate.plan_id))


def _selector_from_args(
    args: argparse.Namespace,
    serving: ServingRequirements,
) -> ModelSelector:
    selector = (
        load_model_selector(args.model_selector)
        if args.model_selector
        else ModelSelector(use_case=serving.use_case)
    )
    updates: dict[str, object] = {}
    for field, values in (
        ("include_patterns", args.include),
        ("exclude_patterns", args.exclude),
        ("families", args.family),
        ("licenses", args.license),
        ("languages", args.language),
        ("capabilities", args.capability),
    ):
        if values:
            updates[field] = (*getattr(selector, field), *values)
    if args.min_parameters is not None:
        updates["minimum_parameters"] = args.min_parameters
    if args.max_parameters is not None:
        updates["maximum_parameters"] = args.max_parameters
    if args.allow_gated:
        updates["allow_gated"] = True
    if not updates:
        return selector
    payload = selector.model_dump(mode="json")
    payload.update(updates)
    return ModelSelector.model_validate(payload)


def _read_bundle(path: Path) -> RecommendationBundle:
    return RecommendationBundle.model_validate_json(path.read_text(encoding="utf-8"))


def _infer_contract(path: Path) -> type[ResourceEnvelope | ServingRequirements | ModelSelector]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contract file must contain a YAML/JSON object")
    if "mode" in payload:
        return ServingRequirements
    if "gcds" in payload:
        return ResourceEnvelope
    if any(
        key in payload
        for key in (
            "exact_model",
            "models",
            "include_patterns",
            "use_case",
            "capabilities",
            "families",
            "licenses",
            "languages",
            "minimum_parameters",
            "maximum_parameters",
        )
    ):
        return ModelSelector
    raise ValueError("cannot infer contract type from the file fields")


def _models_command(args: argparse.Namespace) -> None:
    catalogs = load_default_catalogs(args.repository)
    assert_valid_catalogs(catalogs)
    serving = load_serving(args.serving)
    selector = _selector_from_args(args, serving)
    bundle = build_resource_first_bundle(
        model_selector=selector,
        serving=serving,
        resources=load_resources(args.resources),
        catalogs=catalogs,
    )
    _emit_bundle(bundle, args)


def _plan_command(args: argparse.Namespace) -> None:
    catalogs = load_default_catalogs(args.repository)
    assert_valid_catalogs(catalogs)
    serving = load_serving(args.serving)
    resources = (
        load_resources(args.resources)
        if args.resources
        else _default_model_first_resources(serving)
    )
    bundle = build_model_first_bundle(
        model_selector=ModelSelector(exact_model=args.model),
        serving=serving,
        resources=resources,
        catalogs=catalogs,
    )
    _emit_bundle(bundle, args)


def _compare_command(args: argparse.Namespace) -> None:
    model_ids = tuple(
        line.strip()
        for line in args.models_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not model_ids:
        raise ValueError("models file does not contain any model IDs")
    catalogs = load_default_catalogs(args.repository)
    assert_valid_catalogs(catalogs)
    serving = load_serving(args.serving)
    bundle = build_resource_first_bundle(
        model_selector=ModelSelector(models=model_ids),
        serving=serving,
        resources=load_resources(args.resources),
        catalogs=catalogs,
    )
    _emit_bundle(bundle, args)


def _catalog_command(args: argparse.Namespace) -> None:
    catalogs = load_default_catalogs(args.repository)
    if args.catalog_command == "validate":
        assert_valid_catalogs(catalogs)
        print("Catalogs are valid.")
        return
    if args.catalog_command == "coverage":
        report = build_model_coverage_report(catalogs.models, catalogs.model_coverage)
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            status = "complete" if report.complete else "incomplete"
            print(f"Model catalog coverage: {status}")
            print(
                f"{report.model_count} models, {report.artifact_count} artifacts, "
                f"{len(report.family_counts)} families, "
                f"{report.gated_model_count} gated"
            )
            provenance = "complete" if report.provenance_complete else "incomplete"
            print(f"provenance: {provenance} ({report.catalog_source_count} catalog sources)")
            for band, count in report.parameter_band_counts.items():
                print(f"{band}: {count}")
            for target in report.unmet_targets:
                print(f"unmet: {target}")
            for issue in report.provenance_issues:
                print(f"provenance issue: {issue}")
        return
    query_parts = args.query[1:] if args.query[:1] == ["search"] else args.query
    query = " ".join(query_parts).casefold()
    for model in catalogs.models.models:
        if query in model.model_id.casefold():
            capabilities = ",".join(capability.value for capability in model.capabilities)
            print(
                f"{model.model_id}\t{model.family}\t"
                f"{model.architecture.total_parameters}\t"
                f"{model.architecture.context_length}\t{model.license or 'unknown'}\t"
                f"{capabilities}"
            )


def _system_command(args: argparse.Namespace) -> None:
    hardware = load_default_catalogs(args.repository).hardware
    print(f"{hardware.display_name} ({hardware.system_id})")
    print(f"{hardware.gcd.gcds_per_node} GCDs/node, {hardware.gcd.hbm_per_gcd} HBM/GCD")
    print(
        f"{hardware.cpu.schedulable_cores} schedulable CPU cores/node, "
        f"{hardware.cpu.memory_per_node} RAM/node"
    )


def _doctor_command(args: argparse.Namespace) -> None:
    assert_valid_catalogs(load_default_catalogs(args.repository))
    print("[ok] catalogs and cross-references")
    print("[ok] Pydantic and YAML contract runtime")
    print("[ok] Prompt Toolkit full-screen terminal interface")


def run(args: argparse.Namespace) -> None:
    if args.command == "models":
        _models_command(args)
    elif args.command == "plan":
        _plan_command(args)
    elif args.command == "compare":
        _compare_command(args)
    elif args.command == "explain":
        print(format_plan_explanation(_read_bundle(args.recommendation), args.plan))
    elif args.command == "validate":
        contract = _infer_contract(args.path)
        load_contract(args.path, contract)
        print(f"Valid {contract.__name__}: {args.path}")
    elif args.command == "catalog":
        _catalog_command(args)
    elif args.command == "system":
        _system_command(args)
    elif args.command == "doctor":
        _doctor_command(args)
    else:
        from lumi_fit.tui import run_tui

        run_tui()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
