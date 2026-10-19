"""Full-screen terminal dashboard for local LUMI serving recommendations."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TypeVar, cast

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Button, Dialog, Frame, Label, TextArea

from lumi_fit.catalog.validation import CatalogSet, assert_valid_catalogs, load_default_catalogs
from lumi_fit.domain.models import ModelCapability
from lumi_fit.domain.requests import (
    LatencyTargets,
    ModelSelector,
    ResourceEnvelope,
    ServingMode,
    ServingRequirements,
    ThroughputTargets,
    TokenLengthPoint,
    UseCase,
)
from lumi_fit.domain.results import (
    CandidateResult,
    LumiVllmFitAnalysis,
    PlanAnalysis,
    RecommendationBundle,
)
from lumi_fit.domain.units import CountRange, Duration, MemoryAmount
from lumi_fit.planning.model_first import build_model_first_bundle
from lumi_fit.planning.resource_first import build_resource_first_bundle
from lumi_fit.presentation import (
    PresentationOptions,
    SortField,
    estimated_coverage_label,
    filter_and_sort_candidates,
    fit_score_label,
    quality_label,
)

SORT_FIELDS: tuple[SortField, ...] = (
    "fit",
    "parameters",
    "hbm",
    "safe-concurrency",
    "gpu-hours",
)
FIT_FILTERS = ("all", "feasible", "excellent", "good", "marginal", "infeasible")
USE_CASE_FILTERS = tuple(item.value for item in UseCase if item is not UseCase.CUSTOM)
_USE_CASE_CAPABILITY = {
    UseCase.GENERAL.value: ModelCapability.TEXT_GENERATION,
    UseCase.CHAT.value: ModelCapability.CHAT,
    UseCase.CODING.value: ModelCapability.CODING,
    UseCase.REASONING.value: ModelCapability.REASONING,
    UseCase.EMBEDDING.value: ModelCapability.EMBEDDING,
}
ALLOCATION_PRESETS = (
    ("small-g · 1", "small-g", 1, 1),
    ("small-g · 2", "small-g", 1, 2),
    ("small-g · 4", "small-g", 1, 4),
    ("standard-g · 8", "standard-g", 1, 8),
)
CONTEXT_OPTIONS = (4096, 8192, 16384, 32768, 65536, 131072)
CONCURRENCY_OPTIONS = (1, 2, 4, 8, 16, 32, 64)
TARGET_RATE_OPTIONS = (
    Decimal("0.05"),
    Decimal("0.1"),
    Decimal("0.25"),
    Decimal("0.5"),
    Decimal("1"),
    Decimal("2"),
)
_SIDE_BY_SIDE_MIN_COLUMNS = 130
_DETAIL_PANEL_ESTIMATED_COLUMNS = 52
_SINGLE_ROW_CONTROLS_MIN_COLUMNS = 140
_CycleValue = TypeVar("_CycleValue")


def _default_serving() -> ServingRequirements:
    return ServingRequirements(
        mode=ServingMode.ONLINE,
        use_case=UseCase.GENERAL,
        input_tokens=4096,
        max_output_tokens=1024,
        max_model_length=8192,
        concurrent_sequences=8,
        request_arrival_rate_per_second=Decimal("0.25"),
        service_duration=Duration.model_validate("8 hours"),
        latency_targets=LatencyTargets(),
        throughput_targets=ThroughputTargets(),
    )


def _default_resources() -> ResourceEnvelope:
    return ResourceEnvelope(
        partition="small-g",
        nodes=1,
        gcds=4,
        cpu_cores_per_node=32,
        cpu_memory_per_node=MemoryAmount.model_validate("256 GB"),
        walltime=Duration.model_validate("8 hours"),
        allow_partial_node=True,
        allow_multi_node=False,
    )


def _short(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width == 1:
        return "…"
    return value[: width - 1] + "…"


def _number(value: Decimal | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _detail_label(value: str) -> str:
    return f"{value:<12}"


def _eligibility_label(fit: LumiVllmFitAnalysis | None) -> str:
    return fit.eligibility.value if fit else "unknown"


def _hbm_label(analysis: PlanAnalysis, *, precision: int = 1) -> str:
    memory = analysis.memory
    return f"{memory.conservative_utilization_fraction * 100:.{precision}f}%" if memory else "-"


def _safe_concurrency_label(item: CandidateResult) -> str:
    envelope = item.capacity_envelope
    safe = envelope.memory_safe_maximum_concurrency if envelope else None
    return str(safe.total_concurrent_sequences) if safe else "-"


def _throughput_label(analysis: PlanAnalysis) -> str:
    performance = analysis.performance
    estimate = performance.aggregate_output_tokens_per_second if performance else None
    return _number(estimate.expected) if estimate else "-"


def _cost_label(analysis: PlanAnalysis) -> str:
    billing = analysis.billing
    return _number(billing.gpu_hours_per_wallclock_hour if billing else None)


@dataclass(frozen=True)
class _TableColumn:
    key: str
    heading: str
    width: int
    alignment: str = "left"

    def render(self, value: str) -> str:
        clipped = _short(value, self.width)
        return (
            f"{clipped:>{self.width}}" if self.alignment == "right" else f"{clipped:<{self.width}}"
        )


_COMPACT_COLUMNS = (
    _TableColumn("marker", "", 1),
    _TableColumn("model", "MODEL", 24),
    _TableColumn("eligibility", "SERVE", 9),
    _TableColumn("hbm", "HBM", 5, "right"),
    _TableColumn("safe", "SAFE", 5, "right"),
    _TableColumn("estimated", "EST FIT", 8, "right"),
    _TableColumn("quality", "QUALITY", 8, "right"),
    _TableColumn("fit", "VALID", 8, "right"),
)

_WIDE_COLUMNS = (
    _TableColumn("marker", "", 1),
    _TableColumn("model", "MODEL", 27),
    _TableColumn("params", "PARAMS", 7, "right"),
    _TableColumn("eligibility", "SERVE", 11),
    _TableColumn("hbm", "HBM", 5, "right"),
    _TableColumn("safe", "SAFE", 5, "right"),
    _TableColumn("rate", "TOK/S", 7, "right"),
    _TableColumn("gcds", "GCDS", 5, "right"),
    _TableColumn("cost", "GPU-H/H", 7, "right"),
    _TableColumn("estimated", "EST FIT", 9, "right"),
    _TableColumn("quality", "QUALITY", 12, "right"),
    _TableColumn("fit", "VALID FIT", 9, "right"),
)


def _table_line(values: dict[str, str], columns: tuple[_TableColumn, ...]) -> str:
    return " ".join(column.render(values.get(column.key, "")) for column in columns).rstrip()


@dataclass
class DashboardState:
    """UI-independent state and actions for the interactive dashboard."""

    repository: Path
    catalogs: CatalogSet
    resources: ResourceEnvelope
    serving: ServingRequirements
    bundle: RecommendationBundle
    mode: str = "models"
    query: str = ""
    fit_filter: str = "all"
    sort_by: SortField = "fit"
    selected_index: int = 0
    marked_plan_id: str | None = None
    details_visible: bool = True
    compare_visible: bool = False
    status: str = "Ready — local estimates only; nothing is launched on LUMI."
    _resource_bundle: RecommendationBundle | None = field(default=None, repr=False)

    @classmethod
    def create(cls, repository: Path | None = None) -> DashboardState:
        root = repository or Path.cwd()
        catalogs = load_default_catalogs(root)
        assert_valid_catalogs(catalogs)
        resources, serving = _default_resources(), _default_serving()
        bundle = cls._resource_first(catalogs, resources, serving)
        return cls(root, catalogs, resources, serving, bundle, _resource_bundle=bundle)

    @staticmethod
    def _resource_first(
        catalogs: CatalogSet,
        resources: ResourceEnvelope,
        serving: ServingRequirements,
    ) -> RecommendationBundle:
        return build_resource_first_bundle(
            model_selector=ModelSelector(include_patterns=("*",), use_case=serving.use_case),
            serving=serving,
            resources=resources,
            catalogs=catalogs,
        )

    @property
    def candidates(self) -> tuple[CandidateResult, ...]:
        statuses: tuple[str, ...] = ()
        bands: tuple[str, ...] = ()
        if self.fit_filter in {"feasible", "infeasible"}:
            statuses = (self.fit_filter,)
        elif self.fit_filter != "all":
            statuses, bands = ("feasible",), (self.fit_filter,)
        candidates = filter_and_sort_candidates(
            self.bundle,
            PresentationOptions(
                statuses=statuses,
                fit_bands=bands,
                sort_by=self.sort_by,
            ),
        )
        query = self.query.strip().casefold()
        if query:
            candidates = tuple(
                item
                for item in candidates
                if query
                in " ".join(
                    (
                        item.plan.model_id,
                        item.plan.partition,
                        item.plan.quantization,
                        item.analysis.feasibility.value,
                        item.analysis.fit_band.value if item.analysis.fit_band else "",
                    )
                ).casefold()
            )
        return candidates

    @property
    def selected(self) -> CandidateResult | None:
        if not self.candidates:
            return None
        self.selected_index = min(self.selected_index, len(self.candidates) - 1)
        return self.candidates[self.selected_index]

    def move(self, offset: int) -> None:
        if self.candidates:
            self.selected_index = max(
                0, min(self.selected_index + offset, len(self.candidates) - 1)
            )

    @staticmethod
    def _cycled(values: tuple[_CycleValue, ...], current: _CycleValue, offset: int) -> _CycleValue:
        try:
            index = values.index(current)
        except ValueError:
            index = -1 if offset > 0 else 0
        return values[(index + offset) % len(values)]

    def cycle_filter(self, offset: int = 1) -> None:
        self.set_fit_filter(self._cycled(FIT_FILTERS, self.fit_filter, offset))

    def set_fit_filter(self, value: str) -> None:
        if value not in FIT_FILTERS:
            raise ValueError(f"unknown fit filter: {value}")
        self.fit_filter = value
        self.selected_index = 0
        self.status = f"Fit filter: {self.fit_filter}"

    def cycle_sort(self, offset: int = 1) -> None:
        self.set_sort(self._cycled(SORT_FIELDS, self.sort_by, offset))

    def set_sort(self, value: str) -> None:
        if value not in SORT_FIELDS:
            raise ValueError(f"unknown sort field: {value}")
        self.sort_by = cast(SortField, value)
        self.selected_index = 0
        self.status = f"Sorted by {self.sort_by}"

    def _reconfigure(self, **changes: object) -> None:
        values: dict[str, object] = {
            "partition": self.resources.partition,
            "nodes": self.resources.nodes.maximum,
            "gcds": self.resources.gcds.maximum,
            "input_tokens": self.serving.input_tokens,
            "output_tokens": self.serving.max_output_tokens,
            "context": self.serving.max_model_length,
            "concurrency": self.serving.concurrent_sequences,
            "duration": str(self.resources.walltime),
            "use_case": self.serving.use_case.value,
            "target_rate": str(self.serving.request_arrival_rate_per_second or ""),
            "max_ttft": str(self.serving.latency_targets.max_ttft_seconds or ""),
            "max_tpot": str(self.serving.latency_targets.max_itl_seconds or ""),
            "slo_attainment": str(self.serving.required_slo_attainment or ""),
            "maximum_failure_rate": str(self.serving.maximum_failure_rate or ""),
            "minimum_quality": str(self.serving.minimum_quality_score or ""),
        }
        values.update(changes)
        self.update_configuration(**values)  # type: ignore[arg-type]

    def cycle_use_case(self, offset: int = 1) -> None:
        value = self._cycled(self.available_use_cases, self.serving.use_case.value, offset)
        self.set_use_case(value)

    @property
    def available_use_cases(self) -> tuple[str, ...]:
        """Use cases represented by at least one non-gated catalog model."""

        capabilities = {
            capability
            for model in self.catalogs.models.models
            if not model.gated
            for capability in model.capabilities
        }
        return tuple(
            use_case
            for use_case in USE_CASE_FILTERS
            if _USE_CASE_CAPABILITY[use_case] in capabilities
        )

    def set_use_case(self, value: str) -> None:
        if value not in self.available_use_cases:
            raise ValueError(f"task is unavailable in the model catalog: {value}")
        self._reconfigure(use_case=value)
        self.status = f"Task: {value}"

    def cycle_allocation(self, offset: int = 1) -> None:
        current = next(
            (
                preset
                for preset in ALLOCATION_PRESETS
                if preset[1:]
                == (
                    self.resources.partition,
                    self.resources.nodes.maximum,
                    self.resources.gcds.maximum,
                )
            ),
            ALLOCATION_PRESETS[-1 if offset > 0 else 0],
        )
        preset = self._cycled(ALLOCATION_PRESETS, current, offset)
        self.set_allocation(preset[0])

    def set_allocation(self, selected_label: str) -> None:
        try:
            label, partition, nodes, gcds = next(
                preset for preset in ALLOCATION_PRESETS if preset[0] == selected_label
            )
        except StopIteration as error:
            raise ValueError(f"unknown allocation preset: {selected_label}") from error
        self._reconfigure(partition=partition, nodes=nodes, gcds=gcds)
        self.status = f"Allocation: {label} GCDs"

    def cycle_context(self, offset: int = 1) -> None:
        minimum = self.serving.input_tokens + self.serving.max_output_tokens
        options = tuple(value for value in CONTEXT_OPTIONS if value >= minimum) or (
            self.serving.max_model_length,
        )
        context = self._cycled(options, self.serving.max_model_length, offset)
        self.set_context(context)

    def set_context(self, context: int) -> None:
        if context < self.serving.input_tokens + self.serving.max_output_tokens:
            raise ValueError("context must cover input and output tokens")
        self._reconfigure(context=context)
        self.status = f"Context: {context:,} tokens"

    def cycle_concurrency(self, offset: int = 1) -> None:
        concurrency = self._cycled(CONCURRENCY_OPTIONS, self.serving.concurrent_sequences, offset)
        self.set_concurrency(concurrency)

    def set_concurrency(self, concurrency: int) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self._reconfigure(concurrency=concurrency)
        self.status = f"Concurrency: {concurrency}"

    def cycle_target_rate(self, offset: int = 1) -> None:
        current = self.serving.request_arrival_rate_per_second or TARGET_RATE_OPTIONS[0]
        target = self._cycled(TARGET_RATE_OPTIONS, current, offset)
        self.set_target_rate(target)

    def set_target_rate(self, target: Decimal) -> None:
        if target <= 0:
            raise ValueError("target rate must be positive")
        self._reconfigure(target_rate=str(target))
        self.status = f"Target rate: {target} req/s"

    def toggle_mark(self) -> None:
        if not self.selected:
            return
        if self.marked_plan_id == self.selected.plan_id:
            self.marked_plan_id, self.compare_visible = None, False
            self.status = "Comparison mark cleared"
        else:
            self.marked_plan_id = self.selected.plan_id
            self.status = f"Marked {self.selected.plan.model_id}"

    def toggle_compare(self) -> None:
        if not self.marked_plan_id:
            self.status = "Press m to mark a row before comparing."
        elif not self.selected or self.selected.plan_id == self.marked_plan_id:
            self.status = "Move to another row, then press c to compare."
        else:
            self.compare_visible = not self.compare_visible
            self.details_visible = True

    def update_configuration(
        self,
        *,
        partition: str,
        nodes: int,
        gcds: int,
        input_tokens: int,
        output_tokens: int,
        context: int,
        concurrency: int,
        duration: str,
        use_case: str,
        target_rate: str,
        max_ttft: str,
        max_tpot: str,
        slo_attainment: str,
        maximum_failure_rate: str,
        minimum_quality: str,
    ) -> None:
        gcds_per_node = max(1, (gcds + nodes - 1) // nodes)
        full_node = partition == "standard-g"
        walltime = Duration.model_validate(duration)
        resources = ResourceEnvelope(
            partition=partition,
            nodes=nodes,
            gcds=gcds,
            cpu_cores_per_node=56 if full_node else min(56, gcds_per_node * 8),
            cpu_memory_per_node=MemoryAmount.model_validate(
                f"{512 if full_node else min(512, gcds_per_node * 64)} GB"
            ),
            walltime=walltime,
            allow_partial_node=not full_node,
            allow_multi_node=nodes > 1,
        )
        serving = ServingRequirements(
            mode=ServingMode.ONLINE,
            use_case=UseCase(use_case),
            input_tokens=input_tokens,
            max_output_tokens=output_tokens,
            max_model_length=context,
            concurrent_sequences=concurrency,
            request_arrival_rate_per_second=(Decimal(target_rate) if target_rate else None),
            service_duration=walltime,
            latency_targets=LatencyTargets(
                max_ttft_seconds=Decimal(max_ttft) if max_ttft else None,
                max_itl_seconds=Decimal(max_tpot) if max_tpot else None,
            ),
            throughput_targets=self.serving.throughput_targets,
            token_length_distribution=(
                TokenLengthPoint(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    probability=Decimal(1),
                ),
            ),
            required_slo_attainment=(Decimal(slo_attainment) if slo_attainment else None),
            maximum_failure_rate=(Decimal(maximum_failure_rate) if maximum_failure_rate else None),
            minimum_quality_score=(Decimal(minimum_quality) if minimum_quality else None),
            minimum_hbm_headroom_fraction=self.serving.minimum_hbm_headroom_fraction,
            allowed_runtimes=self.serving.allowed_runtimes,
            allowed_quantizations=self.serving.allowed_quantizations,
        )
        bundle = self._resource_first(self.catalogs, resources, serving)
        self.resources, self.serving, self.bundle = resources, serving, bundle
        self._resource_bundle, self.mode = bundle, "models"
        self.selected_index, self.marked_plan_id, self.compare_visible = 0, None, False
        self.status = f"Recalculated {len(bundle.model_groups)} model recommendations."

    def enter_plan_mode(self) -> None:
        if not self.selected:
            return
        model_id = self.selected.plan.model_id
        bounds = ResourceEnvelope(
            partition="auto",
            nodes=CountRange(minimum=1, maximum=max(2, self.resources.nodes.maximum)),
            gcds=CountRange(minimum=1, maximum=max(16, self.resources.gcds.maximum)),
            walltime=self.resources.walltime,
            allow_partial_node=True,
            allow_multi_node=True,
            enabled_partitions=("small-g", "standard-g"),
        )
        self.bundle = build_model_first_bundle(
            model_selector=ModelSelector(exact_model=model_id),
            serving=self.serving,
            resources=bounds,
            catalogs=self.catalogs,
        )
        self.mode, self.selected_index = "plan", 0
        self.marked_plan_id, self.compare_visible = None, False
        self.status = f"Resource plans for {model_id}. Press b to return."

    def leave_plan_mode(self) -> None:
        if self.mode == "plan" and self._resource_bundle:
            self.bundle, self.mode, self.selected_index = self._resource_bundle, "models", 0
            self.status = "Returned to model recommendations."

    def allocation_summary(self) -> str:
        hbm = self.resources.gcds.maximum * self.catalogs.hardware.gcd.hbm_per_gcd.value
        return (
            f"LUMI-G  {self.resources.partition}  {self.resources.nodes.maximum} node(s)  "
            f"{self.resources.gcds.maximum} GCDs  {hbm:.0f} GB HBM  {self.resources.walltime}"
        )

    def workload_summary(self) -> str:
        mode = "resource plans" if self.mode == "plan" else "model fit"
        return (
            f"{self.serving.use_case.value} · vLLM goodput fit  "
            f"ctx {self.serving.max_model_length:,}  "
            f"concurrency {self.serving.concurrent_sequences}  "
            f"target {self.serving.request_arrival_rate_per_second or '-'} req/s  mode: {mode}"
        )


class LumiFitApplication:
    """Prompt Toolkit full-screen application."""

    def __init__(self, state: DashboardState) -> None:
        self.state = state
        self.edit_visible = self.help_visible = False
        self._build_layout()
        self.application: Application[None] = Application(
            layout=Layout(self.container, focused_element=self.table),
            key_bindings=self._bindings(),
            full_screen=True,
            mouse_support=True,
            style=self._style(),
        )

    def _build_layout(self) -> None:
        self.header = Window(FormattedTextControl(self._header), height=2, style="class:header")
        self.search = TextArea(
            height=1,
            multiline=False,
            wrap_lines=False,
            accept_handler=self._accept_search,
        )
        self.search.buffer.on_text_changed += self._search_changed
        self.search_box = Frame(self.search, title="Search [/]", width=26)
        selector_specs = (
            ("task", "Task", 14),
            ("allocation", "Allocation", 20),
            ("context", "Context", 12),
            ("concurrency", "Conc.", 10),
            ("target", "Target", 12),
            ("filter", "Fit [f]", 15),
            ("sort", "Sort [s]", 20),
        )
        self.selector_controls: dict[str, FormattedTextControl] = {}
        self.selector_boxes: dict[str, Frame] = {}
        for name, title, width in selector_specs:
            control = FormattedTextControl(
                lambda name=name: self._selector_text(name),
                focusable=True,
            )
            self.selector_controls[name] = control
            self.selector_boxes[name] = Frame(
                Window(control, height=1, always_hide_cursor=True),
                title=title,
                width=width,
            )
        self.control_bar = DynamicContainer(self._controls)
        self.table_control = FormattedTextControl(self._rows, focusable=True)
        self.table = Window(self.table_control, always_hide_cursor=True, style="class:table")
        self.details = Window(
            FormattedTextControl(self._detail), wrap_lines=True, style="class:details"
        )
        self.status = Window(FormattedTextControl(self._status), height=1)
        self.keys = Window(FormattedTextControl(self._keys), height=1, style="class:keys")
        self.root = HSplit(
            [self.header, self.control_bar, DynamicContainer(self._body), self.status, self.keys]
        )
        self.edit_dialog = self._edit_dialog()
        self.help_dialog = self._help_dialog()
        self.container = FloatContainer(
            content=self.root,
            floats=[
                Float(
                    content=ConditionalContainer(
                        self.edit_dialog, Condition(lambda: self.edit_visible)
                    )
                ),
                Float(
                    content=ConditionalContainer(
                        self.help_dialog, Condition(lambda: self.help_visible)
                    )
                ),
            ],
        )

    def _controls(self) -> HSplit | VSplit:
        try:
            columns = get_app().output.get_size().columns
        except Exception:
            columns = 120
        task = self.selector_boxes["task"]
        allocation = self.selector_boxes["allocation"]
        context = self.selector_boxes["context"]
        concurrency = self.selector_boxes["concurrency"]
        target = self.selector_boxes["target"]
        fit_filter = self.selector_boxes["filter"]
        sort = self.selector_boxes["sort"]
        if columns >= _SINGLE_ROW_CONTROLS_MIN_COLUMNS:
            return VSplit(
                [
                    self.search_box,
                    task,
                    allocation,
                    context,
                    concurrency,
                    target,
                    fit_filter,
                    sort,
                ],
                padding=1,
            )
        return HSplit(
            [
                VSplit([self.search_box, task, allocation], padding=1),
                VSplit([context, concurrency, target, fit_filter, sort], padding=1),
            ]
        )

    def _selector_text(self, name: str) -> StyleAndTextTuples:
        if name == "task":
            value = self.state.serving.use_case.value
        elif name == "allocation":
            value = next(
                (
                    preset[0]
                    for preset in ALLOCATION_PRESETS
                    if preset[1:]
                    == (
                        self.state.resources.partition,
                        self.state.resources.nodes.maximum,
                        self.state.resources.gcds.maximum,
                    )
                ),
                f"{self.state.resources.partition} · {self.state.resources.gcds.maximum}",
            )
        elif name == "context":
            value = f"{self.state.serving.max_model_length // 1024}k"
        elif name == "concurrency":
            value = str(self.state.serving.concurrent_sequences)
        elif name == "target":
            target = self.state.serving.request_arrival_rate_per_second
            value = f"{target}/s" if target is not None else "none"
        elif name == "filter":
            value = self.state.fit_filter
        else:
            value = self.state.sort_by
        focused = (
            hasattr(self, "application")
            and self.application.layout.current_control is self.selector_controls[name]
        )
        style = "class:control.focused" if focused else "class:control.value"
        return [(style, f"‹{value}›")]

    def _body(self) -> HSplit | VSplit:
        try:
            columns = get_app().output.get_size().columns
        except Exception:
            columns = 120
        noun = "resource plans" if self.state.mode == "plan" else "model recommendations"
        title = f"{len(self.state.candidates)} {noun}"
        table = Frame(self.table, title=title)
        detail = Frame(self.details, title="Plan details")
        if self.state.details_visible and columns >= _SIDE_BY_SIDE_MIN_COLUMNS:
            return VSplit([table, detail], padding=1)
        return HSplit([table, detail] if self.state.details_visible else [table])

    def _header(self) -> StyleAndTextTuples:
        return [
            ("class:brand", "  lumi-fit"),
            ("class:muted", "  local LUMI serving advisor\n"),
            ("class:hardware", f"  {self.state.allocation_summary()}\n"),
            ("", f"  {self.state.workload_summary()}"),
        ]

    def _visible(self) -> tuple[int, tuple[CandidateResult, ...]]:
        try:
            page = max(4, get_app().output.get_size().rows - 15)
        except Exception:
            page = 16
        start = max(0, self.state.selected_index - page // 2)
        start = min(start, max(0, len(self.state.candidates) - page))
        return start, self.state.candidates[start : start + page]

    @staticmethod
    def _row(
        item: CandidateResult,
        columns: tuple[_TableColumn, ...],
        marker: str,
    ) -> str:
        analysis, plan = item.analysis, item.plan
        params = (
            f"{Decimal(item.model_parameters) / Decimal(1_000_000_000):.1f}B"
            if item.model_parameters
            else "-"
        )
        fit = item.lumi_vllm_fit
        quality = quality_label(fit)
        if columns is _COMPACT_COLUMNS and quality == "NOT REQUIRED":
            quality = "NOT REQ"
        return _table_line(
            {
                "marker": marker,
                "model": plan.model_id,
                "params": params,
                "eligibility": _eligibility_label(fit),
                "hbm": _hbm_label(analysis, precision=0),
                "safe": _safe_concurrency_label(item),
                "rate": _throughput_label(analysis),
                "gcds": f"{plan.used_gcds}/{plan.allocated_gcds}",
                "cost": _cost_label(analysis),
                "estimated": estimated_coverage_label(fit),
                "quality": quality,
                "fit": fit_score_label(fit),
            },
            columns,
        )

    def _rows(self) -> StyleAndTextTuples:
        try:
            width = get_app().output.get_size().columns
            if self.state.details_visible and width >= _SIDE_BY_SIDE_MIN_COLUMNS:
                width -= _DETAIL_PANEL_ESTIMATED_COLUMNS
        except Exception:
            width = 120
        columns = _COMPACT_COLUMNS if width < 120 else _WIDE_COLUMNS
        heading = _table_line(
            {column.key: column.heading for column in columns},
            columns,
        )
        result: StyleAndTextTuples = [("class:table.header", heading + "\n")]
        start, rows = self._visible()
        if not rows:
            return result + [("class:muted", "\n  No rows match the search/filter.")]
        for offset, item in enumerate(rows):
            selected = start + offset == self.state.selected_index
            mark = "◆" if item.plan_id == self.state.marked_plan_id else ("›" if selected else " ")
            style = "class:row.selected" if selected else "class:row"
            if item.analysis.feasibility.value == "infeasible" and not selected:
                style = "class:row.infeasible"
            result.append((style, f"{self._row(item, columns, mark)}\n"))
        return result

    def _detail(self) -> StyleAndTextTuples:
        item = self.state.selected
        if not item:
            return [("class:muted", "No plan selected.")]
        if self.state.compare_visible and self.state.marked_plan_id:
            marked = next(
                (
                    row
                    for row in self.state.bundle.candidates
                    if row.plan_id == self.state.marked_plan_id
                ),
                None,
            )
            if marked:
                return self._comparison(marked, item)
        plan, analysis = item.plan, item.analysis
        memory, performance, billing = analysis.memory, analysis.performance, analysis.billing
        capacity = item.capacity_envelope
        fit = item.lumi_vllm_fit
        safe_x = capacity.memory_safe_maximum_context if capacity else None
        safe_x_text = f"{safe_x.context_tokens:,}" if safe_x else "-"
        text: StyleAndTextTuples = [
            ("class:detail.title", f"{plan.model_id}\n"),
            ("class:muted", f"{item.plan_id}\n\n"),
            ("class:label", _detail_label("Plan")),
            ("", f"{analysis.feasibility.value}"),
            ("", f" / memory {analysis.fit_band.value}" if analysis.fit_band else ""),
            ("", "\n"),
            ("class:label", _detail_label("Serveable")),
            ("", f"{_eligibility_label(fit)}\n"),
            ("class:label", _detail_label("Est. fit")),
            ("", f"{estimated_coverage_label(fit)} (analytical)\n"),
            ("class:label", _detail_label("Quality")),
            ("", f"{quality_label(fit)}\n"),
            ("class:label", _detail_label("Valid fit")),
            ("", f"{fit_score_label(fit)}\n"),
            ("class:label", _detail_label("Layout")),
            (
                "",
                f"TP{plan.tensor_parallel} × {plan.replicas}; "
                f"{plan.used_gcds}/{plan.allocated_gcds} GCDs\n",
            ),
            ("class:label", _detail_label("Runtime")),
            ("", f"{plan.runtime_id} · {plan.quantization} · {plan.partition}\n"),
            ("class:label", _detail_label("Request")),
            (
                "",
                f"context {plan.runtime_options.max_model_length:,}, "
                f"concurrency {plan.runtime_options.max_concurrent_sequences}\n",
            ),
            ("class:label", _detail_label("HBM")),
            ("", f"{_hbm_label(analysis)} conservative\n" if memory else "unknown\n"),
            ("class:label", _detail_label("Throughput")),
            (
                "",
                f"{_throughput_label(analysis)} tok/s (analytical)\n"
                if performance and performance.aggregate_output_tokens_per_second
                else "unknown\n",
            ),
            ("class:label", _detail_label("Billing")),
            (
                "",
                f"{_cost_label(analysis)} GPU-h/h; {billing.total_gpu_hours:.1f} total\n"
                if billing
                else "unknown\n",
            ),
            ("class:label", _detail_label("Capacity")),
            (
                "",
                f"safe concurrency {_safe_concurrency_label(item)}, safe context {safe_x_text}\n",
            ),
        ]
        failed = [constraint for constraint in analysis.constraints if not constraint.passed]
        if failed:
            text.append(("class:error", "\nWhy it does not fit\n"))
            text.extend(("class:muted", f"• {constraint.message}\n") for constraint in failed[:4])
        else:
            text.append(("class:muted", "\nPress p to inspect resource plans for this model."))
        return text

    @staticmethod
    def _comparison(left: CandidateResult, right: CandidateResult) -> StyleAndTextTuples:
        def values(item: CandidateResult) -> tuple[str, ...]:
            analysis = item.analysis
            return (
                _eligibility_label(item.lumi_vllm_fit),
                estimated_coverage_label(item.lumi_vllm_fit),
                quality_label(item.lumi_vllm_fit),
                fit_score_label(item.lumi_vllm_fit),
                _hbm_label(analysis),
                _safe_concurrency_label(item),
                str(item.plan.allocated_gcds),
                _cost_label(analysis),
            )

        a, b = values(left), values(right)
        result: StyleAndTextTuples = [
            ("class:detail.title", "Plan comparison\n"),
            (
                "class:muted",
                f"◆ {_short(left.plan.model_id, 24)}\n› {_short(right.plan.model_id, 24)}\n\n",
            ),
        ]
        for index, label in enumerate(
            (
                "Serveable",
                "Est. fit",
                "Quality",
                "Valid fit",
                "HBM",
                "Safe conc",
                "GCDs",
                "GPU-h/h",
            )
        ):
            result.extend(
                [("class:label", f"{label:<10}"), ("", f"{a[index]:>12}  →  {b[index]:<12}\n")]
            )
        result.append(("class:muted", "\nPress c to close comparison."))
        return result

    def _status(self) -> StyleAndTextTuples:
        return [
            ("class:mode", f" {self.state.mode.upper()} "),
            ("class:status", f"  {self.state.status}"),
        ]

    def _keys(self) -> StyleAndTextTuples:
        if self.edit_visible:
            hint = "Tab fields  Enter apply/cancel button  Escape close"
        elif self.help_visible:
            hint = "Escape or Enter close"
        elif self._controls_focused():
            hint = (
                "←/→ or Enter/Space change this control's value  ↓ back to table  "
                "Tab/⇧Tab controls  Esc table  q quit"
            )
        else:
            hint = (
                "↑↓/jk move  Enter details  Tab controls  Space next value  / search  "
                "e advanced  m mark  c compare  p plan  h help  q quit"
            )
        return [("class:keys", f" {hint} ")]

    def _edit_dialog(self) -> Dialog:
        resources, serving = self.state.resources, self.state.serving
        defaults = {
            "partition": resources.partition,
            "nodes": str(resources.nodes.maximum),
            "gcds": str(resources.gcds.maximum),
            "input_tokens": str(serving.input_tokens),
            "output_tokens": str(serving.max_output_tokens),
            "context": str(serving.max_model_length),
            "concurrency": str(serving.concurrent_sequences),
            "duration": str(resources.walltime),
            "use_case": serving.use_case.value,
            "target_rate": str(serving.request_arrival_rate_per_second or ""),
            "max_ttft": str(serving.latency_targets.max_ttft_seconds or ""),
            "max_tpot": str(serving.latency_targets.max_itl_seconds or ""),
            "slo_attainment": str(serving.required_slo_attainment or ""),
            "failure_rate": str(serving.maximum_failure_rate or ""),
            "minimum_quality": str(serving.minimum_quality_score or ""),
        }
        labels = {
            "partition": "Partition (small-g/standard-g/auto)",
            "nodes": "Nodes",
            "gcds": "Total GCDs",
            "input_tokens": "Typical input tokens",
            "output_tokens": "Maximum output tokens",
            "context": "Model context",
            "concurrency": "Concurrent sequences",
            "duration": "Duration",
            "use_case": "Use case",
            "target_rate": "Target requests/second",
            "max_ttft": "Maximum TTFT seconds",
            "max_tpot": "Maximum TPOT seconds",
            "slo_attainment": "Required SLO attainment",
            "failure_rate": "Maximum failure rate",
            "minimum_quality": "Minimum task quality (0-100)",
        }
        self.edit_fields = {
            name: TextArea(text=value, height=1, multiline=False)
            for name, value in defaults.items()
        }
        rows = [
            VSplit([Label(labels[name], width=36), widget], padding=1)
            for name, widget in self.edit_fields.items()
        ]
        return Dialog(
            title="Simulated LUMI allocation and workload",
            body=HSplit(rows),
            buttons=[
                Button(text="Apply", handler=self._apply_edit),
                Button(text="Cancel", handler=self._cancel_edit),
            ],
            width=76,
            modal=True,
        )

    def _help_dialog(self) -> Dialog:
        text = (
            "On a control (Tab to reach one):\n"
            "  ← / → / Enter / Space   change its value      ↓ / Escape       back to table\n\n"
            "On the results table:\n"
            "  ↑/↓ or j/k       select result row       g/G             top/bottom\n"
            "  Enter            toggle details panel    /               focus search\n"
            "  f / s            filter / sort           m / c           mark / compare\n"
            "  p / b            plan mode/back          e               advanced editor\n\n"
            "This interface only calculates recommendations. It never connects to LUMI,\n"
            "submits jobs, generates Slurm scripts, or starts a serving engine."
        )
        return Dialog(
            title="lumi-fit help",
            body=Label(text),
            buttons=[Button(text="Close", handler=self._close_help)],
            width=76,
            modal=True,
        )

    def _open_edit(self) -> None:
        current = {
            "partition": self.state.resources.partition,
            "nodes": str(self.state.resources.nodes.maximum),
            "gcds": str(self.state.resources.gcds.maximum),
            "input_tokens": str(self.state.serving.input_tokens),
            "output_tokens": str(self.state.serving.max_output_tokens),
            "context": str(self.state.serving.max_model_length),
            "concurrency": str(self.state.serving.concurrent_sequences),
            "duration": str(self.state.resources.walltime),
            "use_case": self.state.serving.use_case.value,
            "target_rate": str(self.state.serving.request_arrival_rate_per_second or ""),
            "max_ttft": str(self.state.serving.latency_targets.max_ttft_seconds or ""),
            "max_tpot": str(self.state.serving.latency_targets.max_itl_seconds or ""),
            "slo_attainment": str(self.state.serving.required_slo_attainment or ""),
            "failure_rate": str(self.state.serving.maximum_failure_rate or ""),
            "minimum_quality": str(self.state.serving.minimum_quality_score or ""),
        }
        for name, value in current.items():
            self.edit_fields[name].text = value
        self.edit_visible = True
        self.application.layout.focus(self.edit_fields["partition"])

    def _apply_edit(self) -> None:
        try:
            value = {name: field.text.strip() for name, field in self.edit_fields.items()}
            self.state.update_configuration(
                partition=value["partition"],
                nodes=int(value["nodes"]),
                gcds=int(value["gcds"]),
                input_tokens=int(value["input_tokens"]),
                output_tokens=int(value["output_tokens"]),
                context=int(value["context"]),
                concurrency=int(value["concurrency"]),
                duration=value["duration"],
                use_case=value["use_case"],
                target_rate=value["target_rate"],
                max_ttft=value["max_ttft"],
                max_tpot=value["max_tpot"],
                slo_attainment=value["slo_attainment"],
                maximum_failure_rate=value["failure_rate"],
                minimum_quality=value["minimum_quality"],
            )
        except (ValueError, KeyError, InvalidOperation) as error:
            self.state.status = f"Configuration error: {error}"
            self.application.invalidate()
            return
        self._cancel_edit()

    def _cancel_edit(self) -> None:
        self.edit_visible = False
        self.application.layout.focus(self.table)
        self.application.invalidate()

    def _close_help(self) -> None:
        self.help_visible = False
        self.application.layout.focus(self.table)
        self.application.invalidate()

    def _accept_search(self, _: object) -> bool:
        self.application.layout.focus(self.table)
        return True

    def _search_changed(self, _: object) -> None:
        self.state.query, self.state.selected_index = self.search.text, 0
        if hasattr(self, "application"):
            self.application.invalidate()

    def _normal(self) -> bool:
        return (
            not self.edit_visible
            and not self.help_visible
            and self.application.layout.current_control is self.table_control
        )

    def _controls_focused(self) -> bool:
        return (
            not self.edit_visible
            and not self.help_visible
            and hasattr(self, "application")
            and self.application.layout.current_control in self.selector_controls.values()
        )

    def _bindings(self) -> KeyBindings:
        keys, normal = KeyBindings(), Condition(self._normal)
        controls_active = Condition(self._controls_focused)
        dashboard_active = Condition(lambda: not self.edit_visible and not self.help_visible)

        def bind(key: str, callback: Callable[[], None]) -> None:
            def handler(event: KeyPressEvent) -> None:
                callback()
                event.app.invalidate()

            keys.add(key, filter=normal)(handler)

        for key in ("down", "j"):
            bind(key, lambda: self.state.move(1))
        for key in ("up", "k"):
            bind(key, lambda: self.state.move(-1))
        for key in ("pagedown", "c-d"):
            bind(key, lambda: self.state.move(10))
        for key in ("pageup", "c-u"):
            bind(key, lambda: self.state.move(-10))
        for key in ("g", "home"):
            bind(key, lambda: setattr(self.state, "selected_index", 0))
        for key in ("G", "end"):
            bind(
                key,
                lambda: setattr(
                    self.state, "selected_index", max(0, len(self.state.candidates) - 1)
                ),
            )
        bind("f", self.state.cycle_filter)
        bind("s", self.state.cycle_sort)
        bind(
            "enter", lambda: setattr(self.state, "details_visible", not self.state.details_visible)
        )
        bind("m", self.state.toggle_mark)
        bind("c", self.state.toggle_compare)
        bind("p", self.state.enter_plan_mode)
        bind("b", self.state.leave_plan_mode)
        bind("e", self._open_edit)

        def cycle_control(offset: int) -> None:
            current = self.application.layout.current_control
            name = next(
                (name for name, control in self.selector_controls.items() if control is current),
                None,
            )
            actions: dict[str, Callable[[int], None]] = {
                "task": self.state.cycle_use_case,
                "allocation": self.state.cycle_allocation,
                "context": self.state.cycle_context,
                "concurrency": self.state.cycle_concurrency,
                "target": self.state.cycle_target_rate,
                "filter": self.state.cycle_filter,
                "sort": self.state.cycle_sort,
            }
            if name is not None:
                try:
                    actions[name](offset)
                except (ValueError, KeyError, InvalidOperation) as error:
                    self.state.status = f"Cannot change {name}: {error}"
                self.application.invalidate()

        keys.add("right", filter=controls_active)(lambda event: cycle_control(1))
        keys.add("left", filter=controls_active)(lambda event: cycle_control(-1))
        keys.add("enter", filter=controls_active)(lambda event: cycle_control(1))
        keys.add(" ", filter=controls_active)(lambda event: cycle_control(1))

        @keys.add("down", filter=controls_active)
        def leave_controls(event: KeyPressEvent) -> None:
            event.app.layout.focus(self.table)

        @keys.add("tab", filter=dashboard_active)
        def focus_next(event: KeyPressEvent) -> None:
            event.app.layout.focus_next()

        @keys.add("s-tab", filter=dashboard_active)
        def focus_previous(event: KeyPressEvent) -> None:
            event.app.layout.focus_previous()

        def open_help() -> None:
            self.help_visible = True
            self.application.layout.focus(self.help_dialog)

        for key in ("h", "?"):
            bind(key, open_help)

        @keys.add("/")
        def search(event: KeyPressEvent) -> None:
            if self._normal():
                self.application.layout.focus(self.search)

        @keys.add("escape")
        def escape(event: KeyPressEvent) -> None:
            if self.edit_visible:
                self._cancel_edit()
            elif self.help_visible:
                self._close_help()
            else:
                self.application.layout.focus(self.table)

        @keys.add("q", filter=normal)
        @keys.add("c-q")
        def quit_app(event: KeyPressEvent) -> None:
            get_app().exit()

        return keys

    @staticmethod
    def _style() -> Style:
        return Style.from_dict(
            {
                "header": "bg:#151923 #d8dee9",
                "brand": "bold #8be9fd",
                "hardware": "#a3be8c",
                "muted": "#7f8c9d",
                "table": "bg:#10131a #d8dee9",
                "table.header": "bold bg:#242b3a #8be9fd",
                "row": "#d8dee9",
                "row.selected": "bold bg:#3b4252 #ffffff",
                "row.infeasible": "#bf616a",
                "details": "bg:#171b24 #d8dee9",
                "detail.title": "bold #88c0d0",
                "label": "bold #b48ead",
                "error": "bold #bf616a",
                "status": "bg:#242b3a #d8dee9",
                "mode": "bold bg:#8be9fd #10131a",
                "keys": "bg:#10131a #7f8c9d",
                "frame.label": "bold #88c0d0",
                "button": "bg:#4c566a #eceff4",
                "button.focused": "bold bg:#88c0d0 #10131a",
                "text-area": "bg:#10131a #eceff4",
                "control.value": "#8be9fd",
                "control.focused": "bold bg:#88c0d0 #10131a",
            }
        )

    def run(self) -> None:
        self.application.run()


def run_tui(repository: Path | None = None) -> None:
    """Launch the full-screen dashboard used by bare ``lumi-fit``."""

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "the interactive dashboard requires a TTY; use `lumi-fit models`, "
            "`lumi-fit plan`, or `lumi-fit compare` for non-interactive use"
        )
    LumiFitApplication(DashboardState.create(repository)).run()
