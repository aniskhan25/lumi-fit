# lumi-fit

`lumi-fit` is a local model-serving advisor for the LUMI supercomputer.
It answers both sides of the allocation question:

- Which models and serving configurations fit inside a planned allocation?
- Which LUMI resources are required to serve a selected model at a requested
  context, concurrency, throughput, or latency?

Phases 0 through 4 are complete. The repository now includes
executable contracts,
versioned LUMI catalogs, exact billing, an initial curated model catalog, local
Hugging Face metadata import, per-rank serving-memory analysis, feasibility
checks, both planning directions, technical serveability, heuristic demand
coverage, validated LUMI-vLLM fit analysis, a production CLI, saved
explanations, and a full-screen terminal dashboard. Recommendations work from
analytical estimates immediately; validated fit remains `UNSCORED` until the
declared workload has matching SLO-goodput evidence.

The central design document is [docs/implementation-plan.md](docs/implementation-plan.md).
It defines the domain model, LUMI hardware assumptions, feasibility rules,
estimators, fit score, CLI contracts, evidence model, testing strategy,
and phased implementation roadmap.

The current executable contracts are documented in
[docs/contracts.md](docs/contracts.md).
The Phase 1 estimator is documented in
[docs/memory-model.md](docs/memory-model.md).
The fixed-allocation search is documented in
[docs/resource-first.md](docs/resource-first.md).
Phase 3 scoring and interfaces are documented in
[docs/scoring.md](docs/scoring.md) and [docs/cli.md](docs/cli.md).
Phase 4 recommendation-quality work is designed in
[docs/phase4-recommendation-quality.md](docs/phase4-recommendation-quality.md).
The memory-safe capacity search is documented in
[docs/capacity-envelopes.md](docs/capacity-envelopes.md).
The decision replacing weighted objective scoring is recorded in
[ADR 0007](docs/adr/0007-workload-specific-vllm-fit.md).

The product is recommendation-only. It runs locally and does not connect to
LUMI, generate Slurm/vLLM launch artifacts, download weights, or operate jobs.

## Working objective

Generate and rank complete model-serving plans:

```text
model artifact + precision/quantization + context + concurrency
+ serving runtime/container + batching limits
+ partition + nodes + GCDs + CPU/RAM
+ tensor/pipeline parallelism + replicas
+ service duration
```

Every recommendation must explain:

- whether the plan is feasible;
- what resources it requests and how LUMI will bill them;
- the expected latency, throughput, and uncertainty;
- separate serveability, estimated coverage, task quality, and validated fit;
- the assumptions and evidence behind every estimate;
- which estimates are heuristic or lack comparable evidence.

## Project status

- Architecture and implementation plan: implemented through Phase 4
- Phase 0 contracts and catalogs: complete
- Phase 1 model-first memory planning: complete
- Phase 2 resource-first model discovery: complete
- Phase 3 scoring, CLI, and interactive terminal interface: complete
- Exact LUMI GPU billing engine: complete
- Technical serveability, analytical coverage, and validated-fit engines: complete
- LUMI hardware, partition, billing, and runtime catalogs: initial profiles
- Curated model catalog: 7 revision-pinned BF16 artifacts across 5 architecture families
- Model coverage/provenance contract and local coverage report: complete
- Memory-safe context/concurrency envelopes and allocation neighbors: complete
- Weighted composites and objective sensitivity: superseded by ADR 0007
- Local filters, deterministic sorting, detail/compare views, and CSV export: complete
- Curated performance evidence catalog: not started
- Production CLI/TUI: complete
- Deployment, Slurm rendering, live probes, and runtime launch: out of scope
- Phase 4A catalog expansion foundation: complete
- Phase 4B capacity envelopes: complete
- Phase 4C comparison and score sensitivity: superseded by ADR 0007
- Phase 4D interface polish and CSV export: complete
- Phase 5 curated evidence and estimator refinement: next

## Phase 3 CLI

Run the interactive dashboard:

```bash
PYTHONPATH=src python3 -m lumi_fit.cli
```

Bare `lumi-fit` opens directly on a ranked, scrollable model table. Selectable
boxes at the top expose search, task, allocation, context, concurrency, target
request rate, fit filter, and sort order. Use `Tab`/`Shift-Tab` or the mouse to
select one. `Left`/`Right` changes the value immediately and `Enter`/`Space`
advances to the next value; recommendations recalculate locally without opening
a dialog. Use `e` only when the complete advanced workload/SLO editor is needed,
`m`/`c` to compare, and `p` to see resource plans for the selected model. The
dashboard remains recommendation-only and never connects to LUMI or launches a
serving engine.

Or call an automation-friendly workflow directly:

```bash
PYTHONPATH=src python3 -m lumi_fit.cli models \
  --resources examples/resources-one-node-four-gcds.yaml \
  --serving examples/serving-online.yaml \
  --model-selector examples/models-coding.yaml
```

Use `--output recommendation.json` to save the full analysis, then inspect it
without recomputation using `lumi-fit explain recommendation.json --plan PLAN_ID`.

Inspect the curated matrix and its versioned coverage/provenance checks locally:

```bash
PYTHONPATH=src python3 -m lumi_fit.cli catalog coverage
```

## Development validation

The test suite can run without installing the package:

```bash
PYTHONPATH=src pytest
```

Regenerate the committed JSON Schemas after changing a public contract:

```bash
PYTHONPATH=src python3 scripts/export_schemas.py
```
