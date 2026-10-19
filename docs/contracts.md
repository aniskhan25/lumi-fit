# Public contracts

The executable contracts live under `src/lumi_fit/domain/`. Pydantic models are
the source of truth; generated JSON Schemas under `data/schemas/` support other
interfaces and editors.

## Input contracts

- `ResourceEnvelope`: fixed allocation in resource-first mode or search bounds
  in model-first mode.
- `ServingRequirements`: context, concurrency, online/batch mode, traffic,
  prompt/output-length distribution, latency/throughput targets, SLO
  attainment, failure tolerance, quality floor, duration, and runtime policy.
- `ModelSelector`: exact model, shortlist, patterns, use case, capabilities,
  families, licenses, languages, and inclusive parameter bounds.
- `ModelCatalog`: conceptual models and concrete serveable artifacts.
- `ServingPlan`: immutable model/runtime/allocation/parallelism configuration.
- `RecommendationBundle`: versioned candidates, analyses, catalog versions, and
  fingerprints consumed by explanation and local export. Resource-first bundles
  include `ModelPlanGroup` records linking each model to its best plan,
  alternatives, diagnostic, and suitability evidence.
- `CandidateResult.model_parameters`: cataloged parameter count retained for
  offline display sorting and deterministic CSV export.
- `CapacityEnvelope`: requested workload point, memory-safe context/concurrency
  boundaries, evaluated probes, performance-capacity status, and optional
  neighboring candidate references. It is attached to feasible and conditional
  `CandidateResult` records.
- `LumiVllmFitAnalysis`: technical serveability from support/memory/context,
  optional quality-requirement status, raw analytical full-node capacity,
  heuristic target-coverage estimate, evidence-backed full-node SLO-goodput
  LCB, and optional validated fit.
- `SloGoodputEvidence`: exact deployment/workload identity, maximum
  SLO-constrained request goodput, 95% LCB, attainment, failure rate, sample
  size, confidence, and evidence provenance.
- `ObjectiveSensitivity` and `PlanComparison` remain readable for saved-bundle
  compatibility, but current planners emit them empty.

Representative files are under `examples/`.

The Phase 1 estimator and current assumptions are documented in
`docs/memory-model.md`.
Phase 2 fixed-allocation discovery is documented in
`docs/resource-first.md`.
Phase 4B bounded capacity search is documented in
`docs/capacity-envelopes.md`.
The current score is documented in `docs/scoring.md`.

`ServingPlan.request_routing` makes replica behavior explicit: `single` for one
replica and `round-robin` for multiple replicas. Runtime concurrency is the
per-replica capacity, while `ServingRequirements.concurrent_sequences` is the
workload-wide request.

`ModelRecord.evaluations` contains optional task-quality evidence. Quality is
`NOT REQUIRED` when the request has no minimum. When a minimum is explicit,
missing evidence produces `UNKNOWN`, not a numeric zero or an implicit pass.

`PerformanceAnalysis` stores ranged analytical capacity estimates with units,
formula IDs, and confidence. These priors remain separate from
`LumiVllmFitAnalysis`: analytical capacity may produce a clearly heuristic
estimated coverage score, while only matching measured or calibrated
SLO-goodput evidence can produce validated fit. A saved `RecommendationBundle` is sufficient for
`lumi-fit explain`; the command does not recompute against newer catalogs.

## Catalog contracts

- `HardwareProfile`: LUMI-G GCD, CPU, topology, and network facts.
- `PartitionCatalog`: allocation granularity, limits, purpose, and billing links.
- `BillingCatalog`: full-node and sub-node billing formulas and slice sizes.
- `RuntimeCatalog`: LUMI container/runtime support and evidence level.
- `ScoringCatalog`: legacy objective weights retained for catalog
  compatibility; current fit scoring does not consume them.
- `ModelCoverageTargets`: versioned minimum catalog representation by family,
  capability, and parameter band.
- `ModelCoverageReport`: offline coverage counts and immutable artifact
  provenance checks used by `lumi-fit catalog coverage`.

Catalog metadata includes a schema version, profile version, validity date,
verification timestamp, and official sources.

## Units

Memory inputs require an explicit suffix such as `64 GB` or `64 GiB`. Values
are converted to integer bytes internally while the source unit is retained.
Durations similarly require a unit such as `30 minutes` or `2 days`.

`GCD`, `MI250X module`, and `node` are separate resource concepts. A generic
unqualified GPU-count field is intentionally absent from the domain model.

## Regenerating JSON Schemas

From the repository root:

```bash
PYTHONPATH=src python3 scripts/export_schemas.py
```

Generated schemas should be committed alongside changes to their Pydantic
contracts.
