# Phase 2 resource-first model discovery

Phase 2 answers: given a concrete LUMI allocation, which cataloged models can
serve this workload, and what is the safest layout for each model?

The allocation is not resized during this search. `nodes` and `gcds` must both
be fixed counts in `ResourceEnvelope`; ranges remain the input to model-first
planning. Every returned plan therefore carries the same allocated node and GCD
count, even when a candidate leaves some GCDs unused.

## Inputs and output

The planner consumes the same contracts as model-first planning:

- `ResourceEnvelope`: fixed partition/allocation, CPU/RAM, walltime, and budget;
- `ServingRequirements`: context, concurrency, mode, duration, runtime, and
  quantization policy;
- `ModelSelector`: explicit models, glob filters, task, exclusions, and gating;
- versioned hardware, partition, billing, runtime, model, and scoring catalogs.

It emits the common `RecommendationBundle`. `candidates` contains every
examined serving plan and its constraint evidence. `model_groups` adds one
cross-model discovery record per model:

- `best_plan_id`: highest-ranked feasible plan, or null;
- `diagnostic_plan_id`: closest conditional/infeasible plan when none runs;
- `alternative_plan_ids`: other feasible layouts in rank order;
- `feasible_plan_count`: transparent search coverage;
- `suitability`: separate task-quality score, confidence, and evidence.

Plan IDs and the analysis fingerprint are deterministic; `generated_at` is not
part of the fingerprint.

## Model and artifact filtering

Selection is deterministic:

1. Start with explicit IDs and the union of include-glob matches. If neither is
   supplied, start with all cataloged models.
2. Reject gated models unless `allow_gated` is true.
3. Require the capability mapped from `ModelSelector.use_case`, when supplied.
4. Apply exclusion globs last.
5. Keep only artifacts allowed by the requested runtime and quantization policy.

An unknown explicit ID is an input error. A selected model with no compatible
artifact remains grouped with no best or diagnostic plan; the initial curated
catalog has a compatible BF16 artifact for every model.

## Fixed-allocation validation

The initial implementation requires a uniform GCD count per node:

```text
gcds_per_node = fixed_total_gcds / fixed_nodes
```

The division must be exact and cannot exceed eight GCDs per LUMI-G node.
`standard-g` requires a complete eight-GCD node. Partial allocations use
`small-g` and require `allow_partial_node`. Partition, node, walltime, CPU,
memory, billing, and GPU-hour constraints are checked for every plan.

## Candidate enumeration

For every selected model/artifact/runtime/partition combination, enumerate:

```text
tensor_parallel in powers_of_two(1 .. allocated_gcds)
replicas        in 1 .. floor(allocated_gcds / tensor_parallel)
pipeline_parallel = 1
```

Layouts whose used rank count cannot be distributed uniformly over the
allocated nodes are skipped. A plan may use fewer GCDs than allocated, but its
billing analysis always uses the full fixed allocation.

Global requested concurrency is divided over replicas:

```text
per_replica_concurrency = ceil(requested_concurrency / replicas)
```

Memory analysis then reserves KV cache and activation capacity using that
per-replica concurrency. Multiple replicas use explicit round-robin request
routing; one replica uses single routing. This makes replica packing materially
different from merely multiplying an unchanged one-replica estimate.

## Feasibility and ranking

Each candidate runs through the shared hard-constraint and per-rank HBM engine.
Ranking is lexicographic and intentionally does not invent performance data:

1. feasible before conditional before infeasible;
2. excellent before good before marginal memory fit;
3. fewer unused allocated GCDs;
4. more replicas;
5. lower conservative per-rank HBM use;
6. stable plan ID as the final tie-breaker.

Only a feasible candidate can become `best-for-model`. Conditional and
infeasible candidates are retained as explanations. A safe TP=4 plan can thus
outrank TP=2 with two replicas when the latter physically fits but violates the
requested HBM reserve.

The analytical latency/capacity estimates are raw priors, not measured or
calibrated performance claims. They produce explicitly heuristic target
coverage for initial planning. Validated workload fit requires matching LUMI
SLO-goodput evidence; task quality is required only when the request declares a
minimum.

## Model suitability

Infrastructure fit and task quality are separate axes. Capability metadata
allows filtering but does not prove relative quality. A suitability score is
therefore null with `unknown` confidence when no comparable evaluation record
exists. It is never coerced to zero.

`ModelEvaluation` records can declare an evaluation ID, use case, raw score,
valid range, source, date, and notes. When matching evidence exists, the planner
normalizes it to 0–100 and includes the evidence ID. Future scoring must compare
only evaluations whose task and methodology are genuinely comparable.

## Current boundaries

- The Phase 4A catalog has seven revision-pinned BF16 artifacts across compact,
  small, medium, large, dense, and MoE model classes.
- Pipeline-parallel and heterogeneous per-node layouts are not searched.
- Replica routing is a plan contract, not yet a generated proxy configuration.
- Latency and throughput are analytical priors; queueing and calibration are absent.
- Multi-node runtime support remains conditional pending an exact benchmark.
- The inspector remains a development surface; the production CLI/TUI is available.
