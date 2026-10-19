# Phase 4: recommendation quality and catalog coverage

Status: complete  
Scope: local, read-only recommendation and explanation

> Note: the Phase 4C weighted composite and objective-sensitivity design below
> is retained as implementation history. ADR 0007 supersedes it with separate
> quality and workload-specific LUMI-vLLM fit.

## 1. Product boundary

`lumi-fit` follows the `llmfit` interaction model: it evaluates cataloged models
against a resource/workload description and returns scores, recommendations,
alternatives, and explanations on the user's local machine.

It does not:

- connect to LUMI;
- inspect a live allocation or Slurm queue;
- generate or propose Slurm scripts;
- launch vLLM or another serving engine;
- download model weights;
- submit, monitor, cancel, or modify jobs;
- generate deployment or benchmark commands.

LUMI, Slurm, and runtime facts are static, versioned inputs to the estimator.
They describe the environment being modeled; they are not operational adapters.

## 2. Phase 4 outcome

Phase 4 makes the existing recommendation engine useful across a broader model
set and more workload shapes. The user should be able to enter a hypothetical
allocation or select a model and receive a stable local answer with:

- feasibility and fit band;
- one recommended configuration per model;
- minimum, balanced, and throughput-oriented resource alternatives;
- memory, capacity, topology, and billing evidence;
- separate task quality and workload-specific LUMI-vLLM fit;
- uncertainty and missing-evidence warnings;
- a concise comparison explaining why one option ranks above another.

No output is intended to be executable.

## 3. Workstream A: model and artifact coverage

Expand the curated catalog from two models to a representative initial matrix:

- compact: approximately 1B–4B parameters;
- general/coding: approximately 7B–14B;
- medium: approximately 20B–40B;
- large and MoE: models requiring a full node or multiple nodes;
- embedding models as a separate capability/runtime class in a later catalog
  increment, after that runtime class has explicit LUMI compatibility evidence.

Selection criteria:

- useful and recognizable model families;
- immutable revision;
- source license and gated status;
- architecture metadata sufficient for per-rank memory analysis;
- exact artifact size when available;
- explicit runtime/ROCm compatibility evidence;
- one or more task capabilities without inventing quality scores.

Each catalog addition requires provenance and a fixture-based validation test.
The normal recommendation path remains offline. Metadata refresh tooling may
produce a candidate snapshot, but it never changes the shipped catalog without
review.

## 4. Workstream B: precision and runtime variants

Represent multiple artifacts only when the LUMI runtime catalog supports them:

- BF16/FP16 baselines;
- FP8, AWQ, GPTQ, or other quantization only with gfx90a/runtime evidence;
- artifact-specific bytes and overhead;
- kernel requirements and support level;
- precision-related suitability caveats when evidence exists.

An artifact that is smaller but unsupported must remain infeasible. File size
alone never grants a fit recommendation.

## 5. Workstream C: capacity envelopes

Each recommended plan should expose more than the requested point:

```text
requested context at requested concurrency
maximum safe concurrency at requested context
maximum safe context at requested concurrency
one lower-cost neighboring allocation, when feasible
one higher-capacity neighboring allocation, when feasible
```

Capacity search reuses the existing componentized memory model. It uses bounded
integer/binary search and returns the evaluated points, not an unexplained
closed-form maximum.

The UI must distinguish:

- memory-safe capacity;
- analytical performance capacity;
- requested workload;
- catalog/model context limit.

## 6. Workstream D: comparison explanations

Add a presentation-independent comparison trace for two plans:

```text
winner and runner-up
feasibility difference
target-satisfaction difference
score-component deltas
HBM-headroom delta
estimated capacity delta
GPU-hour delta
topology/runtime-confidence delta
decisive tie-breaker
```

For resource-first results, the default view remains one best plan per model.
Users can expand a model to see its alternative tensor-parallel/replica layouts.

When task quality is unknown for both models, the explanation must explicitly
say that the ranking compares infrastructure fit rather than model intelligence.

## 7. Workstream E: local interface refinement

The terminal interface adds:

- sortable columns for score, parameters, HBM, throughput prior, and GPU-hours;
- filters for capability, model family, license, gating, parameter range,
  precision, partition, and fit band;
- compact and detailed views;
- model search with stable selection;
- side-by-side plan comparison;
- clear badges for `estimated`, `conditional`, and `quality unknown`;
- deterministic JSON and optional CSV recommendation export.

The interface never shows actions such as deploy, submit, start, probe, or
benchmark. The final action is save/export recommendation.

## 8. Workstream F: evidence without execution

The local engine may consume curated or user-supplied observation files, but it
does not generate the workload that produced them.

Accepted evidence must declare:

- source and date;
- model/artifact revision;
- runtime and accelerator identity;
- allocation and parallelism;
- context, input/output tokens, concurrency, and batching;
- measured metric, units, sample count, and dispersion.

Evidence matching is exact or explicitly near. Unmatched evidence is retained
for inspection but cannot change a recommendation. Imported evidence is read as
data and cannot contain commands, paths to execute, or environment capture.

## 9. Ranking refinements

Maintain the public ordering:

1. feasible before conditional before infeasible;
2. serving targets satisfied before missed;
3. objective-aware composite descending;
4. confidence descending;
5. billed GPU-hours ascending;
6. stable plan ID.

Phase 4 adds:

- score sensitivity showing how a result changes across objective profiles;
- rank stability when an unavailable component is removed;
- explicit decisive-component explanations;
- no cross-model quality ordering when suitability evidence is incomparable;
- separate `runtime_compatibility` terminology instead of deployment readiness.

## 10. Local CLI surface

The recommendation-only command set remains:

```bash
lumi-fit                         # full-screen recommendation dashboard
lumi-fit models --resources ... --serving ...
lumi-fit plan MODEL --serving ... [--resources ...]
lumi-fit compare models.txt --resources ... --serving ...
lumi-fit explain recommendation.json --plan PLAN_ID
lumi-fit validate INPUT
lumi-fit catalog validate
lumi-fit catalog models search QUERY
lumi-fit catalog coverage [--json]
lumi-fit system show
lumi-fit doctor
```

There are no render, deploy, launch, probe, submit, or benchmark-plan commands.

## 11. Implementation sequence

### 4A. Catalog expansion foundation

Completed on 2026-08-09:

1. Added executable, versioned coverage thresholds for compact, small, medium,
   and large parameter bands.
2. Added filters for family, parameter range, license, language, and capability.
3. Expanded the catalog to seven immutable BF16 artifacts across five families:
   Phi 3, Mistral, Qwen 2, Qwen 3 dense, and Qwen 3 MoE.
4. Added offline coverage/provenance reporting and made unmet thresholds part of
   whole-catalog validation.
5. Added recommendation scenarios for fixed one-, two-, four-, and eight-GCD
   allocations.

The initial matrix deliberately excludes gated models, unsupported quantized
artifacts, and embedding models. This avoids treating upstream availability as
evidence of compatibility with the curated LUMI runtime.

#### 4A-2. Catalog expansion, second increment

Completed on 2026-09-07:

1. Added four ungated BF16 artifacts covering four families absent from the
   initial matrix: SmolLM2 (llama), Granite 3.1 (granite), OLMo 2 (olmo2), and
   Mixtral 8x22B (mixtral) — bringing the catalog to eleven models across nine
   families.
2. Grew every parameter band beyond its bare minimum: compact 1→2, small 2→3,
   medium 3→4, large 1→2, so each band offers a real alternative rather than a
   single model.
3. Every addition still follows the same provenance discipline: an immutable
   40-character revision SHA, an HTTPS Hugging Face source URL pinned to that
   revision, and an exact artifact byte size (from `model.safetensors.index.json`
   metadata, or the single-file `model.safetensors` listing when a model has no
   shard index). Mixtral 8x22B's active-parameter count is cited from Mistral
   AI's own announcement, not estimated, since Hugging Face config metadata has
   no `active_parameters` field.
4. Gated models remain excluded, consistent with 4A's original scope decision.

### 4B. Capacity-envelope engine

Completed:

1. Added capacity-probe, envelope, and allocation-neighbor contracts.
2. Added bounded integer searches for safe context and concurrency using the
   existing componentized HBM estimator.
3. Attached envelopes without changing plan IDs or recommendation ranking.
4. Persisted evaluated points and explicitly separated memory safety from
   unknown sustainable performance capacity.
5. Added neighboring-plan references, monotonicity tests, exact boundary tests,
   JSON serialization checks, and one/two/four/eight-GCD coverage.

### 4C. Comparison and score sensitivity

Superseded by ADR 0007. The following records the work completed at the time;
current planners emit the compatibility fields empty.

Completed:

1. Added persisted pairwise comparison and score-component delta contracts.
2. Re-scored the identical feasible candidate set across all five objectives.
3. Added winner changes, rank movement, per-profile weights, and contributions.
4. Made shared evidence IDs a prerequisite for cross-model quality scoring.
5. Added decisive explanations following the public ranking gates exactly.
6. Added rank-set, unavailable-component, winner-change, evidence-comparability,
   ranking-gate, CLI, and serialization tests.

### 4D. Interface polish

Completed:

1. Added direct model filters for patterns, family, license, language,
   capability, gating, and parameter range.
2. Added result filters for status, fit band, partition, and precision.
3. Added deterministic sorting by model, parameters, score, HBM, throughput,
   GPU-hours, and memory-safe concurrency.
4. Added compact, detail, compare, and expanded-alternative views.
5. Added stable CSV stdout/file export while retaining complete JSON bundles.
6. Added golden CSV scenarios for one-, two-, four-, and eight-GCD allocations.

## 12. Exit criteria

- the curated catalog covers representative compact through full-node models;
- every artifact has immutable provenance and explicit compatibility evidence;
- one-, two-, four-, and eight-GCD scenarios produce useful ranked results;
- selected plans include capacity envelopes with tested boundaries;
- comparisons explain the decisive rank differences;
- changing objectives produces deterministic, explainable changes;
- unknown or incomparable quality never becomes a fabricated numeric ordering;
- all normal operations work without network or LUMI access;
- the codebase contains no LUMI execution, Slurm rendering, launch, or probe path.
