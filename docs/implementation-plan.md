# lumi-fit: model-serving architecture and implementation plan

Status: implemented architecture through Phase 4  
Last reviewed: 2026-08-10  
Working CLI name: `lumi-fit`

Current milestone: Phases 0 through 4 completed. Both planning directions,
workload-specific LUMI-vLLM fit analysis, versioned CLI output, offline
explanation, and the llmfit-style interactive terminal workflow are available.
Weighted objective scoring was superseded by ADR 0007. Curated task-quality and
SLO-goodput evidence is the next milestone.

## 1. Product definition

`lumi-fit` is an interface for deciding which large language models can be
self-hosted on LUMI and what allocation is required to serve them.

Unlike local-machine tools, the available hardware is not a fixed input that is
automatically detected. LUMI resources are requested from Slurm, so allocation
shape is part of the decision. The tool must search over model configuration and
resource configuration together.

The ranked object is therefore a complete `ServingPlan`:

```text
model artifact + precision/quantization + context + concurrency
+ serving runtime and container + batching limits
+ partition + nodes + GCDs + CPU memory/cores
+ tensor/pipeline parallelism + replicas
+ requested service duration
```

The initial product is specifically about inference serving. Training,
fine-tuning, and general HPC workload planning are outside the first release.

## 2. Primary user workflows

The same analysis engine supports two inverse workflows.

### 2.1 Resource-first: what can I serve?

The user provides an allocation envelope, for example:

```text
partition: small-g
nodes: 1
gcds: 4
cpu_cores: 32
cpu_memory: 256 GB
walltime: 8 hours
```

The tool searches the model catalog, artifact formats, quantizations, serving
runtimes, tensor-parallel sizes, batching configurations, and replica counts.
It returns runnable model-plan pairs ranked for the selected use case and
serving objective.

Questions answered:

- Which models fit this allocation?
- At what precision and maximum context?
- How many concurrent sequences can be supported safely?
- What latency and aggregate throughput are plausible?
- Is it better to shard one large model or run several replicas?
- Is the allocation underused or constrained by HBM, CPU memory, or topology?

### 2.2 Model-first: what resources do I need?

The user provides a model and serving requirements, for example:

```text
model: Qwen/Qwen3-32B
context: 32768
concurrent_sequences: 8
minimum_output_tokens_per_second: 120
service_duration: 12 hours
```

The tool searches LUMI partitions and allocation shapes. It returns three
distinct recommendations when possible:

- `minimum_runnable`: the smallest safe allocation;
- `recommended`: the best balance of performance, safety, and billing;
- `throughput_optimized`: the fastest plan before scaling efficiency becomes
  wasteful.

Questions answered:

- What is the minimum GCD count?
- Does the model fit on one GCD, one MI250X module, one node, or multiple nodes?
- Which tensor-parallel or replica layout should be used?
- Which partition is appropriate?
- What will the allocation consume in GPU-hours per hour and over the requested
  duration?
- Which runtime assumptions and parallel layout support the estimate?

### 2.3 Compare models for a serving target

The user supplies a shortlist or use-case filter plus serving requirements.
The tool compares the best feasible plan for each model. This keeps model
suitability separate from hardware fit: the cheapest model is not necessarily
the best model for coding, reasoning, multilingual work, or embeddings.

## 3. Explicit non-goals for version 0.1

- Training or fine-tuning resource planning.
- Any Slurm script generation, submission, cancellation, or queue management.
- vLLM/container launch commands or deployment instructions.
- Live LUMI, SSH, scheduler, allocation, hardware, or filesystem probes.
- Benchmark workload or job generation.
- A permanent public inference endpoint.
- Kubernetes deployment on LUMI-K.
- Model downloading or acceptance of gated-model licenses.
- A replacement for vLLM, Hugging Face, Slurm, or LUMI monitoring tools.
- Exact queue start-time prediction.
- Exact latency guarantees without comparable measured evidence.

The tool describes hypothetical serving configurations only. It never presents
a recommendation as a running or deployable service.

## 4. Design principles

### 4.1 Feasibility is a gate, not a score

A candidate first passes hard memory, software, topology, partition, and
resource constraints. A high quality or throughput score cannot compensate for
an out-of-memory or unsupported configuration.

### 4.2 Allocation is a search variable

The engine must not merely detect the current node. It must be able to simulate
one through many GCDs and compare `small-g` with full-node `standard-g` plans.

### 4.3 Per-GCD placement matters

Eight 64 GB memory domains are not equivalent to one flat 512 GB device. Model
weights, KV cache, activations, and communication buffers must fit after the
runtime's actual sharding and replication behavior is applied.

### 4.4 Safety and utilization are different

HBM headroom prevents runtime failures. Allocation and billing facts show the
cost of unused GCDs, CPU cores, or memory. Neither is blended into fit.

### 4.5 Serving capacity includes context and concurrency

“The weights fit” is insufficient. A useful serving plan must reserve memory
for KV cache at the requested context and concurrent-sequence capacity, plus
runtime workspaces and allocator fragmentation.

### 4.6 Estimates expose their basis

Every memory and performance result identifies its formula, inputs, runtime
assumptions, benchmark match, uncertainty range, and confidence class.

### 4.7 Better evidence replaces weaker estimates

Evidence precedence is:

1. curated or user-supplied exact evidence for the same configuration;
2. close published evidence accepted by an explicit matching rule;
3. calibrated analytical estimate;
4. uncalibrated analytical estimate;
5. heuristic fallback.

### 4.8 Offline and recommendation-only

Recommendation works without network or LUMI access. There are no live probes,
deployment artifacts, launch commands, or job operations. The final action is a
local explanation or saved recommendation bundle.

## 5. LUMI facts represented by the catalog

LUMI facts are versioned data with official source URLs and verification dates,
not constants spread through application code.

### 5.1 LUMI-G units and topology

- One LUMI-G node contains four AMD MI250X modules.
- Each MI250X module contains two Graphics Compute Dies.
- Slurm and HIP expose a full node as eight GPUs/GCDs.
- Each GCD has access to a documented 64 GB HBM slice.
- A node has 512 GiB CPU memory.
- Only 56 CPU cores are schedulable for jobs.
- Links between GCDs are non-uniform; the two GCDs in one MI250X module have
  stronger connectivity than arbitrary pairs across modules.
- Each MI250X module has a Slingshot network endpoint.

Source: [LUMI-G hardware documentation](https://docs.lumi-supercomputer.eu/hardware/lumig/).

Internal types distinguish:

- `gcd`: one Slurm/HIP device and HBM memory domain;
- `mi250x_module`: two GCDs and one billing GPU unit;
- `node`: four MI250X modules and eight GCDs.

The word `gpu` is not used as an unqualified internal unit.

### 5.2 Partitions

The initial catalog contains:

- `dev-g`: short validation and benchmark runs;
- `small-g`: sub-node and small multi-node serving jobs;
- `standard-g`: exclusive full-node serving jobs;
- `bench`: exceptional benchmark access, disabled unless explicitly enabled.

Node limits, walltime, purpose, allocation granularity, and access requirements
are profile fields because they may change.

Source: [LUMI Slurm partitions](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/).

### 5.3 Billing

For `standard-g`:

```text
billed_gpu_hours = 4 * nodes * service_hours
```

For sub-node GPU partitions such as `small-g` and `dev-g`:

```text
billed_gpu_hours = 0.5 * max(
    allocated_gcds,
    ceil(allocated_cpu_cores / 8),
    ceil(allocated_cpu_memory_gb / 64)
) * service_hours
```

For a persistent inference server, billing depends on allocation duration, not
request activity. The interface reports:

- GPU-hours per wall-clock hour;
- GPU-hours for requested duration;
- estimated utilization under the supplied traffic profile;
- cost per million generated tokens when traffic is known.

Source: [LUMI billing policy](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/).

### 5.4 Runtime and container facts

The LUMI AI Factory full container includes vLLM and a LUMI-compatible AI stack.
The first runtime adapter should target that documented path. Generic containers
may work on one GCD but must not receive high multi-node confidence unless their
MPI/RCCL/Slingshot compatibility is verified.

Sources: [LUMI AI containers](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/),
[container jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/),
and [Python/container guidance](https://docs.lumi-supercomputer.eu/software/installing/python/).

## 6. Interface design

The product exposes the same core through an interactive terminal interface,
automation-friendly CLI commands, and versioned JSON. A web interface is not
required for the first release.

### 6.1 Interactive TUI

Running `lumi-fit` opens directly into a full-screen TUI inspired by `llmfit`.
It is a persistent dashboard, not a sequential questionnaire:

```text
┌ lumi-fit ─ simulated LUMI-G allocation and workload ─────────────┐
│ Search │ Task │ Allocation │ Context │ Conc. │ Target │ Fit │ Sort│
├ Ranked model recommendations ─────────┬ Selected plan details ───┤
│ model · eligibility · quality · fit   │ HBM · layout · evidence │
└ keyboard actions and status ─────────────────────────────────────┘
```

The top bar provides focusable controls for search, task, allocation, context,
concurrency, target request rate, fit filtering, and sorting. Tab or mouse focus
selects a box. Left/right changes the value and recalculates immediately;
Enter/Space advances to the next value. No selection modal interrupts keyboard
navigation. An advanced overlay editor changes the complete partition,
resource, token, duration, quality, and SLO declaration.

#### Resource-first dashboard

The default table shows one recommendation or diagnostic per cataloged model.
Arrow or Vim keys navigate it; `/` searches; `f` filters; `s` sorts; and
`Enter` toggles the selected plan details. `m` and `c` compare two rows.

#### Model-first plan mode

Pressing `p` on a model replaces the table with minimum, recommended, and
throughput-oriented resource plans for that model. `b` returns to the model
table without losing the simulated allocation.

#### Results table

The common table contains:

```text
Model | Quant | Context | Concurrency | Runtime | Partition
Nodes/GCDs | TP x Replicas | HBM peak | Decode tok/s | GPU-h/h
Memory fit | Serveability | Estimated coverage | Quality | Validated fit
```

Selecting a row opens a plan explanation, technical gates, estimated coverage,
optional quality, validated fit, raw capacity, assumptions, and alternatives.

### 6.2 CLI workflows

```text
# What fits a planned allocation?
lumi-fit models --resources resources.yaml --serving serving.yaml

# What allocation is needed for one model?
lumi-fit plan Qwen/Qwen3-32B --serving serving.yaml

# Compare the best plan for selected models
lumi-fit compare models.txt --serving serving.yaml

# Stable automation output
lumi-fit plan Qwen/Qwen3-32B --serving serving.yaml \
  --output recommendation.json

# Consume the exact saved analysis without recalculating it
lumi-fit explain recommendation.json --plan <plan-id>

# Validation and catalog inspection
lumi-fit validate serving.yaml
lumi-fit catalog models search qwen
lumi-fit catalog validate
lumi-fit system show
lumi-fit doctor
```

### 6.3 Recommendation bundles

Every analysis can be saved as a self-contained JSON bundle. It includes:

- schema and tool versions;
- normalized inputs;
- model and LUMI catalog versions;
- every candidate or the configured retained subset;
- feasibility results and full estimate traces;
- scoring profile and weights;
- stable plan identifiers;
- analysis fingerprint and timestamp.

`plan_id` is derived from normalized plan content, not rank. `explain` consumes
the bundle so a catalog update cannot silently alter an already reviewed
recommendation.

## 7. Input contracts

### 7.1 Resource envelope

`ResourceEnvelope` defines allocated or allowed resources:

- partition or `auto`;
- fixed, minimum, and maximum node counts;
- fixed, minimum, and maximum GCD counts;
- GCDs per node when fixed;
- CPU cores and memory per node;
- walltime or serving duration;
- project-enabled partitions;
- maximum GPU-hours;
- whether partial-node or multi-node plans are allowed.

In resource-first mode, fixed fields are hard constraints. In model-first mode,
bounds define the search space.

### 7.2 Serving requirements

`ServingRequirements` contains:

- mode: `online` or `batch`;
- use case: general, chat, coding, reasoning, embedding, or custom;
- input context and maximum output tokens;
- maximum model context required;
- concurrent sequences;
- prompt/output-length distribution and minimum context coverage;
- target request rate or batch dataset size when known;
- latency targets:
  - time to first token;
  - inter-token latency;
  - end-to-end latency;
- throughput targets:
  - aggregate input tokens/s;
  - aggregate output tokens/s;
  - requests/s;
- required SLO attainment and maximum failure rate;
- minimum task-quality score;
- service duration for online mode;
- completion deadline for batch mode;
- minimum HBM headroom;
- allowed runtimes and quantizations;

Unknown traffic fields remain unknown. The tool must not assume a fully utilized
server when calculating real workload cost unless the user selects that
scenario explicitly.

### 7.3 Model selector

A selector can contain:

- an exact Hugging Face repository and revision;
- a local model metadata file;
- an allowlist of models;
- parameter, license, capability, language, or use-case filters;
- minimum external benchmark quality when available;
- include/exclude patterns.

## 8. Model and artifact catalog

The catalog distinguishes a conceptual model from serveable artifacts.

### 8.1 Model record

Fields include:

- canonical model identifier, family, revision, and release date;
- license and gated-access state;
- capabilities, languages, architecture, and intended uses;
- total and active parameter counts;
- hidden size, layers, attention heads, KV heads, head dimension, vocabulary;
- context limit and attention implementation;
- MoE expert metadata where applicable;
- external benchmark evidence with source and date.

### 8.2 Artifact record

Fields include:

- repository/revision and file format;
- weight dtype or quantization;
- actual shard sizes and total bytes when known;
- runtime compatibility;
- architecture-specific kernel requirements;
- expected load-time and storage metadata;
- source and last verification date.

Fit calculations use artifact bytes when available. Parameter-count estimates
are a fallback with lower confidence.

### 8.3 Catalog ingestion

The initial release uses a curated catalog of verified models plus an importer
for explicit Hugging Face model configuration. A later refresh script may query
the Hugging Face API and produce a reviewed snapshot.

Remote metadata is never required during normal recommendation. Gated status
is reported, and the tool never accepts licenses or stores access tokens.

### 8.4 Model suitability

Model suitability is separate from infrastructure fit. It uses sourced
task-specific benchmarks or user-provided evaluations. Family-name and raw
parameter-count heuristics must not masquerade as measured quality.

When no quality evidence exists, the UI displays `quality: unknown` and ranks
by serving fit unless the user supplies a model preference. Unknown quality is
not treated as zero quality.

## 9. Serving plan domain model

`ServingPlan` is immutable after validation and contains the complete
hypothetical configuration being scored:

- model and artifact identifiers;
- runtime and immutable container identity;
- weight and KV-cache precision;
- maximum model length;
- maximum concurrent sequences;
- maximum batched tokens;
- scheduler/batching policy;
- partition, nodes, GCDs per node, total GCDs;
- CPU cores and CPU memory per node;
- tensor-parallel and pipeline-parallel sizes;
- replica count and request-routing strategy;
- ranks, threads, GPU binding, and CPU binding;
- walltime/service duration;
- typed runtime assumptions required by memory and capacity estimation.

Core invariants include:

```text
gcds_per_replica = tensor_parallel * pipeline_parallel
total_used_gcds = replicas * gcds_per_replica
total_used_gcds <= allocated_gcds
tensor_parallel divides required attention/tensor dimensions
each replica receives the configured context and concurrency capacity
each rank maps to exactly one allocated GCD unless the runtime documents otherwise
```

Runtime options are inert validated data. Workload YAML cannot execute commands
or inject operational configuration.

## 10. Candidate-plan generation

Candidate generation is deterministic for identical inputs and catalog
versions.

### 10.1 Search dimensions

For every selected model/artifact, search:

- compatible serving runtime/container;
- allowed precision and quantization;
- partition;
- node and GCD count;
- tensor-parallel size;
- pipeline-parallel size if supported;
- replica count;
- context/concurrency capacity at or above the requested minimum;
- maximum batched tokens;
- optional CPU offload only when supported and explicitly allowed.

### 10.2 Resource-first search

The allocation is fixed. The search packs one or more serving replicas inside
it. It considers:

- one larger tensor-parallel replica;
- several smaller replicas;
- leaving a GCD unused only when topology or memory safety justifies it;
- the best artifact precision that satisfies requirements.

### 10.3 Model-first search

Enumerate from the smallest plausible allocation upward:

1. one GCD;
2. two GCDs within one MI250X module;
3. topology-compatible four-GCD layouts;
4. a full eight-GCD node;
5. multiple full nodes when the runtime and project bounds allow it.

Sub-node candidates prefer `small-g`; full-node and scale-out candidates compare
eligible partitions using their current limits and billing rules.

### 10.4 Early pruning

Prune when:

- partition or resource limits fail;
- the artifact/runtime is unsupported on ROCm/gfx90a;
- tensor dimensions are incompatible with tensor parallelism;
- a weight-only lower bound exceeds per-rank HBM;
- replica/parallelism products exceed allocated GCDs;
- context or concurrency is below the user's hard minimum;
- the candidate exceeds an explicit GPU-hour bound;
- multi-node communication is required but the container path is unverified.

Search caps are explicit. If a cap is reached, output says that the search was
incomplete and does not claim global optimality.

## 11. Hard feasibility engine

Every check returns a structured result containing identifier, severity,
actual value, required value, evidence, and remediation.

### 11.1 Feasibility status

- `feasible`: all hard constraints pass with known supported behavior;
- `conditional`: physical fit is plausible, but required metadata, artifact,
  access, or runtime validation is missing;
- `infeasible`: at least one hard constraint fails.

Only feasible plans receive normal fit bands. Conditional plans remain visible
but cannot outrank feasible plans.

### 11.2 Resource checks

- Conservative peak HBM fits on every rank/GCD.
- CPU memory includes model loading, tokenizer, runtime, file cache, and offload.
- CPU cores and memory obey node and partition limits.
- Requested GCD placement can support the parallelism layout.
- Walltime/service duration obeys partition limits.
- Full-node partitions are modeled and billed as exclusive nodes.

### 11.3 Software and artifact checks

- Artifact can be loaded by the runtime.
- Quantization has a supported ROCm kernel path.
- Runtime supports the architecture, context, tensor parallelism, and batching.
- Container is available or explicitly provided.
- Multi-node communication stack is LUMI-compatible.
- Gated artifacts are accessible, or the plan is conditional.

### 11.4 Fit bands

For feasible plans:

- `excellent`: recommended HBM headroom and proven software path;
- `good`: adequate headroom with a minor efficiency or evidence caveat;
- `marginal`: tight memory or a significant operational caveat.

These summarize plan memory/operational feasibility. They remain separate from
the workload-specific LUMI-vLLM fit score.

## 12. Serving memory model

The estimator produces a per-rank and per-replica breakdown with expected and
conservative values.

### 12.1 Components

```text
peak_hbm_per_rank =
    weight_shard
  + kv_cache_shard
  + peak_activations
  + runtime_workspace
  + communication_buffers
  + allocator_fragmentation_reserve
```

CPU memory includes artifact loading/staging, tokenizer/runtime state, host-side
cache, pinned communication buffers, and any intentional offload.

### 12.2 Weights

Prefer actual artifact file and tensor metadata. Fallback:

```text
weight_bytes = parameters * effective_bytes_per_parameter
weight_shard = runtime_sharding_rule(weight_bytes, parallelism, architecture)
```

Quantized formats include block metadata and scale overhead rather than assuming
exactly `bits / 8` bytes per parameter.

### 12.3 KV cache

When architecture metadata is available:

```text
kv_bytes =
    2
  * layers
  * kv_heads
  * head_dimension
  * reserved_tokens
  * kv_element_bytes
```

Where `reserved_tokens` reflects runtime capacity, not only one prompt:

```text
reserved_tokens >= concurrent_sequences * configured_sequence_capacity
```

Paged KV implementations may reserve blocks dynamically. Their allocator and
sharding behavior are runtime-versioned rules. The output distinguishes:

- requested working set;
- configured maximum capacity;
- runtime reservation;
- per-rank KV shard.

### 12.4 Activations and workspaces

Prefill and decode peaks are estimated separately. Flash-attention workspaces,
logit buffers, CUDA-named PyTorch APIs backed by ROCm, collective buffers, graph
capture, and runtime allocator reserves are included through compatibility and
runtime profiles.

### 12.5 Safety policy

The default recommendation target leaves at least 15% physical HBM headroom.
A candidate that physically fits but misses that reserve is `marginal` unless an
exact matching benchmark proves it stable.

Missing architecture/runtime data uses a conservative fallback and lowers
confidence. Missing decision-critical data may make the plan conditional rather
than inventing a precise result.

## 13. Serving performance model

Inference performance has distinct phases and metrics.

### 13.1 Online serving metrics

- time to first token (TTFT);
- inter-token latency (ITL);
- per-request decode tokens/s;
- aggregate input and output tokens/s;
- requests/s at the supplied input/output distribution;
- p50 and conservative latency range;
- saturation concurrency.

### 13.2 Batch serving metrics

- aggregate input/output tokens/s;
- estimated completion time;
- billed GPU-hours to completion;
- model load/startup time;
- effective utilization.

### 13.3 Analytical estimates

- Prefill uses a compute-oriented roofline adjusted for prompt length, batch,
  attention implementation, and achieved efficiency.
- Decode uses a memory-bandwidth-oriented roofline adjusted for weight bytes,
  KV traffic, batch, quantization kernel efficiency, and collectives.
- Multi-GCD estimates add communication cost based on tensor/pipeline
  parallelism and whether traffic is intra-module, intra-node, or cross-node.
- Multiple replicas increase aggregate capacity but do not improve one
  request's latency; the output keeps these effects separate.

The initial formulas return optimistic, expected, and conservative values.
They must not claim percentile SLAs without matching load-test data.

### 13.4 Traffic and utilization

Online cost efficiency depends on request arrival rate. When traffic is known,
estimate utilization and queueing conservatively. When it is absent, report
capacity at selected synthetic loads and GPU-hours per hour, but do not claim
real cost per request.

Version 0.1 may use a simple bounded queueing approximation. A discrete-event
simulator is deferred until real arrival and service-time traces justify it.

## 14. Billing and resource efficiency

Persistent online serving uses requested service duration. Offline batch mode
uses predicted completion time, capped by walltime.

Outputs include:

- allocation rate in GPU-hours per wall-clock hour;
- requested-duration GPU-hours;
- expected batch-completion GPU-hours;
- GPU-hours per million output tokens when workload volume is known;
- CPU- or memory-driven billing inflation on sub-node partitions;
- unused full-node GCD fraction;
- useful replicas and GCDs versus allocated resources.

Storage consumption is reported separately in TB-hours. Model startup and
weight staging time are reported but are not guessed when filesystem and cache
state are unknown.

## 15. Workload-specific LUMI-vLLM fit and ranking

ADR 0007 supersedes the earlier weighted component design. The scoring object
is one concrete model revision, artifact/precision, context, vLLM/ROCm runtime,
concurrency, and GCD layout under one declared workload distribution and SLO.

### 15.1 Technical serveability and optional quality

Technical serveability is the product of three locally evaluated gates:

1. exact vLLM/ROCm support;
2. weights, KV cache, and runtime memory fit independently on each 64 GiB GCD;
3. the declared prompt/output distribution meets minimum context coverage.

Task quality remains separate. With no requested minimum it is `NOT REQUIRED`.
An explicit threshold produces `PASS`, `FAIL`, or `UNKNOWN` from comparable
cataloged evidence. It does not change whether the deployment can technically
run, but a failed or unresolved explicit requirement prevents validated fit.

### 15.2 Estimated capacity and SLO-constrained goodput

Before benchmark evidence exists, the analytical lower request-capacity prior
is normalized to an eight-GCD node and divided by target request rate. This
`ESTIMATED_CAPACITY_COVERAGE` is labeled `heuristic` and is used for initial
planning; it does not establish latency-SLO attainment.

A good request succeeds and meets all declared TTFT, TPOT, and optional E2E
limits. The scored capacity is the maximum request goodput at which attainment
meets the requested fraction and failure rate remains below its limit. Only an
exact measured or calibrated observation and its 95% lower confidence bound
can create an absolute fit score. Analytical performance estimates are exposed
as raw capacity only.

For a deployment consuming `k` GCDs per replica:

```text
equivalent_replicas_per_node = floor(8 / k)
full_node_goodput_lcb95 = equivalent_replicas_per_node * deployment_goodput_lcb95
VALIDATED_LUMI_VLLM_FIT = 100 * min(1, full_node_goodput_lcb95 / target_request_rate)
```

### 15.3 Ranking and explanation

The public ranker orders feasible before conditional before infeasible, then
technically eligible before unknown before ineligible, followed by optional
quality status, validated fit, estimated coverage, raw capacity, billing, and a
stable plan ID. Quality and fit are never collapsed into an arbitrary weighted
average.

Every result exposes technical gates, optional quality status, target demand,
estimated coverage, deployment and full-node goodput LCB, equivalent replicas,
raw analytical capacity, confidence, formula IDs, evidence IDs, and concise
status explanations.
The complete design is in `docs/scoring.md`.

## 16. Output plans

### 16.1 Resource-first output

Return the best plan for each model plus optional alternate configurations.
The default view avoids allowing one model's many nearly identical
quantizations to crowd every other model out of the table.

### 16.2 Model-first output tiers

`minimum_runnable` minimizes billed allocation rate while satisfying hard
context/concurrency and memory safety requirements.

`recommended` follows the workload-specific eligibility and fit rank and must
explain why its resources improve the outcome.

`throughput_optimized` adds resources only while marginal throughput gain and
scaling efficiency remain above configurable thresholds.

### 16.3 Capacity envelope

Each plan reports a small capacity curve rather than one unexplained point:

```text
context | concurrent sequences | expected HBM | expected aggregate tok/s
```

At minimum include the requested point, maximum safe concurrency at requested
context, and maximum safe context at requested concurrency.

## 17. Software architecture

The implementation uses a functional recommendation core with local file and
terminal adapters.

```text
resource/model request              model + LUMI catalogs
          |                                  |
          +----------- normalization --------+
                                      |
                           candidate-plan generator
                                      |
                            hard feasibility engine
                                      |
                 +--------------------+-------------------+
                 |                    |                   |
             memory model      performance model   billing/topology
                 |                    |                   |
                 +---- eligibility + SLO-goodput fit ------+
                                      |
                           ranker + explanation trace
                                      |
                          TUI / table / JSON
```

### 17.1 Layers

- `domain`: immutable, unit-safe request, catalog, plan, evidence, and result
  types;
- `catalog`: versioned LUMI, model, artifact, runtime, and legacy profiles;
- `planning`: resource-first and model-first candidate generation/pruning;
- `analysis`: feasibility, memory, performance, topology, and billing;
- `scoring`: eligibility gates, full-node goodput fit, ranking, and result tiers;
- `explain`: presentation-independent trace and remediation messages;
- `evidence`: inert observation records, matching, and optional calibration;
- `adapters`: TUI, CLI, local input files, and local recommendation exports.

The core performs no shell, network, SSH, or Slurm I/O.

### 17.2 Repository layout

```text
lumi-fit/
├── README.md
├── pyproject.toml
├── docs/
│   ├── implementation-plan.md
│   ├── scoring.md
│   ├── schemas.md
│   ├── benchmark-protocol.md
│   └── adr/
├── src/lumi_fit/
│   ├── cli.py
│   ├── tui.py
│   ├── domain/
│   │   ├── units.py
│   │   ├── requests.py
│   │   ├── models.py
│   │   ├── plans.py
│   │   ├── evidence.py
│   │   └── results.py
│   ├── catalog/
│   │   ├── loader.py
│   │   ├── schemas.py
│   │   └── validation.py
│   ├── planning/
│   │   ├── resource_first.py
│   │   ├── model_first.py
│   │   └── pruning.py
│   ├── analysis/
│   │   ├── feasibility.py
│   │   ├── memory.py
│   │   ├── performance.py
│   │   ├── topology.py
│   │   └── billing.py
│   ├── scoring/
│   │   ├── components.py
│   │   ├── profiles.py
│   │   └── ranking.py
│   ├── explain/
│   │   ├── trace.py
│   │   └── report.py
│   ├── evidence/
│   │   ├── fingerprint.py
│   │   ├── records.py
│   │   ├── matching.py
│   │   └── calibration.py
│   └── adapters/
│       ├── config.py
│       ├── hf_import.py
│       └── output.py
├── data/
│   ├── systems/lumi-g.yaml
│   ├── partitions/lumi.yaml
│   ├── billing/lumi.yaml
│   ├── runtimes/vllm-lumi.yaml
│   ├── models/
│   ├── scoring/defaults.yaml
│   └── schemas/
├── examples/
│   ├── resource-first.yaml
│   ├── online-serving.yaml
│   └── batch-serving.yaml
└── tests/
    ├── unit/
    ├── integration/
    ├── golden/
    └── fixtures/
```

### 17.3 Technology

Use Python 3.11+ initially, with:

- Pydantic v2 for domain/config validation;
- standard-library `argparse` for a dependency-light automation CLI;
- Prompt Toolkit for the full-screen, keyboard-driven terminal dashboard;
- PyYAML for catalogs and requests;
- pytest and Hypothesis for tests.

The TUI is only a presentation adapter. It owns no scoring or feasibility
logic, and can be replaced without changing planning or bundle contracts.

The estimator workload is small, so a compiled language is unnecessary in the
first implementation. Versioned JSON remains the interface boundary for future
web, API, or alternative-language clients.

Memory is canonicalized to integer bytes internally. Inputs and output retain
explicit GB/GiB suffixes. Rates distinguish GB/s from GiB/s, time is seconds,
and billing is decimal GPU-hours.

## 18. Curated evidence and calibration

### 18.1 Fingerprint

A serving-evidence fingerprint includes:

- LUMI profile and allocation topology;
- container digest/path and runtime version;
- ROCm, PyTorch, RCCL, MPI/libfabric versions;
- model/artifact revision;
- quantization and KV dtype;
- context, input/output distribution, concurrency, and batching limits;
- tensor/pipeline parallelism and replicas;
- rank/GCD and CPU binding;
- important environment variables;
- storage/cache state.

Matching rules are explicit. A batch-one decode result cannot silently
calibrate high-concurrency throughput.

### 18.2 Metrics

- load/startup time;
- peak HBM per rank and CPU memory;
- TTFT, ITL, and end-to-end latency;
- prefill and decode tokens/s;
- aggregate throughput and requests/s;
- error and OOM status;
- exact input/output token distribution;
- allocation duration and billed GPU-hours;
- optional collective timing for sharded serving.

### 18.3 Records and calibration

Use append-only schema-validated JSON records from curated publications or
user-supplied inert data. Raw observations remain separate from derived
calibration factors. The tool does not generate benchmark workloads or commands.

Start with robust corrections in strict fingerprint groups:

```text
factor = median(measured / estimated)
calibrated = analytical * factor
```

Retain sample count and dispersion, require a minimum sample count, reject
invalid runs, and bound extreme corrections. More complex models must beat this
baseline on held-out results before adoption.

## 19. Recommendation-only boundary

The application stops at ranked local decision support. It has no renderer,
launcher, live adapter, scheduler integration, SSH integration, or job monitor.
It does not propose Slurm/vLLM commands as a secondary output.

`ServingPlan` is a scored hypothetical configuration, not an executable
deployment specification. Runtime/container data is retained only because
software compatibility and tensor-parallel behavior affect whether a model fits.

## 20. Catalog maintenance

Catalog updates are reviewed, versioned repository changes. Normal
recommendation never fetches remote state. A maintenance importer may transform
explicit upstream metadata into a candidate catalog file, but it cannot replace
the active snapshot automatically.

Catalog drift is handled through verification dates, validity windows, source
URLs, and new reviewed releases—not live LUMI probing.

## 21. Testing strategy

### 21.1 Unit tests

- GCD/module/node and GB/GiB conversions;
- billing at every GCD, CPU, memory, and rounding boundary;
- artifact weight and quantization overhead;
- KV-cache formulas and runtime sharding rules;
- replica/tensor/pipeline invariants;
- per-rank HBM feasibility;
- partition limits and exclusive-node semantics;
- eligibility-gate behavior with nullable quality;
- exact goodput-evidence matching and 95% LCB fit calculations;
- deterministic plan identifiers and ranking.

### 21.2 Property-based tests

- increasing context or concurrency cannot reduce required KV memory;
- increasing conservative HBM cannot improve memory safety;
- a failed gate never produces a numeric fit score;
- unmatched or heuristic evidence never produces a numeric fit score;
- increasing an eligible full-node goodput LCB cannot reduce fit;
- infeasible plans never outrank feasible plans;
- more replicas cannot improve single-request latency unless the performance
  model explicitly changes request scheduling;
- serialization round trips preserve analysis fingerprints.

### 21.3 Golden scenarios

- small model on one `small-g` GCD;
- model fitting only after supported quantization;
- model requiring two GCDs within one MI250X module;
- full-node tensor-parallel model;
- two smaller replicas outperforming one sharded replica for batch throughput;
- model weights fit but requested KV capacity does not;
- CPU-memory-driven sub-node billing inflation;
- full-node plan with unused billed GCDs;
- unsupported CUDA-only quantization kernel;
- conditional gated artifact;
- multi-node plan with unverified container networking;
- online server with low traffic and poor utilization;
- unknown model quality with otherwise excellent infrastructure fit.

### 21.4 Empirical acceptance metrics

- false-feasible rate: predicted-to-fit plans that OOM or fail compatibility;
- conservative HBM error;
- TTFT, ITL, and throughput error by model/runtime class;
- billing calculation error;
- recommendation stability across catalog refreshes;
- held-out calibration error.

The first maintained benchmark suite targets zero false-feasible plans.

## 22. Security and operational boundaries

- Normal recommendation performs no network access.
- Workload files cannot execute commands.
- Remote metadata and evidence imports are untrusted inert input.
- No scripts, commands, credentials, or environment captures are generated.
- Model licenses and gated access remain the user's responsibility.
- The tool performs no remote mutation or LUMI operation.

## 23. Implementation phases

### Phase 0: contracts and catalogs

Deliver:

- Python package skeleton and quality tooling;
- domain units and schemas;
- initial LUMI-G, partition, and billing profiles;
- serving, resource-envelope, model, artifact, plan, and result schemas;
- initial vLLM/LUMI container compatibility record;
- representative resource-first and model-first examples;
- architecture decision records.

Exit criteria:

- all catalogs/examples validate;
- terminology distinguishes GCD, module, and node;
- billing boundary tests pass;
- recommendation bundle schema is reviewed before estimator work.

### Phase 1: model-first memory planning

Deliver:

- curated initial model/artifact catalog;
- artifact importer;
- model-first candidate generator;
- per-rank weight, KV, activation, workspace, and reserve estimates;
- hard feasibility engine;
- minimum-runnable and safe-resource tiers.

Exit criteria:

- representative models receive explainable resource plans;
- weights fitting without KV capacity is detected correctly;
- no flat aggregate-HBM shortcut is used;
- missing critical metadata produces conditional output.

### Phase 2: resource-first model discovery

Status: complete (initial curated-catalog implementation).

Deliver:

- allocation-envelope input;
- cross-model candidate generation;
- quantization/runtime filtering;
- replica-versus-sharding alternatives;
- one-best-plan-per-model result grouping;
- model suitability evidence contract.

Exit criteria:

- users can list models fitting one, two, four, and eight GCD envelopes;
- alternatives explain context/concurrency tradeoffs;
- unknown quality is represented honestly.

### Phase 3: billing, scoring, CLI, and TUI

Status: complete (analytical-prior implementation; calibration is Phase 5).

Deliver:

- exact billing engine;
- workload-specific eligibility and fit contracts;
- deterministic ranker and plan tiers;
- CLI commands and JSON bundles;
- interactive resource-first/model-first TUI;
- complete explanation view.

Exit criteria:

- all ranking properties pass;
- every numeric fit exposes its exact workload and evidence;
- TUI and CLI consume the same core results;
- saved results can be explained without recomputation.

### Phase 4: recommendation quality and catalog coverage

Status: complete.
See `docs/phase4-recommendation-quality.md`.

Deliver:

- broader immutable model/artifact coverage;
- model/license/language/parameter filters;
- context/concurrency capacity envelopes;
- separate quality, eligibility, fit, and raw-capacity presentation;
- local sorting, detail views, and CSV export.

Exit criteria:

- one-, two-, four-, and eight-GCD scenarios have useful catalog coverage;
- capacity boundaries and ranks are explainable and deterministic;
- unknown or incomparable quality is never fabricated;
- normal operation remains fully local and offline;
- no execution, Slurm rendering, launch, or live-probe path exists.

### Phase 5: curated evidence and estimator refinement

Deliver:

- inert published/user-supplied evidence records;
- strict exact/near/unmatched evidence rules;
- phase-aware TTFT/ITL/throughput estimator refinements;
- topology/collective priors for hypothetical multi-node recommendations;
- published error against held-out curated observations.

Exit criteria:

- evidence provenance and comparability are explicit;
- unmatched evidence cannot alter scores;
- estimator error is reported by model/runtime/workload class;
- all work remains local recommendation analysis.

## 24. Initial architecture decisions

Record these as ADRs in Phase 0:

1. The initial product plans inference serving only.
2. Support both resource-first and model-first searches.
3. Rank complete serving plans, not model names.
4. Apply feasibility gates before scoring.
5. Treat context and concurrency as required memory dimensions.
6. Keep memory safety separate from allocation and billing efficiency.
7. Treat GCD, MI250X module, and node as distinct units.
8. Use versioned data catalogs and pure analysis functions.
9. Use Python 3.11+, with a shared TUI/CLI core.
10. Use the LUMI AI Factory vLLM container path as the first supported runtime.
11. Prefer comparable curated evidence over analytical estimates.
12. Produce local recommendations only; never render or operate deployments.

## 25. Risks and mitigations

### Runtime support changes

Quantization and model-architecture support can change with vLLM, ROCm, and
container releases. Compatibility is versioned data and exact container/runtime
versions are part of every fingerprint.

### Misleading aggregate memory

The planner may falsely claim that a model fits by summing HBM across devices.
All estimates are per-rank after concrete sharding rules, with per-component
breakdowns.

### KV-capacity surprises

Weights may load successfully while production concurrency OOMs. Context,
concurrent sequences, batched tokens, KV dtype, and allocator reserve are
first-class inputs and reported separately.

### Unsupported quantization kernels

An artifact may be small enough but unusable on ROCm. Artifact/runtime/kernel
compatibility is a hard gate, not inferred from file size.

### Performance overconfidence

Analytical rooflines can misrepresent batching and collective behavior. Results
are ranges with confidence classes; exact benchmarks override them.

### Idle persistent allocations

An online server may be feasible but economically poor at low traffic. The UI
shows GPU-hours per hour and refuses to invent utilization when arrival rate is
unknown.

### LUMI policy drift

Partitions, containers, and billing can change. Profiles contain validity dates,
official sources, and optional live drift detection.

### Candidate explosion

Models, artifacts, quantizations, parallelism, replicas, and resources form a
large search. Lower-bound pruning, topology-aware enumeration, one-best-plan
grouping, and explicit caps keep the search tractable without hiding truncation.

## 26. Definition of the first useful release

Version `0.1` is useful when both of these work offline:

```text
lumi-fit models --resources resources.yaml --serving serving.yaml
lumi-fit plan <model> --serving serving.yaml
```

The result must include:

- correct LUMI partition and billing semantics;
- per-GCD conservative HBM breakdown including KV capacity;
- supported artifact, precision, runtime, and parallelism checks;
- minimum, recommended, and throughput-oriented resource plans;
- context/concurrency capacity envelope;
- expected serving metrics with uncertainty and confidence;
- allocation rate and requested-duration GPU-hours;
- explainable eligibility, task quality, and workload-specific fit;
- clear conditional and infeasible reasons;
- a saved versioned recommendation bundle;
- no generated operational artifacts or commands.

The release is not ready if it only answers whether model weights fit. It must
plan useful serving capacity and the LUMI resources required to provide it.

## 27. Immediate next steps

1. Define inert observation/evidence contracts and strict matching rules.
2. Add another catalog increment only when its runtime class has explicit LUMI
   compatibility evidence.
3. Calibrate analytical priors only from comparable imported evidence.

## 28. Reference material

- [`llmfit` repository](https://github.com/AlexsJones/llmfit)
- [`llmfit` scoring overview](https://github.com/AlexsJones/llmfit/blob/main/docs/how-it-works.md)
- [`llmfit` scoring implementation](https://github.com/AlexsJones/llmfit/blob/main/llmfit-core/src/fit.rs)
- [`llmfit` benchmark design](https://github.com/AlexsJones/llmfit/blob/main/docs/cli.md)
- [LUMI AI inference overview](https://docs.lumi-supercomputer.eu/laif/inference/)
- [LUMI-G hardware](https://docs.lumi-supercomputer.eu/hardware/lumig/)
- [LUMI Slurm partitions](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/)
- [LUMI billing policy](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/)
- [LUMI-G job examples](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/)
- [LUMI AI container environment](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/)
- [LUMI container jobs](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs/)

`llmfit` is MIT-licensed. Concepts can be reimplemented independently; any
future source-code reuse must preserve the applicable license and attribution.
