# Phase 1 serving-memory model

Phase 1 answers the model-first question: given a model, context, concurrency,
runtime policy, and resource bounds, which LUMI allocations are physically and
operationally plausible?

It does not yet predict throughput or latency. Until performance calibration is
implemented, `recommended` means the least expensive feasible plan with good or
excellent memory headroom—not the fastest plan.

## Per-rank memory breakdown

The estimator evaluates one GCD/rank after applying tensor parallelism:

```text
peak HBM per rank =
    weight shard
  + KV-cache shard
  + activation envelope
  + runtime workspace
  + tensor-parallel communication buffers
  + allocator-fragmentation reserve
```

Each component contains lower, expected, and conservative byte values, a
formula identifier, sharding rule, and confidence class.

## Weights

Actual safetensors index size is preferred:

```text
ideal shard = artifact bytes / tensor parallel size
expected    = ideal shard * 1.01
upper       = ideal shard * 1.03
```

The allowance represents uneven placement and metadata until exact per-tensor
placement is implemented. Parameter count and dtype are used only when artifact
bytes are unavailable, with lower confidence.

## KV cache

For each rank:

```text
KV bytes =
    2
  * layers
  * KV heads per rank
  * head dimension
  * reserved tokens
  * KV element bytes
```

`KV heads per rank` is at least one. This conservatively captures KV-head
replication when tensor parallelism exceeds the number of KV heads.

Expected tokens use the requested input plus output length for every concurrent
sequence. Conservative tokens reserve the configured maximum model length for
every configured sequence.

## Other components

Activation, runtime-workspace, communication-buffer, and fragmentation values
are versioned priors. They intentionally carry `heuristic` confidence until
matching vLLM benchmarks replace them.

The operational recommendation target leaves the configured HBM headroom,
15% by default. A plan may physically fit yet be marked marginal when it misses
that reserve.

## Curated catalog

The Phase 4A catalog contains seven immutable BF16 revisions across compact,
small, medium, large, dense, and MoE model classes.

Parameter counts come from the Hugging Face model API; architecture comes from
`config.json`; artifact bytes come from `model.safetensors.index.json`.

## Current candidate search

Phase 1 enumerates:

- BF16 artifacts allowed by the serving request;
- the cataloged LUMI AI Factory vLLM runtime;
- power-of-two tensor parallelism;
- one serving replica;
- eligible `small-g` and `standard-g` allocation shapes;
- up to the node/GCD/GPU-hour bounds in the resource envelope.

Multi-node plans remain conditional until an exact container/network benchmark
validates them. Phase 2 adds replica optimization and Phase 3 adds an explicitly
heuristic `throughput-optimized` tier; measured calibration remains future work.

Phase 4B reuses this estimator in bounded integer searches for maximum
memory-safe context and concurrency. See `docs/capacity-envelopes.md`.

## Inspecting a plan

```bash
lumi-fit plan Qwen/Qwen3-32B \
  --resources examples/resources-model-first.yaml \
  --serving examples/serving-online.yaml \
  --view detail
```

Add `--format json` to inspect the complete recommendation bundle and every
constraint and memory component. See `docs/cli.md` for the full command
reference.
