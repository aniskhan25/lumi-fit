# Phase 4B memory-safe capacity envelopes

Phase 4B extends a recommendation from one requested workload point to a
bounded memory-safety envelope. It remains local and analytical: it does not
run a model, generate a benchmark, connect to LUMI, or claim a performance SLA.

## Reported values

Every feasible or conditional candidate contains a `CapacityEnvelope` with:

- the requested context and workload-wide concurrency;
- maximum memory-safe concurrency at the requested context;
- maximum memory-safe context at the requested concurrency;
- the model's cataloged context limit;
- every memory point evaluated by both searches;
- an explicit unknown analytical-performance capacity;
- a lower-cost and higher-capacity candidate reference when one exists.

Infeasible candidates do not receive an envelope. Their existing constraint
trace remains the useful diagnostic.

## Safety definition

A point is memory-safe when:

```text
conservative peak HBM per rank
    <= physical HBM per rank * (1 - requested headroom fraction)
```

This is stricter than physical fit. A point may fit in 64 GB of HBM while not
preserving the default 15% operational reserve.

The capacity estimator reuses the same componentized weight, KV-cache,
activation, workspace, communication, and fragmentation formulas as normal
plan analysis. It does not use a second closed-form approximation.

## Search bounds

Both searches are deterministic integer binary searches.

Concurrency search:

```text
context = requested max_model_length
total concurrency in [1, 65,536]
per-replica concurrency = ceil(total concurrency / replicas)
```

If the upper limit is still safe, `concurrency_search_capped` is true. The
reported point is then a tested lower bound rather than an unbounded maximum.

Context search:

```text
total concurrency = requested concurrent_sequences
context in [input_tokens + max_output_tokens, cataloged model context]
```

The maximum context can never exceed the immutable model catalog limit.

All evaluated points are stored in the recommendation bundle, including the
first unsafe point immediately beyond a memory-limited maximum. This makes the
boundary reproducible and explainable.

## Memory capacity versus performance capacity

The envelope answers whether the estimated memory placement is safe. It does
not assert that the maximum point meets latency, throughput, or arrival-rate
targets. Existing performance estimates describe the requested point only and
remain heuristic.

`analytical_performance_maximum_concurrency` therefore remains null until
comparable evidence supports a sustainable-capacity model. Its explanation is
persisted beside the memory-safe limits.

## Neighbor selection

Neighbors reference other candidates already present in the same deterministic
search result:

- `lower-cost` is the closest cheaper feasible plan that remains memory-safe at
  the requested point;
- `higher-capacity` is the least costly feasible plan with a strictly larger
  memory-safe concurrency boundary.

Resource-first searches may have no lower-cost neighbor because their
allocation is fixed. A missing neighbor is represented as null rather than
inventing a plan outside the supplied resource envelope.

## CLI presentation

The recommendation table shows `SAFE CONC` and `SAFE CTX` for selected plans.
`lumi-fit explain` shows the requested point, both limits, the catalog boundary,
probe count, performance-capacity status, and available neighbors.

The complete probe trace is available in JSON output under
`candidate.capacity_envelope.evaluated_points`.
