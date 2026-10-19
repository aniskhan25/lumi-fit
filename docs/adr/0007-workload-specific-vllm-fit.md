# ADR 0007: Workload-specific vLLM fit replaces composite scoring

- Status: accepted
- Date: 2026-08-10

## Context

The earlier recommendation engine combined quality, performance, memory,
allocation, billing, topology, and runtime signals through objective-dependent
weights. That number did not answer the product's central question: whether a
model can meet a specific vLLM inference workload on LUMI.

## Decision

Separate technical serveability (support, per-GCD memory, and context) from an
optional task-quality requirement. Provide heuristic analytical target coverage
for initial planning. Define validated capacity as the 95% lower confidence
bound of maximum request goodput that satisfies the declared TTFT, TPOT,
optional end-to-end, attainment, and failure constraints. Normalize a deployment
to an eight-GCD LUMI node with `floor(8 / gcds_per_replica)` and calculate:

```text
100 * min(1, full_node_goodput_lcb95 / target_request_rate)
```

Task quality remains separate and is non-blocking unless the user declares a
minimum. Heuristic coverage is labeled as estimated and never presented as
validated SLO performance. Incomplete validated evidence produces `UNSCORED`;
a failed technical gate produces `INELIGIBLE`.

## Consequences

- Scores are workload-specific and evidence-backed.
- Efficient but unsuitable models cannot win by speed alone.
- Missing evidence is visible rather than encoded as a low number.
- Objective profiles, weighted score components, sensitivity views, and
  pairwise component comparisons are superseded. Their fields remain in the
  versioned contract only for backward compatibility and are emitted empty.
- Curated goodput evidence is required before recommendations show validated
  fit; quality evidence is required only when a quality minimum is requested.

## Amendment: progressive evidence

Technical serveability is support × memory × context and does not depend on
task-quality evidence. Quality is optional unless the user declares a minimum.
Before measured goodput exists, the tool reports heuristic analytical demand
coverage for useful initial planning. The original LCB formula remains the
separate validated fit and is never replaced by the estimate.
