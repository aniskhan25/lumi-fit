# ADR 0006: local recommendations only

Status: accepted  
Date: 2026-08-09

## Context

The intended product follows `llmfit`: it evaluates possible configurations and
returns scores and recommendations. LUMI differs because resources are planned
rather than auto-detected, but this does not require operational cluster tools.

Slurm rendering, runtime launch commands, live probes, and benchmark job plans
would broaden the product from decision support into deployment operations.
That is not the desired interface.

## Decision

`lumi-fit` is a local, recommendation-only application. It reads versioned
catalogs and user resource/workload descriptions, then produces ranked plans,
scores, explanations, and local exports.

It will not connect to LUMI, inspect live state, generate Slurm or vLLM launch
instructions, download weights, or submit/monitor jobs.

Runtime and partition information remains in the catalog solely to determine
whether a hypothetical serving configuration is credible.

## Consequences

- Normal operation is deterministic, offline, and non-destructive.
- The final user action is saving or exporting a recommendation.
- Phase 4 focuses on catalog breadth, capacity envelopes, comparison quality,
  and interface refinement.
- Empirical evidence may be curated or imported as inert data, but the tool does
  not generate or execute the workload that produced it.
- ADR 0005 is superseded.
