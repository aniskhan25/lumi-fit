# ADR 0002: resource-first and model-first planning

Status: accepted  
Date: 2026-08-09

## Context

Local model-fit tools treat detected hardware as fixed. LUMI resources are
requested through Slurm, so hardware is sometimes a constraint and sometimes a
decision variable.

## Decision

One analysis core supports two inverse searches:

- resource-first ranks model-serving configurations inside a fixed allocation;
- model-first ranks allocation plans for a fixed model and serving target.

Both produce the same immutable `ServingPlan` and analysis result types.

## Consequences

The candidate generator has separate search entry points, but feasibility,
memory, billing, scoring, explanation, and local presentation remain shared.
This avoids
two tools drifting into inconsistent answers.
