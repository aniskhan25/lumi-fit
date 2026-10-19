# ADR 0005: render deployment plans without submitting jobs

Status: superseded by ADR 0006  
Date: 2026-08-09

## Context

Submitting a Slurm job consumes project allocation and creates cluster state.
The first product goal is planning and explanation, not autonomous operations.

## Decision

Version 0.1 may render `sbatch`, vLLM, smoke-test, and benchmark commands, but it
does not submit, cancel, or delete jobs.

## Consequences

This decision was superseded when the product scope was narrowed to local
recommendations only. No deployment artifacts are generated.
