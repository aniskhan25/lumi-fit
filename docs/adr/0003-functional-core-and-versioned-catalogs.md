# ADR 0003: functional core and versioned catalogs

Status: accepted  
Date: 2026-08-09

## Context

LUMI partition limits, billing rules, container versions, and runtime support
can change. Shell probes and remote metadata are also difficult to test
deterministically.

## Decision

The planning core consumes validated immutable contracts and performs no shell,
network, Slurm, SSH, or filesystem I/O. Side-effecting adapters normalize data
before calling the core. LUMI and runtime facts live in versioned YAML catalogs
with source URLs and verification dates.

## Consequences

Offline results are reproducible and unit-testable. Catalog drift is handled by
reviewing and versioning a new snapshot, never by probing LUMI during analysis.
