# ADR 0001: inference-serving-only initial scope

Status: accepted  
Date: 2026-08-09

## Context

Training, fine-tuning, batch inference, and persistent online serving have
different resource models. Combining all of them would delay a reliable answer
to the initial question: which model can be served on a planned LUMI allocation,
and which allocation is needed for a selected model?

## Decision

Version 0.1 plans inference serving only. It supports online and offline batch
serving because both use an inference runtime and the same weight/KV placement
model. Training and fine-tuning are deferred.

## Consequences

The initial domain excludes gradients, optimizer state, and training
parallelism. Memory, performance, scoring, tests, and the TUI can focus on
context, concurrency, batching, latency, throughput, and allocation duration.

