# LUMI serveability, estimated coverage, and validated fit

`lumi-fit` exposes three different answers rather than collapsing infrastructure,
quality, and evidence into one weighted composite.

## 1. Technical serveability

Technical serveability answers: **can this deployment be hosted by vLLM on the
declared LUMI allocation?** It requires no model execution.

```text
SERVEABLE = support × memory × context
```

The gates are:

1. **Support** — the exact architecture, precision, runtime, and layout are
   supported by the selected vLLM/ROCm profile.
2. **Memory** — weights, KV cache, and runtime memory fit independently on each
   GCD at the declared context and concurrency.
3. **Context** — the deployment covers `minimum_context_coverage` of the
   prompt/output-length distribution.

The result is `ELIGIBLE`, `INELIGIBLE`, or `UNKNOWN`. LUMI-G exposes eight
64 GiB GCD devices per node, so the memory calculation never treats the node's
aggregate HBM as one flat pool. See the
[LUMI-G hardware guide](https://docs.lumi-supercomputer.eu/hardware/lumig/) and
[vLLM ROCm requirements](https://docs.vllm.ai/en/v0.22.0/getting_started/installation/gpu/).

When a request does not contain a token-length distribution, the contract
creates a one-point distribution from `input_tokens` and `max_output_tokens`
with probability 1.0. Minimum context coverage defaults to 100%.

## 2. Optional task-quality requirement

Task quality is separate from technical serveability:

- no `minimum_quality_score`: `NOT REQUIRED` and no ranking penalty;
- threshold plus comparable evidence above it: `PASS`;
- threshold plus evidence below it: `FAIL`;
- threshold without comparable evidence: `UNKNOWN`.

A quality failure or unknown result does not claim that the model cannot load
on LUMI. It prevents the deployment from receiving a validated application-fit
score when the user explicitly made quality a requirement. Parameter count,
popularity, and capability tags are not silently converted into official
quality evidence.

## 3. Estimated capacity coverage

The analytical performance prior provides a useful planning score before LUMI
benchmark evidence exists:

```text
equivalent_replicas_per_node = floor(8 / gcds_per_replica)
analytical_full_node_capacity =
    per_replica_request_capacity_lower × equivalent_replicas_per_node

ESTIMATED_CAPACITY_COVERAGE = 100 × min(
    1,
    analytical_full_node_capacity / target_request_rate
)
```

This value is explicitly `heuristic`. It is suitable for initial ranking and
benchmark planning, but it is not a claim that TTFT, TPOT, failure, or
attainment SLOs have been demonstrated. Without a target request rate the tool
reports raw analytical capacity and `NO TARGET` instead of a coverage score.

## 4. Validated LUMI-vLLM fit

A request counts as good only when it succeeds and meets every declared TTFT,
TPOT, and optional end-to-end limit. This follows vLLM's documented
[per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
and [`vllm bench serve` goodput](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve/)
semantics.

Matching measured or calibrated evidence records the maximum goodput at which
attainment and failure constraints pass. Evidence must match model revision,
artifact, runtime, precision, context, GCD count, tensor parallelism,
concurrency, and token-length distribution.

```text
full_node_goodput_lcb95 =
    floor(8 / gcds_per_replica) × deployment_goodput_lcb95

VALIDATED_LUMI_VLLM_FIT = 100 × min(
    1,
    full_node_goodput_lcb95 / target_request_rate
)
```

The validated score is `UNSCORED` until the complete SLO declaration and
matching evidence exist. It also requires an explicit quality requirement to
pass; quality is automatically permitted when no minimum was requested.

## 5. Ranking

Ranking is lexicographic:

1. plan feasibility;
2. technical serveability;
3. optional quality requirement: `PASS`/`NOT REQUIRED`, `UNKNOWN`, then `FAIL`;
4. validated fit descending, when available;
5. estimated capacity coverage descending;
6. separately reported task quality as a tie-breaker only after validated fit
   reaches 100;
7. raw analytical capacity, billing, and stable plan ID.

There is no arbitrary weighted composite. The legacy `objective` field and
profile catalog remain readable for saved-input compatibility but do not affect
current scoring or ranking.

The standard output therefore keeps these facts visible:

```text
SERVEABILITY=...
ESTIMATED_CAPACITY_COVERAGE=... (heuristic)
QUALITY=...
VALIDATED_LUMI_VLLM_FIT=...
```
