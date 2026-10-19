# Superseded: Phase 4C weighted comparison and objective sensitivity

The weighted composite, objective sensitivity, and score-component comparison
design implemented during Phase 4C was superseded by
[ADR 0007](adr/0007-workload-specific-vllm-fit.md).

Current recommendation bundles keep the old contract fields readable for saved
bundle compatibility but emit `objective_sensitivity` and `comparisons` as
empty collections. The `objective` request field is likewise accepted but does
not influence scoring or ranking.

The active interface compares separate, decision-relevant facts:

- plan feasibility and per-GCD memory fit;
- technical serveability from support, memory, and context;
- optional task-quality status;
- heuristic estimated capacity coverage;
- workload-specific, evidence-backed validated LUMI-vLLM fit;
- resources and LUMI billing.

See [the current scoring specification](scoring.md).
