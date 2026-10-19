# ADR 0004: TUI-first interface with CLI and JSON parity

Status: accepted  
Date: 2026-08-09

## Context

LUMI users work primarily in terminals through SSH and Slurm. The `llmfit`
interaction model demonstrates that a model table, keyboard navigation, live
filters, plan mode, and simulated hardware are effective for fit exploration.
Automation still requires non-interactive commands.

## Decision

Running `lumi-fit` will open an interactive TUI. The top bar represents a
simulated LUMI allocation rather than detected local hardware. Scriptable CLI
commands expose the same analysis, and versioned JSON is the stable automation
contract.

The default view is a full-screen, keyboard-driven ranked model table rather
than a sequential questionnaire. Common allocation and workload values are
focusable controls across the top and recalculate the table immediately. A
control cycles through constrained values directly from the keyboard, while the
search control accepts free text. A complete editor remains available as an
overlay for advanced inputs. Search,
fit filtering, deterministic sorting, details, comparison, and model-first plan
mode operate over the same recommendation bundle used by the CLI.

## Consequences

TUI, CLI, and future interfaces must consume the same core result and cannot
implement separate scoring logic. Saved recommendation bundles can be explained
or exported without recalculation.
