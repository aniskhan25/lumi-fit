# Phase 3 terminal interface

Install the project and invoke `lumi-fit`, or run it from a checkout with:

```bash
PYTHONPATH=src python3 -m lumi_fit.cli
```

With no arguments, `lumi-fit` opens a full-screen terminal dashboard. It starts
on a ranked model table for a simulated four-GCD `small-g` allocation. The
most useful workload controls are selectable boxes at the top. Changing one
recalculates the table locally. The `e` editor remains available for complete
SLO and workload configuration.

| Key | Action |
|---|---|
| `Tab`/`Shift-Tab` | Select the next/previous top control |
| `Left`/`Right` | Select the previous/next value and recalculate immediately |
| `Enter`/`Space` | Advance the selected control to its next value |
| `Up`/`Down` or `j`/`k` | Navigate model or plan rows |
| `/` | Search model, partition, precision, or fit status |
| `f` | Cycle fit filters |
| `s` | Cycle fit/resource sort columns |
| `Enter` | Toggle the selected plan detail panel |
| `m`, then `c` | Mark a row and compare it with another |
| `p` | Show resource plans for the selected model |
| `b` | Return from resource-plan mode to model recommendations |
| `e` | Open the complete advanced workload and SLO editor |
| `h` | Open help |
| `q` | Quit |

The dashboard is deliberately recommendation-only: there are no download,
Slurm, job submission, or serving-engine actions.

## Commands

Resource-first discovery:

```bash
lumi-fit models \
  --resources examples/resources-one-node-four-gcds.yaml \
  --serving examples/serving-online.yaml \
  --model-selector examples/models-coding.yaml
```

Model-first allocation planning:

```bash
lumi-fit plan Qwen/Qwen3-32B \
  --resources examples/resources-model-first.yaml \
  --serving examples/serving-online.yaml
```

`--resources` is optional for `plan`; the default search is one to 16 GCDs and
one to two nodes across `small-g` and `standard-g`.

Compare an explicit newline-separated model list inside a fixed allocation:

```bash
lumi-fit compare models.txt \
  --resources examples/resources-one-node-four-gcds.yaml \
  --serving examples/serving-online.yaml
```

Save and explain without recomputation:

```bash
lumi-fit plan Qwen/Qwen3-32B \
  --serving examples/serving-online.yaml \
  --output recommendation.json

lumi-fit explain recommendation.json --plan <plan-id>
```

`explain` loads the exact versioned bundle. It does not reload catalogs or rerun
estimation, so reviewed evidence does not drift silently.

Recommendation tables include memory-safe maximum concurrency and context.
The explanation view shows the requested point, catalog context limit, bounded
search results, evaluated-point count, and available allocation neighbors.
These are memory-safety estimates and are displayed separately from heuristic
performance estimates.

Human output keeps plan memory feasibility, technical serveability, heuristic
capacity coverage, optional task quality, and validated fit separate. `explain`
also shows every gate and the evidence class behind each result.

## Phase 4D filters and views

Model discovery accepts direct filters without requiring a selector file:

```bash
lumi-fit models \
  --resources resources.yaml \
  --serving serving.yaml \
  --include 'Qwen/*' \
  --exclude '*235B*' \
  --family qwen3 \
  --license apache-2.0 \
  --language en \
  --capability coding \
  --min-parameters 4000000000 \
  --max-parameters 40000000000
```

Direct filters extend fields loaded with `--model-selector`. Result presentation
can then be filtered independently with repeatable `--status`, `--fit-band`,
`--partition`, and `--precision` flags.

Deterministic sort fields are `model`, `parameters`, `fit`, `hbm`,
`throughput`, `gpu-hours`, and `safe-concurrency`. Each has an intuitive default
direction, overridable with `--sort-direction asc|desc`.

Views are:

- `--view compact`: serveability, estimated coverage, quality, and validated fit;
- `--view detail`: additional columns and the complete explanation for every
  displayed plan;
- `--view compare`: retained for compatibility; no obsolete weighted pairwise
  comparisons are generated;
- `--all-plans`: include alternative and diagnostic layouts before filtering.

## CSV and JSON export

```bash
lumi-fit models ... --format csv --output recommendations.csv
lumi-fit models ... --format json --output recommendation.json
```

CSV contains the filtered and sorted displayed rows with a stable column order
and `\n` line endings. JSON always contains the complete, unfiltered evidence
bundle, and table output with `--output` continues to save JSON.

Supporting commands:

```bash
lumi-fit validate examples/serving-online.yaml
lumi-fit catalog validate
lumi-fit catalog models search qwen
lumi-fit catalog coverage
lumi-fit system show
lumi-fit doctor
```

All planning commands accept `--format json` for stdout and `--output PATH` for a
self-contained recommendation bundle. Human tables and the interactive dashboard
are views over that same bundle.

`catalog coverage` checks the versioned minimum model/family/capability and
parameter-band targets, plus immutable revision and artifact-size provenance.
Use `catalog coverage --json` for its machine-readable report.
