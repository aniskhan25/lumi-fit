# lumi-fit

`lumi-fit` is a local terminal advisor for choosing LLMs and planning vLLM
inference resources on LUMI. It estimates model serveability, memory use,
capacity, and LUMI GPU-hour cost without connecting to LUMI or launching jobs.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/aniskhan25/lumi-fit.git
cd lumi-fit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

## Use the interactive interface

```bash
lumi-fit
```

The dashboard ranks models for a simulated LUMI allocation. Use `Tab` and
`Shift-Tab` to move between the controls, then `Left`/`Right`, `Enter`, or
`Space` to change values. Press `h` for the complete keyboard reference and
`q` to quit.

## Use the CLI

Find models for a fixed allocation:

```bash
lumi-fit models \
  --resources examples/resources-one-node-four-gcds.yaml \
  --serving examples/serving-online.yaml \
  --model-selector examples/models-coding.yaml
```

Find suitable LUMI resources for one model:

```bash
lumi-fit plan Qwen/Qwen3-32B \
  --resources examples/resources-model-first.yaml \
  --serving examples/serving-online.yaml
```

Save a recommendation and explain it later without recalculating:

```bash
lumi-fit plan Qwen/Qwen3-32B \
  --serving examples/serving-online.yaml \
  --output recommendation.json

lumi-fit explain recommendation.json --plan <plan-id>
```

Run `lumi-fit --help` for all commands, filters, and export formats.
