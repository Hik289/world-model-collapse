# World Model Collapse

World Model Collapse is a Python research codebase for studying how language-model agents maintain, update, and sometimes collapse internal world models during long-horizon planning tasks.

The repository contains deterministic task environments, LLM-backed and oracle-style agents, episode runners, cost tracking, experiment launch scripts, and analysis utilities. It is organized so that the source code can be reused independently while the experiment scripts remain available for reproducibility.

## Repository Structure

- `src/environments/`: deterministic planning environments and the shared environment API.
  - `graph_nav.py`: graph-navigation tasks.
  - `tool_dag.py`: dependency-structured tool-use tasks.
  - `stateful_puzzle.py`: stateful puzzle tasks with hidden dependencies.
- `src/agents/`: agent interfaces, prompt templates, JSON parsing, oracle agents, and LLM-backed agents.
- `src/evaluation/`: episode execution and JSONL logging utilities.
- `src/runner/`: batch runners, cost tracking, and pilot-slice orchestration.
- `experiments/`: reproducible experiment entry points for pilot runs, smoke tests, ablations, critical scans, and cross-harness comparisons.
- `analysis/`: scripts for computing acceptance checks, effect sizes, confidence intervals, lag analysis, and multiple-testing summaries.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Model Configuration

LLM runs are configured through environment variables. Do not commit keys, local proxy credentials, or generated logs.

OpenAI:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Optional Azure OpenAI fallback for model names prefixed with `azure:`:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/openai/v1"
export AZURE_OPENAI_KEY="your-azure-key"
```

Optional Anthropic-compatible local proxy:

```bash
export ANTHROPIC_PROXY_URL="http://127.0.0.1:18801/v1/messages"
```

Optional AWS Bedrock support requires local AWS credentials configured outside this repository.

## Quick Start

Run a small smoke experiment after setting `OPENAI_API_KEY`:

```bash
python experiments/stage5_smoke/run_stage5_smoke.py
```

The runner writes generated logs under `data/raw_logs/` and experiment summaries under the relevant experiment folder. These generated artifacts are ignored by Git by default.

You can also import the core components directly:

```python
from src.environments import ENV_REGISTRY

env_cls = ENV_REGISTRY["stateful_puzzle"]
env = env_cls()
obs = env.reset(
    task_config={
        "archetype": "demo",
        "stress_config": {
            "T": 20,
            "state_card": 5,
            "branching": 2,
            "obs_noise": "clean",
            "mut_rate": "static",
            "dep_density": 2,
        },
    },
    seed=123,
)

print(obs.text)
```

## Reproducibility Notes

The environments are designed to be deterministic: randomness flows through seeded per-environment RNG instances, state is canonicalized before hashing, and emitted unordered structures are sorted where possible.

Experiment scripts use fixed task-seed namespaces and write structured JSON/JSONL logs. For public release, generated logs, cost trackers, caches, and local environment files are excluded from the package.

## License

Add your preferred license before publishing if this repository will be distributed publicly.
