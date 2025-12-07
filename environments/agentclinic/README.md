# AgentClinic - Responses API

Multi-agent medical diagnosis environment using OpenAI's Responses API for reasoning models.

## Installation

```bash
# Install the environment
uv pip install -e environments/agentclinic/

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### MedQA Dataset (214 cases)

```bash
uv run --active -m verifiers.scripts.eval \
  -m o4-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic_response \
  -n 10 \
  --rollouts-per-example 2 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_medqa_extended.jsonl"}'
```

### NEJM Dataset (119 cases)

```bash
uv run --active -m verifiers.scripts.eval \
  -m o4-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic_response \
  -n 10 \
  --rollouts-per-example 2 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_nejm_extended.jsonl"}'
```


### Environment Args

```bash
--env-args '{
  "dataset_path": "agentclinic_medqa_extended.jsonl",
  "max_turns": 20,
  "patient_model": "gpt-4o-mini",
  "measurement_model": "gpt-4o-mini",
  "moderator_model": "gpt-4o-mini"
}'
```

