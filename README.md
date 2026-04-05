---
title: CSV RL Environment
emoji: 🗃️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - rl
  - csv
  - data
  - curation
  - openenv
---

# CSV RL Environment

This repository implements a full OpenEnv-compatible reinforcement learning environment for CSV data collection, curation, and refinement. The environment exposes:

- Typed `MyAction` and `MyObservation` models
- `reset()`, `step()`, and `state` APIs
- Three graded tasks: `easy`, `medium`, and `hard`
- Partial progress reward signals and a final task score in `[0.0, 1.0]`
- A FastAPI-backed OpenEnv server suitable for Hugging Face Spaces and Docker deployment

## Task Suite

### Easy
- Identify rows that require CSV collection or review.
- Reward is given for correct selections and penalized for incorrect choices.

### Medium
- Curate by removing duplicate rows and correcting invalid category labels.
- Reward is obtained from issue reduction and final dataset consistency.

### Hard
- Refine the dataset by normalizing company names, standardizing status values, and imputing missing prices.
- The reward model captures partial progress while encouraging a fully cleaned dataset.

## Action Schema

`MyAction` fields:

- `task_id` (`str`): `easy`, `medium`, or `hard`
- `operation` (`str`): one of `select`, `repair`, `remove_duplicate`, `normalize`, `impute`
- `row_index` (`int`, optional): row index to target
- `field_name` (`str`, optional): column name to modify
- `value` (`str`, optional): proposed value for repair, normalization, or imputation

## Observation Schema

`MyObservation` fields:

- `task_id` (`str`)
- `step_count` (`int`)
- `task_description` (`str`)
- `task_goal` (`str`)
- `remaining_issues` (`int`)
- `progress_fraction` (`float`)
- `preview_rows` (`list[dict]`)
- `task_score` (`float`)
- `done` (`bool`)
- `reward` (`float`)
- `metadata` (`dict`)

## Reward and Grading

The environment provides partial progress signals and an interpretable final score:

- Positive reward for correct selections, repairs, removals, normalizations, and imputations
- Small negative reward for invalid or incorrect actions
- `task_score` in each observation reflects actual progress toward the target dataset

## Quick Start

### Build the Docker image

```bash
cd my_env
docker build -t my_env-env:latest -f server/Dockerfile .
```

### Run locally

```bash
docker run --rm -p 7860:7860 my_env-env:latest
```

### Connect with the client

```python
from my_env import MyAction, MyEnv

with MyEnv(base_url="http://localhost:7860") as env:
    obs = env.reset()
    print(obs.task_description)
    action = MyAction(task_id=obs.task_id, operation="select", row_index=1)
    result = env.step(action)
    print(result.progress_fraction, result.reward, result.metadata)
```

## Baseline Inference

A baseline heuristic is included in `baseline.py`. Run it to verify reproducible scores across all tasks:

```bash
python baseline.py
```

## Deploying to Hugging Face Spaces

Use the built-in Docker-ready OpenEnv configuration:

```bash
openenv push
```

If you prefer a manual Docker deployment:

```bash
docker build -t my_env-env:latest -f server/Dockerfile .
```

Then push the image to Hugging Face or another container registry.

## Recent Project Changes

- Added support for the `IMAGE_NAME` environment variable in `inference.py`.
- This update enables the inference workflow to accept an optional image name from the environment, while retaining the existing `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` configuration.

## Local Development

### Run the FastAPI server

```bash
cd my_env
uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
```

### Start via Python module

```bash
python -m server.app
```

## Project Structure

```
my_env/
+-- .dockerignore
+-- README.md
+-- baseline.py
+-- client.py
+-- models.py
+-- openenv.yaml
+-- pyproject.toml
+-- server/
�   +-- __init__.py
�   +-- app.py
�   +-- my_env_environment.py
+-- uv.lock
```

## Notes

- `server/app.py` now enables multiple concurrent sessions for WebSocket clients
- The environment cycles through `easy`, `medium`, and `hard` tasks by default
- `task_score` in each observation gives a reproducible graded reward signal
