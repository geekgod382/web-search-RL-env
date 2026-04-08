"""
Inference Script — CSV RL Environment
======================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.

STDOUT FORMAT (do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.csv_env import MyEnvironment
from models import MyAction

# ── env / API config ──────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "")

# ── task config ───────────────────────────────────────────────────────────────
_raw_tasks = os.getenv("CSV_TASK", "easy,medium,hard")
TASK_NAMES  = [t.strip() for t in _raw_tasks.split(",") if t.strip()]
if not TASK_NAMES:
    TASK_NAMES = ["easy", "medium", "hard"]

BENCHMARK  = os.getenv("CSV_BENCHMARK", "csv-rl-env")

MAX_STEPS               = 8
TEMPERATURE             = 0.2
MAX_TOKENS              = 300
SUCCESS_SCORE_THRESHOLD = 0.5

# ── logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── LLM helpers ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data-cleaning agent operating on a CSV dataset.
    Each turn you receive the current task description, goal, and a preview of rows.
    You must respond with EXACTLY ONE action as a JSON object — nothing else.

    Supported operations and required fields:
      select           : {"task_id":"...", "operation":"select", "row_index": <int>}
      repair           : {"task_id":"...", "operation":"repair", "row_index": <int>, "field_name":"<col>", "value":"<str>"}
      remove_duplicate : {"task_id":"...", "operation":"remove_duplicate", "row_index": <int>}
      normalize        : {"task_id":"...", "operation":"normalize", "row_index": <int>, "field_name":"<col>", "value":"<str>"}
      impute           : {"task_id":"...", "operation":"impute", "row_index": <int>, "field_name":"<col>", "value":"<str>"}

    Rules:
    - row_index is the integer in the "row_index" field of preview_rows, NOT the list position.
    - Output ONLY valid JSON. No explanation, no markdown, no extra text.
""").strip()


def build_user_prompt(obs) -> str:
    try:
        rows_text = "\n".join(str(r) for r in obs.preview_rows)
        return textwrap.dedent(f"""
            task_id          : {obs.task_id}
            task_description : {obs.task_description}
            task_goal        : {obs.task_goal}
            remaining_issues : {obs.remaining_issues}
            progress         : {obs.progress_fraction:.2f}
            step_count       : {obs.step_count}

            preview_rows (use the "row_index" value from here in your action):
            {rows_text}

            Respond with ONE JSON action.
        """).strip()
    except Exception as exc:
        print(f"[DEBUG] build_user_prompt error: {exc}", flush=True)
        return "Respond with ONE JSON action."


def get_model_action(client: OpenAI, obs) -> dict:
    """Call the LLM and parse the JSON action it returns. Returns {} on any failure."""
    try:
        user_prompt = build_user_prompt(obs)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # strip markdown fences if the model wraps output in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM/parse error: {exc}", flush=True)
        return {}


def build_action(action_dict: dict) -> Optional[MyAction]:
    """Convert parsed dict to MyAction, returning None on failure."""
    try:
        if not action_dict:
            return None
        return MyAction(**action_dict)
    except Exception as exc:
        print(f"[DEBUG] Action build error: {exc}", flush=True)
        return None

# ── baseline fallback policies (if LLM fails) ────────────────────────────────

def easy_policy(obs) -> Optional[MyAction]:
    """Select every row tagged needs_review=yes."""
    for row in obs.preview_rows:
        if row.get("needs_review") == "yes":
            return MyAction(
                task_id=obs.task_id,
                operation="select",
                row_index=int(row["row_index"]),
            )
    return None


def medium_policy(obs) -> Optional[MyAction]:
    """Remove duplicates first, then repair wrong category labels."""
    for row in obs.preview_rows:
        if row.get("duplicate_of"):
            return MyAction(
                task_id=obs.task_id,
                operation="remove_duplicate",
                row_index=int(row["row_index"]),
            )
    correction_map = {"ai": "AI", "search": "Search", "dev": "Dev"}
    for row in obs.preview_rows:
        category = row.get("category", "")
        target = correction_map.get(category.lower())
        if target and category != target:
            return MyAction(
                task_id=obs.task_id,
                operation="repair",
                row_index=int(row["row_index"]),
                field_name="category",
                value=target,
            )
    return None


def hard_policy(obs) -> Optional[MyAction]:
    """Normalize company names, fix blank statuses, impute missing prices."""
    normalized_names = {
        "open ai":        "OpenAI",
        "openai":         "OpenAI",
        "google llc":     "Google",
        "meta platforms": "Meta Platforms",
    }
    for row in obs.preview_rows:
        company = row.get("company", "")
        normalized = normalized_names.get(company.lower())
        if normalized and company != normalized:
            return MyAction(
                task_id=obs.task_id,
                operation="normalize",
                row_index=int(row["row_index"]),
                field_name="company",
                value=normalized,
            )
    for row in obs.preview_rows:
        if row.get("status", "").lower() in {"unknown", ""}:
            return MyAction(
                task_id=obs.task_id,
                operation="normalize",
                row_index=int(row["row_index"]),
                field_name="status",
                value="pending",
            )
    for row in obs.preview_rows:
        if row.get("price", "") == "":
            notes = row.get("notes", "")
            if "estimate" in notes:
                value = "25.00" if "25" in notes else "30.00"
                return MyAction(
                    task_id=obs.task_id,
                    operation="impute",
                    row_index=int(row["row_index"]),
                    field_name="price",
                    value=value,
                )
    return None


FALLBACK_POLICIES = {
    "easy":   easy_policy,
    "medium": medium_policy,
    "hard":   hard_policy,
}


def get_fallback_action(obs) -> Optional[MyAction]:
    """Return a rule-based fallback action, or None if no rule applies."""
    policy = FALLBACK_POLICIES.get(obs.task_id)
    if policy is None:
        return None
    try:
        return policy(obs)
    except Exception as exc:
        print(f"[DEBUG] Fallback policy error: {exc}", flush=True)
        return None

# ── main ──────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, env: MyEnvironment, task_name: str) -> None:
    """Run a single task episode with full error handling."""
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    obs                      = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_name)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # 1. try LLM
            action_dict   = get_model_action(client, obs)
            action        = build_action(action_dict)
            used_fallback = False

            # 2. fallback to rule-based policy if LLM failed
            if action is None:
                print(f"[DEBUG] step={step} LLM action invalid, using fallback policy", flush=True)
                action        = get_fallback_action(obs)
                used_fallback = True

            # 3. if both failed, skip step gracefully
            if action is None:
                action_str = str(action_dict)
                error_msg  = "invalid_action_no_fallback"
                reward     = 0.0
                done       = obs.done
            else:
                try:
                    obs        = env.step(action)
                    reward     = float(obs.reward or 0.0)
                    done       = obs.done
                    error_msg  = "fallback_used" if used_fallback else None
                    action_str = str(action.model_dump(exclude_none=True))
                except Exception as step_exc:
                    print(f"[DEBUG] env.step error: {step_exc}", flush=True)
                    reward     = 0.0
                    done       = False
                    error_msg  = "env_step_error"
                    action_str = str(action_dict)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        if obs is not None:
            score   = float(obs.task_score or 0.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unexpected error in task {task_name}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] Failed to create OpenAI client: {exc}", flush=True)
        client = None

    try:
        env = MyEnvironment(seed=42)
    except Exception as exc:
        print(f"[DEBUG] Failed to create environment: {exc}", flush=True)
        # Emit [END] for each task so validator doesn't hang
        for task_name in TASK_NAMES:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    for task_name in TASK_NAMES:
        run_task(client, env, task_name)


if __name__ == "__main__":
    main()