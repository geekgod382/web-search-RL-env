from pathlib import Path

from server.my_env_environment import MyEnvironment
from models import MyAction


def easy_policy(observation):
    for row in observation.preview_rows:
        if row.get("needs_review") == "yes":
            return MyAction(task_id=observation.task_id, operation="select", row_index=int(row["row_index"]))
    for row in observation.preview_rows:
        if int(row["row_index"]) not in getattr(easy_policy, "selected", set()):
            return MyAction(task_id=observation.task_id, operation="select", row_index=int(row["row_index"]))
    return None


def medium_policy(observation):
    for row in observation.preview_rows:
        if row.get("duplicate_of"):
            return MyAction(task_id=observation.task_id, operation="remove_duplicate", row_index=int(row["row_index"]))
    correction_map = {
        "ai": "AI",
        "Search": "Search",
        "Dev": "Dev",
    }
    for row in observation.preview_rows:
        category = row.get("category", "")
        target = correction_map.get(category.lower(), category)
        if category != target:
            return MyAction(
                task_id=observation.task_id,
                operation="repair",
                row_index=int(row["row_index"]),
                field_name="category",
                value=target,
            )
    return None


def hard_policy(observation):
    normalized_names = {
        "open ai": "OpenAI",
        "openai": "OpenAI",
        "google llc": "Google",
        "meta platforms": "Meta Platforms",
    }
    for row in observation.preview_rows:
        company = row.get("company", "")
        normalized = normalized_names.get(company.lower())
        if normalized and company != normalized:
            return MyAction(
                task_id=observation.task_id,
                operation="normalize",
                row_index=int(row["row_index"]),
                field_name="company",
                value=normalized,
            )
    for row in observation.preview_rows:
        status = row.get("status", "").lower()
        if status in {"unknown", ""}:
            return MyAction(
                task_id=observation.task_id,
                operation="normalize",
                row_index=int(row["row_index"]),
                field_name="status",
                value="pending",
            )
    for row in observation.preview_rows:
        if row.get("price", "") == "":
            notes = row.get("notes", "")
            if "estimate" in notes:
                estimate_value = "25.00" if "25" in notes else "30.00"
                return MyAction(
                    task_id=observation.task_id,
                    operation="impute",
                    row_index=int(row["row_index"]),
                    field_name="price",
                    value=estimate_value,
                )
    return None


def run_task(env, task_id, policy):
    observation = env.reset(task_id=task_id)
    scores = []
    while not observation.done:
        action = policy(observation)
        if action is None:
            break
        observation = env.step(action)
        scores.append(observation.task_score)
    return observation


def main():
    env = MyEnvironment(seed=42)
    total_score = 0.0
    outputs = []

    for task_id, policy in [("easy", easy_policy), ("medium", medium_policy), ("hard", hard_policy)]:
        result = run_task(env, task_id, policy)
        outputs.append((task_id, result.task_score, result.progress_fraction, result.remaining_issues))
        total_score += result.task_score

    avg_score = total_score / len(outputs)
    print("Baseline task results:")
    for task_id, score, progress, remaining in outputs:
        print(f"- {task_id.title()}: score={score:.3f}, progress={progress:.3f}, remaining_issues={remaining}")
    print(f"Average baseline score: {avg_score:.3f}")


if __name__ == "__main__":
    main()
