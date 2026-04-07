# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CSV training Environment Implementation.

This environment provides a CSV data collection, curation, and refinement
reinforcement learning task suite that follows OpenEnv conventions.
"""

from enum import Enum
from random import Random
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation
except ImportError:
    from models import MyAction, MyObservation


from openenv.core.rubrics.base import Rubric

class EasyTaskRubric(Rubric):
    def forward(self, action, observation) -> float:
        return observation.reward or 0.0

class MediumTaskRubric(Rubric):
    def forward(self, action, observation) -> float:
        return observation.reward or 0.0

class HardTaskRubric(Rubric):
    def forward(self, action, observation) -> float:
        return observation.reward or 0.0

class CsvRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.easy = EasyTaskRubric()    # auto-registered as child!
        self.medium = MediumTaskRubric()
        self.hard = HardTaskRubric()
    
    def forward(self, action, observation) -> float:
        return observation.reward or 0.0


class TaskOperation(str, Enum):
    SELECT = "select"
    REPAIR = "repair"
    REMOVE_DUPLICATE = "remove_duplicate"
    NORMALIZE = "normalize"
    IMPUTE = "impute"


class MyEnvironment(Environment):
    """CSV RL environment that exposes three graded tasks."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    TASK_ORDER: Tuple[str, ...] = ("easy", "medium", "hard")
    MAX_STEPS_PER_EPISODE: int = 8

    def __init__(self, seed: int = 0):
        self._rng = Random(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._task_id = "easy"
        self._current_rows: List[Dict[str, str]] = []
        self._target_data: Dict[str, object] = {}
        self._selected_indices: set[int] = set()
        self._removed_indices: set[int] = set()
        self._fixed_fields: Dict[int, Dict[str, str]] = {}
        self._initial_issue_count = 0
        self._task_description = ""
        self._task_goal = ""
        self._done = False
        self.rubric = CsvRubric()

    def reset(self, task_id: Optional[str] = None) -> MyObservation:
        """Start a new episode for the next or requested task."""
        self._reset_count += 1
        self._task_id = task_id or self.TASK_ORDER[(self._reset_count - 1) % len(self.TASK_ORDER)]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._selected_indices.clear()
        self._removed_indices.clear()
        self._fixed_fields.clear()
        self._done = False
        self._build_task(self._task_id)

        return MyObservation(
            task_id=self._task_id,
            step_count=self._state.step_count,
            task_description=self._task_description,
            task_goal=self._task_goal,
            remaining_issues=self._initial_issue_count,
            progress_fraction=0.0,
            preview_rows=self._preview_rows(),
            task_score=0.0,
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._task_id,
                "max_steps": self.MAX_STEPS_PER_EPISODE,
                "task_order": ",".join(self.TASK_ORDER),
            },
        )

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        """Execute one operation against the active CSV task."""
        if self._done:
            return self._current_observation(
                reward=0.0,
                message="Episode already completed. Reset to start a new task.",
            )

        self._state.step_count += 1
        reward, message = self._apply_action(action)
        remaining_issues = self._compute_remaining_issues()
        progress_fraction = self._compute_progress(remaining_issues)
        task_score = self._compute_task_score(progress_fraction)
        self._done = remaining_issues == 0 or self._state.step_count >= self.MAX_STEPS_PER_EPISODE

        return MyObservation(
            task_id=self._task_id,
            step_count=self._state.step_count,
            task_description=self._task_description,
            task_goal=self._task_goal,
            remaining_issues=remaining_issues,
            progress_fraction=progress_fraction,
            preview_rows=self._preview_rows(),
            task_score=task_score,
            done=self._done,
            reward=round(max(-0.1, min(reward, 0.4)), 3),
            metadata={
                "message": message,
                "task_score": round(task_score, 3),
                "step_reward": round(reward, 3),
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def _build_task(self, task_id: str) -> None:
        if task_id == "easy":
            self._build_easy_task()
        elif task_id == "medium":
            self._build_medium_task()
        elif task_id == "hard":
            self._build_hard_task()
        else:
            raise ValueError(f"Unsupported task_id: {task_id}")

    def _build_easy_task(self) -> None:
        self._task_description = "Collect the rows that require CSV data collection and review."
        self._task_goal = "Select every row tagged for review without selecting incorrect rows."
        self._current_rows = [
            {"id": "1", "title": "North region survey", "source": "remote", "needs_review": "no"},
            {"id": "2", "title": "Customer feedback", "source": "partner", "needs_review": "yes"},
            {"id": "3", "title": "Sales pipeline", "source": "synthetic", "needs_review": "no"},
            {"id": "4", "title": "Product catalog", "source": "scraped", "needs_review": "yes"},
            {"id": "5", "title": "Market benchmark", "source": "api", "needs_review": "no"},
            {"id": "6", "title": "Pricing audit", "source": "manual", "needs_review": "yes"},
            {"id": "7", "title": "Support tickets", "source": "crm", "needs_review": "yes"},
        ]
        self._target_data = {"review_indices": {1, 3, 5, 6}}
        self._initial_issue_count = len(self._target_data["review_indices"])

    def _build_medium_task(self) -> None:
        self._task_description = "Curate the dataset by removing duplicates and correcting category labels."
        self._task_goal = "Remove duplicate rows and correct any invalid category values."
        self._current_rows = [
            {"id": "1", "title": "Search engine guide", "category": "Search", "duplicate_of": "", "price": "12.99"},
            {"id": "2", "title": "Search engine guide", "category": "Search", "duplicate_of": "1", "price": "12.99"},
            {"id": "3", "title": "AI platform overview", "category": "AI", "duplicate_of": "", "price": "39.99"},
            {"id": "4", "title": "AI platform overview", "category": "ai", "duplicate_of": "3", "price": "39.99"},
            {"id": "5", "title": "Developer manual", "category": "Dev", "duplicate_of": "", "price": "24.50"},
        ]
        self._target_data = {
            "category_corrections": {0: "Search", 1: "Search", 2: "AI", 3: "AI", 4: "Dev"},
            "duplicate_indices": {1, 3},
        }
        self._initial_issue_count = len(self._target_data["duplicate_indices"]) + self._count_medium_category_issues()

    def _build_hard_task(self) -> None:
        self._task_description = "Refine the dataset by normalizing company names, standardizing status values, and imputing missing prices."
        self._task_goal = "Produce a clean, normalized dataset ready for model training."
        self._current_rows = [
            {"id": "1", "company": "open ai", "status": "pending", "price": "", "notes": "estimate 25"},
            {"id": "2", "company": "Meta Platforms", "status": "Confirmed", "price": "89.00", "notes": "approved"},
            {"id": "3", "company": "Google LLC", "status": "unknown", "price": "70.00", "notes": "review"},
            {"id": "4", "company": "openai", "status": "", "price": "30.00", "notes": "final"},
        ]
        self._target_data = {
            "normalization": {
                0: {"company": "OpenAI", "status": "pending", "price": "25.00"},
                1: {"company": "Meta Platforms", "status": "confirmed", "price": "89.00"},
                2: {"company": "Google", "status": "pending", "price": "70.00"},
                3: {"company": "OpenAI", "status": "pending", "price": "30.00"},
            }
        }
        self._initial_issue_count = self._count_hard_field_issues()

    def _apply_action(self, action: MyAction) -> Tuple[float, str]:
        if action.task_id != self._task_id:
            return -0.05, f"Incorrect task_id: expected {self._task_id}, got {action.task_id}"

        if self._task_id == "easy":
            return self._apply_easy_action(action)
        if self._task_id == "medium":
            return self._apply_medium_action(action)
        if self._task_id == "hard":
            return self._apply_hard_action(action)
        return -0.05, "Unsupported task"

    def _apply_easy_action(self, action: MyAction) -> Tuple[float, str]:
        if action.operation != TaskOperation.SELECT:
            return -0.05, "Easy task only supports select operations."
        if action.row_index is None or not 0 <= action.row_index < len(self._current_rows):
            return -0.05, "Missing or invalid row_index for easy task."

        if action.row_index in self._selected_indices:
            return 0.0, "Row already selected."
        self._selected_indices.add(action.row_index)

        targets = self._target_data["review_indices"]
        if action.row_index in targets:
            return 0.25, "Correct row selection."
        return -0.1, "Selected a row that does not require review."

    def _apply_medium_action(self, action: MyAction) -> Tuple[float, str]:
        if action.operation == TaskOperation.REMOVE_DUPLICATE:
            if action.row_index is None or not 0 <= action.row_index < len(self._current_rows):
                return -0.05, "Missing or invalid row_index for duplicate removal."
            if action.row_index in self._removed_indices:
                return 0.0, "Row already removed."
            duplicate_indices = self._target_data["duplicate_indices"]
            self._removed_indices.add(action.row_index)
            if action.row_index in duplicate_indices:
                return 0.35, "Duplicate row removed correctly."
            return -0.1, "Row removed that was not a duplicate."

        if action.operation == TaskOperation.REPAIR:
            if action.row_index is None or action.field_name != "category" or action.value is None:
                return -0.05, "Repair requires row_index, field_name='category', and value."
            if not 0 <= action.row_index < len(self._current_rows):
                return -0.05, "Invalid row_index for repair."
            if action.row_index in self._removed_indices:
                return -0.05, "Cannot repair a removed row."

            current_value = self._current_rows[action.row_index]["category"]
            self._current_rows[action.row_index]["category"] = action.value
            self._fixed_fields.setdefault(action.row_index, {})["category"] = action.value
            target_value = self._target_data["category_corrections"][action.row_index]
            if action.value == target_value and current_value != action.value:
                return 0.25, "Category corrected successfully."
            if action.value == current_value:
                return 0.0, "No effective change to category."
            return -0.1, "Category correction is not the expected value."

        return -0.05, "Medium task only supports repair and remove_duplicate operations."

    def _apply_hard_action(self, action: MyAction) -> Tuple[float, str]:
        if action.operation not in {TaskOperation.NORMALIZE, TaskOperation.IMPUTE}:
            return -0.05, "Hard task only supports normalize and impute operations."
        if action.row_index is None or action.field_name is None or action.value is None:
            return -0.05, "Hard task actions require row_index, field_name, and value."
        if not 0 <= action.row_index < len(self._current_rows):
            return -0.05, "Invalid row_index for hard task."
        if action.row_index in self._removed_indices:
            return -0.05, "Cannot modify a removed row."

        row = self._current_rows[action.row_index]
        field_name = action.field_name
        target_fields = self._target_data["normalization"][action.row_index]
        if field_name not in target_fields:
            return -0.1, f"Field '{field_name}' is not part of the hard task target set."

        previous_value = row.get(field_name, "")
        row[field_name] = action.value
        self._fixed_fields.setdefault(action.row_index, {})[field_name] = action.value
        if action.value == target_fields[field_name] and previous_value != action.value:
            return 0.3, f"{field_name} normalized or imputed correctly."
        if action.value == previous_value:
            return 0.0, f"No change applied to {field_name}."
        return -0.1, f"Value for {field_name} is not the expected normalization target."

    def _preview_rows(self) -> List[Dict[str, str]]:
        preview: List[Dict[str, str]] = []
        for index, row in enumerate(self._current_rows):
            if index in self._removed_indices:
                continue
            preview.append({"row_index": str(index), **row})
            if len(preview) >= 5:
                break
        return preview

    def _compute_remaining_issues(self) -> int:
        if self._task_id == "easy":
            targets = self._target_data["review_indices"]
            return len(targets - self._selected_indices)

        if self._task_id == "medium":
            remaining_duplicates = len(self._target_data["duplicate_indices"] - self._removed_indices)
            category_issues = self._count_medium_category_issues()
            return remaining_duplicates + category_issues

        if self._task_id == "hard":
            return self._count_hard_field_issues()

        return 0

    def _compute_progress(self, remaining_issues: int) -> float:
        if self._initial_issue_count == 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - remaining_issues / self._initial_issue_count))

    def _compute_task_score(self, progress_fraction: float) -> float:
        return round(progress_fraction, 3)

    def _count_medium_category_issues(self) -> int:
        issues = 0
        for index, row in enumerate(self._current_rows):
            if index in self._removed_indices:
                continue
            expected = self._target_data["category_corrections"].get(index)
            if expected is None:
                continue
            if row["category"] != expected:
                issues += 1
        return issues

    def _count_hard_field_issues(self) -> int:
        issues = 0
        for index, row in enumerate(self._current_rows):
            if index in self._removed_indices:
                continue
            expected = self._target_data["normalization"][index]
            for field_name, target_value in expected.items():
                if row.get(field_name, "") != target_value:
                    issues += 1
        return issues

    def _current_observation(self, reward: float, message: str) -> MyObservation:
        remaining_issues = self._compute_remaining_issues()
        progress_fraction = self._compute_progress(remaining_issues)
        task_score = self._compute_task_score(progress_fraction)
        return MyObservation(
            task_id=self._task_id,
            step_count=self._state.step_count,
            task_description=self._task_description,
            task_goal=self._task_goal,
            remaining_issues=remaining_issues,
            progress_fraction=progress_fraction,
            preview_rows=self._preview_rows(),
            task_score=task_score,
            done=self._done,
            reward=round(reward, 3),
            metadata={"message": message, "task_score": task_score},
        )
