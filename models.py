# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env CSV RL Environment.

The environment exposes typed actions and observations for
CSV collection, curation, and refining tasks.
"""

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TaskOperation(str, Enum):
    """Supported operations in the CSV RL environment."""

    SELECT = "select"
    REPAIR = "repair"
    REMOVE_DUPLICATE = "remove_duplicate"
    NORMALIZE = "normalize"
    IMPUTE = "impute"


class CsvAction(Action):
    """Action for the CSV RL environment."""

    task_id: str = Field(..., description="Current task difficulty: easy, medium, or hard")
    operation: TaskOperation = Field(..., description="Type of operation to perform")
    row_index: Optional[int] = Field(
        None,
        description="Index of the row to act upon. Use preview_rows to discover row_index values.",
    )
    field_name: Optional[str] = Field(
        None,
        description="Column name to repair, normalize, or impute.",
    )
    value: Optional[str] = Field(
        None,
        description="Proposed new value for repair, normalization, or imputation operations.",
    )


class CsvObservation(Observation):
    """Observation from the CSV RL environment."""

    task_id: str = Field(..., description="Active task difficulty for the current episode")
    step_count: int = Field(..., description="Number of steps taken in the current episode")
    task_description: str = Field(..., description="Short description of the active task")
    task_goal: str = Field(..., description="Concrete goal for the active task")
    remaining_issues: int = Field(..., description="Number of unresolved issues remaining")
    progress_fraction: float = Field(
        ..., description="Estimated completion progress for the current task", ge=0.0, le=1.0
    )
    preview_rows: List[Dict[str, str]] = Field(
        ..., description="Visible preview of the current dataset rows"
    )
    task_score: float = Field(
        ..., description="Current grader score for the active task", ge=0.0, le=1.0
    )


MyAction = CsvAction
MyObservation = CsvObservation
