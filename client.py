# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyAction, MyObservation


class MyEnv(
    EnvClient[MyAction, MyObservation, State]
):
    """
    Client for the My Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MyAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyAction) -> Dict:
        return {
            "task_id": action.task_id,
            "operation": getattr(action.operation, "value", action.operation),
            "row_index": action.row_index,
            "field_name": action.field_name,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        metadata = obs_data.get("metadata") or payload.get("metadata") or {}
        observation = MyObservation(
            task_id=obs_data.get("task_id", ""),
            step_count=obs_data.get("step_count", 0),
            task_description=obs_data.get("task_description", ""),
            task_goal=obs_data.get("task_goal", ""),
            remaining_issues=obs_data.get("remaining_issues", 0),
            progress_fraction=obs_data.get("progress_fraction", 0.0),
            preview_rows=obs_data.get("preview_rows", []),
            task_score=obs_data.get("task_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
