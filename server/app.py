# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the CSV training Environment.

This module creates an HTTP server that exposes the MyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

from typing import Dict, Optional
import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from openenv.core.env_server.http_server import create_app

try:
    from models import MyAction, MyObservation
    from server.csv_env import MyEnvironment
except ImportError:
    # Fallback for development/direct execution
    from models import MyAction, MyObservation
    from server.csv_env import MyEnvironment

base_app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="",
)


app = FastAPI(title="CSV RL Environment API", description="OpenEnv-compatible CSV data curation environment")

# Store environments by session_id for concurrent sessions
environments: Dict[str, MyEnvironment] = {}


def get_or_create_env(session_id: str) -> MyEnvironment:
    """Get existing environment or create new one for the session."""
    if session_id not in environments:
        environments[session_id] = MyEnvironment(seed=42)
    return environments[session_id]

app.mount("/", base_app)

@app.get("/")
def root():
    return {"message": "Server running"}


@app.post("/reset")
async def reset_environment(task_id: Optional[str] = None, session_id: str = "default") -> MyObservation:
    """Reset the environment for a new episode."""
    env = get_or_create_env(session_id)
    try:
        obs = env.reset(task_id=task_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_environment(action: MyAction, session_id: str = "default") -> MyObservation:
    """Execute an action in the environment."""
    env = get_or_create_env(session_id)
    try:
        obs = env.step(action)
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_environment_state(session_id: str = "default") -> Dict:
    """Get the current state of the environment."""
    env = get_or_create_env(session_id)
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks."""
    return {"status": "healthy"}

@app.get("/schema")
async def get_schemas():
    """Get the JSON schemas for action and observation models."""
    return {
        "action_schema": MyAction.model_json_schema(),
        "observation_schema": MyObservation.model_json_schema(),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = "default"):
    """WebSocket endpoint for persistent sessions."""
    await websocket.accept()
    env = get_or_create_env(session_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "reset":
                task_id = message.get("task_id")
                obs = env.reset(task_id=task_id)
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})

            elif message.get("type") == "step":
                action_data = message.get("action", {})
                action = MyAction(**action_data)
                obs = env.step(action)
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})

            elif message.get("type") == "state":
                state = {
                    "episode_id": env.state.episode_id,
                    "step_count": env.state.step_count,
                }
                await websocket.send_json({"type": "state", "data": state})

            elif message.get("type") == "schema":
                schemas = {
                    "action_schema": MyAction.model_json_schema(),
                    "observation_schema": MyObservation.model_json_schema(),
                }
                await websocket.send_json({"type": "schema", "data": schemas})

            else:
                await websocket.send_json({"type": "error", "message": "Unknown message type"})

    except WebSocketDisconnect:
        # Clean up if needed
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 7860
        python -m my_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_env.server.app:app --workers 4
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=7860)
    # args = parser.parse_args()
    main()
