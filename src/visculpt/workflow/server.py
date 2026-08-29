"""Agent Server entrypoint for the Sculpt workflow graph."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from visculpt.workflow.config import ServiceConfig
from visculpt.workflow.graph import SculptAgentWorkflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def get_server_workflow() -> SculptAgentWorkflow:
    """Build live dependencies once per Agent Server worker process."""
    return SculptAgentWorkflow.from_config(workdir=PROJECT_ROOT)


server_workflow = get_server_workflow()
graph = server_workflow.graph.with_config(
    {
        "recursion_limit": (
            server_workflow.config.workflow.effective_recursion_limit
        ),
        "configurable": {
            "artifact_root": str(
                server_workflow.config.artifact_root(PROJECT_ROOT)
            )
        },
    }
)


def apply_runtime_service_config(services: ServiceConfig) -> None:
    """Retarget the shared live clients used by all compiled graph tools."""
    dependencies = server_workflow.dependencies
    if dependencies.blender_client is None or dependencies.sam3_client is None:
        raise RuntimeError(
            "The Agent Server workflow does not expose mutable service clients"
        )
    dependencies.blender_client.config = services.blender_rpc_config()
    dependencies.sam3_client.config = services.sam3_config()
    # Later node configuration reads must observe the same service settings.
    server_workflow.config = replace(server_workflow.config, services=services)
