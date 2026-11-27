"""Deep Agent graph for deployment."""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime

from deepagents import create_deep_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.config import get_config
from langgraph_sdk.runtime import ServerRuntime
from langsmith import get_current_run_tree

from deep_agent.sandbox import get_or_create_sandbox

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "deepseek:deepseek-chat")

SYSTEM_PROMPT = """
You are a deep agent.

Workflow:
1. Write and maintain a todo list for non-trivial requests.
2. Delegate focused fact-finding to subagents when helpful.
3. Store intermediate drafts in files when the task is long.
4. Before finalizing, critique your work for risks, gaps, and missing constraints.
5. Return concise, actionable output.

- Prefer concrete evidence over assumptions.
- State unresolved uncertainty explicitly.
- Keep output compact unless the user asks for depth.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=UTC).isoformat()


SUBAGENTS = [
    {
        "name": "researcher",
        "description": "Use for evidence collection and source-grounded fact finding.",
        "system_prompt": (
            "You are a focused researcher. Gather evidence, list assumptions, and "
            "report contradictions clearly."
        ),
        "tools": [utc_now],
    },
    {
        "name": "critic",
        "description": "Use for adversarial review of drafts and plans.",
        "system_prompt": (
            "You are a critical reviewer. Find weak logic, untested assumptions, and "
            "missing constraints."
        ),
        "tools": [utc_now],
    },
]


class _RequestTraceMiddleware(HumanInTheLoopMiddleware):
    @property
    def name(self):
        return "HumanInTheLoopMiddleware"

    async def aafter_model(self, state, runtime):
        try:
            return await super().aafter_model(state, runtime)
        finally:
            await self._annotate_root(state)

    async def _annotate_root(self, state):
        with contextlib.suppress(Exception):
            root = get_current_run_tree()
            if root is None:
                return
            while root.parent_run is not None:
                root = root.parent_run
            config = get_config()
            if root.parent_run_id is not None:
                # Some SDK versions retain parent IDs without parent_run links.
                for handler in getattr(config.get("callbacks"), "handlers", []):
                    traced_root = getattr(handler, "run_map", {}).get(
                        str(root.trace_id)
                    )
                    if traced_root is not None:
                        root = traced_root
                        break
                else:
                    return
            # interrupt() consumes the resume payload into the node's scratchpad.
            scratchpad = config.get("configurable", {}).get("__pregel_scratchpad")
            resumed = bool(scratchpad and scratchpad.resume)
            metadata = root.extra.get("metadata", {})
            resumed = resumed or metadata.get("hitl_resume", False)
            annotation = {"hitl_resume": resumed}
            for message in reversed(state.get("messages", [])):
                if isinstance(message, HumanMessage):
                    annotation["user_request"] = message.text[:500]
                    break
            root.add_metadata(annotation)
            if resumed and "hitl-resume" not in (root.tags or []):
                root.add_tags(["hitl-resume"])
            await asyncio.to_thread(
                root.client.update_run, root.id, extra=root.extra, tags=root.tags
            )


def _build_agent(backend=None):
    return create_deep_agent(
        model=DEFAULT_MODEL,
        tools=[utc_now],
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        subagents=SUBAGENTS,
        # You can disable these if you want to run without interrupts
        interrupt_on={"execute": True, "write_file": True},
        middleware=[
            _RequestTraceMiddleware(interrupt_on={"execute": True, "write_file": True})
        ],
        name="deep_agent",
    )


RO_AGENT = _build_agent()


@contextlib.asynccontextmanager
async def get_agent(config: RunnableConfig, runtime: ServerRuntime):
    ert = runtime.execution_runtime
    if ert:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        backend = await get_or_create_sandbox(thread_id)
        yield _build_agent(backend=backend)
    else:
        yield RO_AGENT
