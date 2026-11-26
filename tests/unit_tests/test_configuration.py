import os
from unittest.mock import patch

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_deepseek import ChatDeepSeek
from langgraph.pregel import Pregel

with patch.dict(
    "os.environ", {"DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY") or "test-key"}
):
    from deep_agent.graph import RO_AGENT, SUBAGENTS, SYSTEM_PROMPT


def test_graph_compiles() -> None:
    assert isinstance(RO_AGENT, Pregel)


def test_deepseek_model_initializes_without_network() -> None:
    with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
        model = init_chat_model("deepseek:deepseek-chat")
    assert isinstance(model, ChatDeepSeek)


def test_subagents_configured() -> None:
    names = {item["name"] for item in SUBAGENTS}
    assert names == {"researcher", "critic"}


def test_system_prompt_is_nonempty() -> None:
    assert len(SYSTEM_PROMPT.strip()) > 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("revision", "git_sha", "expected"),
    [
        ("deployed-sha", None, "deployed-sha"),
        (None, "fallback-sha", "fallback-sha"),
        ("preferred-sha", "fallback-sha", "preferred-sha"),
        ("", "fallback-sha", "fallback-sha"),
        (None, None, None),
    ],
)
async def test_revision_metadata(monkeypatch, revision, git_sha, expected) -> None:
    from deep_agent import graph

    for name, value in (
        ("LANGSMITH_REVISION_ID", revision),
        ("GIT_COMMIT_SHA", git_sha),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    class FakeToolCallingModel(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    monkeypatch.setattr(
        graph, "DEFAULT_MODEL", FakeToolCallingModel(responses=["Hello"])
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    agent = graph._build_agent()
    assert agent.config.get("metadata", {}).get("revision_id") == expected
    root_metadata = []

    class RootMetadataHandler(AsyncCallbackHandler):
        async def on_chain_start(
            self, serialized, inputs, *, parent_run_id=None, metadata=None, **kwargs
        ):
            if parent_run_id is None:
                root_metadata.append(metadata or {})

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"callbacks": [RootMetadataHandler()]},
    )
    assert result["messages"][-1].content == "Hello"
    assert len(root_metadata) == 1
    assert root_metadata[0].get("revision_id") == expected
