import os
from unittest.mock import Mock, patch

import pytest
from langchain.chat_models import init_chat_model
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.types import Command
from langsmith import RunTree, tracing_context

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


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.anyio
async def test_approval_resume_persists_request_annotation():
    from deep_agent import graph

    model = _ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/draft", "content": "draft"},
                        "id": "write-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Finished"),
        ]
    )
    with patch.object(graph, "DEFAULT_MODEL", model):
        agent = graph._build_agent().copy(update={"checkpointer": InMemorySaver()})
    client = Mock()
    tracer = LangChainTracer(client=client)
    config = {"configurable": {"thread_id": "approval"}, "callbacks": [tracer]}
    with tracing_context(enabled=True, client=client):
        paused = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Research this request"},
                        ]
                    )
                ]
            },
            config,
        )
        assert paused["__interrupt__"]
        initial = tracer.latest_run
        assert initial.metadata["hitl_resume"] is False
        assert "hitl-resume" not in initial.tags
        result = await agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config
        )
        resumed = tracer.latest_run
    assert result["messages"][-1].content == "Finished"
    assert resumed.id != initial.id
    assert resumed.metadata["user_request"] == "Research this request"
    assert resumed.metadata["hitl_resume"] is True
    assert "hitl-resume" in resumed.tags
    updates = [
        call
        for call in client.update_run.call_args_list
        if call.args and call.args[0] == resumed.id
    ]
    assert any(
        call.kwargs["extra"]["metadata"]["user_request"] == "Research this request"
        and "hitl-resume" in call.kwargs["tags"]
        for call in updates
    )


@pytest.mark.anyio
async def test_request_annotation_handles_nested_roots_and_missing_tracing():
    from deep_agent.graph import _RequestTraceMiddleware

    middleware = _RequestTraceMiddleware(interrupt_on={"write_file": True})
    client = Mock()
    root = RunTree(
        name="root",
        client=client,
        extra={"metadata": {"existing": 1}},
        tags=["existing"],
    )
    child = root.create_child("child")
    child.parent_run = root
    state = {"messages": [HumanMessage(content="old"), HumanMessage(content="x" * 600)]}
    with (
        patch("deep_agent.graph.get_current_run_tree", return_value=child),
        patch("deep_agent.graph.get_config", return_value={}),
    ):
        await middleware._annotate_root(state)
        assert root.metadata == {
            "existing": 1,
            "hitl_resume": False,
            "user_request": "x" * 500,
        }
        assert root.tags == ["existing"]
        await middleware._annotate_root({})
        client.update_run.side_effect = RuntimeError("tracing unavailable")
        await middleware._annotate_root(state)
    with patch("deep_agent.graph.get_current_run_tree", return_value=None):
        await middleware._annotate_root(state)
