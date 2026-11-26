import os
from unittest.mock import patch

from langchain.chat_models import init_chat_model
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
