from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deep_agent import sandbox as sandbox_module

pytestmark = pytest.mark.anyio


async def test_awrite_returns_success() -> None:
    sandbox = SimpleNamespace(write=AsyncMock())
    backend = sandbox_module.LangSmithBackend(sandbox)

    result = await backend.awrite("/tmp/x.md", "hi")

    assert result.path == "/tmp/x.md"
    assert result.error is None
    sandbox.write.assert_awaited_once_with("/tmp/x.md", b"hi")


async def test_awrite_returns_write_failure() -> None:
    sandbox = SimpleNamespace(write=AsyncMock(side_effect=RuntimeError("write failed")))
    backend = sandbox_module.LangSmithBackend(sandbox)

    result = await backend.awrite("/tmp/x.md", "hi")

    assert result.path is None
    assert result.error == "Failed to write file '/tmp/x.md': write failed"


async def test_creates_snapshot_and_sandbox_when_missing() -> None:
    client = SimpleNamespace(
        list_snapshots=AsyncMock(return_value=[]),
        create_snapshot=AsyncMock(return_value=SimpleNamespace(id="snapshot-id")),
        create_sandbox=AsyncMock(return_value=SimpleNamespace(name="sandbox-id")),
    )
    sandbox_module._backends.clear()
    with patch.object(sandbox_module, "AsyncSandboxClient", return_value=client):
        backend = await sandbox_module.get_or_create_sandbox("thread-id")
        cached = await sandbox_module.get_or_create_sandbox("thread-id")

    assert cached is backend
    assert backend.id == "sandbox-id"
    client.create_snapshot.assert_awaited_once_with(
        name="deep-agent",
        docker_image="python:3",
        fs_capacity_bytes=2 * 1024**3,
        timeout=180,
    )
    client.create_sandbox.assert_awaited_once_with(
        snapshot_id="snapshot-id", timeout=180
    )


async def test_reuses_existing_ready_snapshot() -> None:
    client = SimpleNamespace(
        list_snapshots=AsyncMock(
            return_value=[
                SimpleNamespace(id="existing-id", name="deep-agent", status="ready")
            ]
        ),
        wait_for_snapshot=AsyncMock(),
        create_snapshot=AsyncMock(),
    )

    assert (
        await sandbox_module._ensure_snapshot(client, "deep-agent", "python:3")
        == "existing-id"
    )
    client.create_snapshot.assert_not_awaited()
    client.wait_for_snapshot.assert_not_awaited()


async def test_waits_for_existing_build_and_filters_exact_name() -> None:
    client = SimpleNamespace(
        list_snapshots=AsyncMock(
            return_value=[
                SimpleNamespace(id="other-id", name="deep-agent-old", status="ready"),
                SimpleNamespace(id="building-id", name="deep-agent", status="building"),
            ]
        ),
        wait_for_snapshot=AsyncMock(return_value=SimpleNamespace(id="building-id")),
        create_snapshot=AsyncMock(),
    )

    assert (
        await sandbox_module._ensure_snapshot(client, "deep-agent", "python:3")
        == "building-id"
    )
    client.wait_for_snapshot.assert_awaited_once_with("building-id", timeout=180)
    client.create_snapshot.assert_not_awaited()


async def test_searches_all_snapshot_pages() -> None:
    others = [SimpleNamespace(name="deep-agent-old") for _ in range(100)]
    client = SimpleNamespace(
        list_snapshots=AsyncMock(
            side_effect=[
                others,
                [SimpleNamespace(id="later-id", name="deep-agent", status="ready")],
            ]
        ),
        create_snapshot=AsyncMock(),
    )

    assert (
        await sandbox_module._ensure_snapshot(client, "deep-agent", "python:3")
        == "later-id"
    )
    assert client.list_snapshots.await_args_list[1].kwargs["offset"] == 100
    client.create_snapshot.assert_not_awaited()
