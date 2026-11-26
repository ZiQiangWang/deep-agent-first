"""LangSmith sandbox backend.

Adapted from open-swe (agent/integrations/langsmith.py).
"""

from __future__ import annotations

import os

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox
from langsmith.sandbox import AsyncSandbox, AsyncSandboxClient

DEFAULT_SNAPSHOT_NAME = "deep-agent"
DEFAULT_SNAPSHOT_IMAGE = "python:3"
SNAPSHOT_FS_CAPACITY_BYTES = 2 * 1024**3
SNAPSHOT_PAGE_SIZE = 100

_backends: dict[str, LangSmithBackend] = {}


class LangSmithBackend(BaseSandbox):
    """LangSmith sandbox backend using the async SDK.

    Overrides aexecute/awrite/adownload_files/aupload_files to use the async
    SDK natively. The sync variants raise NotImplementedError since the
    framework always calls the async versions.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._default_timeout: int = 300

    @property
    def id(self) -> str:
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError("Use aexecute()")

    async def aexecute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = await self._sandbox.run(command, timeout=effective_timeout)
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        raise NotImplementedError("Use awrite()")

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        try:
            await self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path, files_update=None)
        except Exception as exc:  # noqa: BLE001
            # The backend protocol returns write failures instead of raising them.
            return WriteResult(error=f"Failed to write file '{file_path}': {exc}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("Use adownload_files()")

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            content = await self._sandbox.read(path)
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("Use aupload_files()")

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            await self._sandbox.write(path, content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


async def get_or_create_sandbox(thread_id: str) -> LangSmithBackend:
    """Get a cached sandbox for this thread, or create a new one."""
    if backend := _backends.get(thread_id):
        return backend

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
        "LANGSMITH_API_KEY_PROD"
    )
    snapshot_name = os.environ.get("SANDBOX_TEMPLATE_NAME", DEFAULT_SNAPSHOT_NAME)
    snapshot_image = os.environ.get("SANDBOX_TEMPLATE_IMAGE", DEFAULT_SNAPSHOT_IMAGE)

    client = AsyncSandboxClient(api_key=api_key)
    snapshot_id = await _ensure_snapshot(client, snapshot_name, snapshot_image)
    sandbox = await client.create_sandbox(snapshot_id=snapshot_id, timeout=180)

    backend = LangSmithBackend(sandbox)
    _backends[thread_id] = backend
    return backend


async def _ensure_snapshot(
    client: AsyncSandboxClient,
    snapshot_name: str,
    snapshot_image: str,
) -> str:
    """Return the ID of a ready snapshot, creating one if it does not exist."""
    offset = 0
    while True:
        snapshots = await client.list_snapshots(
            name_contains=snapshot_name, limit=SNAPSHOT_PAGE_SIZE, offset=offset
        )
        for snapshot in snapshots:
            if snapshot.name == snapshot_name:
                if snapshot.status != "ready":
                    snapshot = await client.wait_for_snapshot(snapshot.id, timeout=180)
                return snapshot.id
        if len(snapshots) < SNAPSHOT_PAGE_SIZE:
            break
        offset += len(snapshots)

    snapshot = await client.create_snapshot(
        name=snapshot_name,
        docker_image=snapshot_image,
        fs_capacity_bytes=SNAPSHOT_FS_CAPACITY_BYTES,
        timeout=180,
    )
    return snapshot.id
