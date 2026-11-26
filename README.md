# Deep Agents Template

Deployment template for a deep agent built with `create_deep_agent(...)`.

## What this template gives you

- A deployable deep agent graph at `src/deep_agent/graph.py`.
- Explicit workflow prompt (plan, delegate, critique, finalize).
- Two predefined sub-agents (`researcher`, `critic`).
- Human-in-the-loop interrupts on `execute` and `write_file`.
- A `uv`-managed local workflow with a small `Makefile` wrapper and starter tests.

## Prerequisites

- A DeepSeek API key (or a key for your chosen model provider)
- A [LangSmith](https://smith.langchain.com/) account (Plus plan or higher) to deploy

## Quickstart

1. Sync the project and configure environment:

```bash
uv sync
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env` before starting the server. If `.env` already
exists, update `DEEP_AGENT_MODEL=deepseek:deepseek-chat` and add your DeepSeek
key instead of overwriting the file. `DEEP_AGENT_MODEL` can still select a
different provider when needed.

2. Start the dev server:

```bash
uv run langgraph dev
```

3. Deploy to LangSmith:

```bash
uv run langgraph deploy
```

See the [CLI docs](https://docs.langchain.com/langsmith/cli#deploy) for deploy options.

To set up CI instead, push this repo to GitHub and configure your deployment through the LangSmith UI.

## Tests and lint

```bash
make test
make integration-tests
make lint
make format
```

Integration tests are skipped unless `DEEPSEEK_API_KEY` is set.

## Reference docs

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents quickstart: https://docs.langchain.com/oss/python/deepagents/quickstart
- LangSmith CLI: https://docs.langchain.com/langsmith/cli
