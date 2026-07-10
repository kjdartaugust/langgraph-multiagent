# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal, teaching-oriented LangGraph multi-agent example. Two agents run in
sequence: a **researcher** gathers notes on a topic, a **writer** turns those
notes into a summary. The whole point is to demonstrate the LangGraph pattern
clearly, so keep `app.py` simple and well-commented — readability beats features.

## Architecture

Single file: `app.py`.

- **State** (`TypedDict`) is the shared memory that flows through the graph:
  `topic` (input) → `research` (set by researcher) → `summary` (set by writer).
  Each node returns only the keys it changes; LangGraph merges them into state.
- **Nodes are plain functions** `node(state) -> {updated keys}`. `researcher` and
  `writer` both call `call_llm()`.
- **The graph** is built in `build_graph()`: `set_entry_point("researcher")`,
  edge `researcher → writer`, edge `writer → END`, then `.compile()`.
  `main()` calls `.invoke({"topic": ...})` and prints `final_state["summary"]`.
- **LLM access** is via OpenRouter, which is OpenAI-API-compatible — the `openai`
  client is pointed at `base_url="https://openrouter.ai/api/v1"`. The client is
  built lazily in `get_client()` so the graph compiles without a key set. `MODEL`
  is a `:free` OpenRouter model so the demo runs without paid credits.

## Environment & commands

The system `python`/`python3` on this machine is a hermes venv without pip. Use
the project's uv-managed venv at `.venv` instead.

```bash
# Recreate the venv / install deps (this machine has a TLS cert issue — the
# --native-tls flag is required or pypi.org fails with UnknownIssuer).
uv venv --python 3.11
uv pip install --native-tls -r requirements.txt

# Compile-check the graph without needing an API key:
.venv/Scripts/python -c "import app; app.build_graph(); print('OK')"

# Run the full flow (needs a key):
export OPENROUTER_API_KEY="sk-or-..."   # PowerShell: $env:OPENROUTER_API_KEY="..."
.venv/Scripts/python app.py "The history of coffee"
```

There are no tests or lint config; if adding checks, wire them into this same venv.
