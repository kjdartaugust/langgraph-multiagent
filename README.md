# Minimal LangGraph Multi-Agent App

Two agents wired in a LangGraph flow:

```
researcher  -->  writer  -->  END
```

- **Researcher** — takes a topic, gathers facts (via OpenRouter).
- **Writer** — turns the research into a one-paragraph summary.
- State (`topic` -> `research` -> `summary`) is passed automatically between nodes.

## Setup

```bash
pip install -r requirements.txt
```

Get a free key at https://openrouter.ai/keys, then copy the template and paste
your key in:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

`graph.py` loads `.env` automatically (via python-dotenv), so you don't need to
export anything.

## Run

```bash
python graph.py "The history of coffee"
```

## The pattern to learn

1. Define a shared `State` (a `TypedDict`) — the memory that flows through the graph.
2. Write each agent as a function `node(state) -> {updated keys}`.
3. `add_node` / `add_edge` to wire the flow; `set_entry_point` + `END` mark the ends.
4. `compile()` then `invoke(initial_state)` runs it and returns the final state.
