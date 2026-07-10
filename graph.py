"""
Minimal LangGraph multi-agent app.

Flow:  researcher --> writer --> END

- The Researcher takes a topic and gathers info (via OpenRouter).
- The Writer takes that research and produces a short summary.
- State is passed automatically between nodes by LangGraph.

Run:
    pip install -r requirements.txt
    cp .env.example .env        # then put your OpenRouter key in .env
    python graph.py "The history of coffee"
"""

import os
import sys
import time
from typing import TypedDict

# Windows terminals often default to cp1252, which can't print characters like a
# Unicode hyphen the model may return. Force UTF-8 so output never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# On Windows, use the OS certificate store for TLS (fixes "unable to get local
# issuer certificate" behind a corporate/self-signed root CA). We scope this to
# Windows only: on Linux hosts like Vercel the OS trust store may be empty, so
# injecting it would break every HTTPS call — there, Python's bundled CAs work.
if sys.platform == "win32":
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass

from dotenv import load_dotenv  # reads key/value pairs from a local .env file
load_dotenv()  # loads .env into os.environ so OPENROUTER_API_KEY is available

from langgraph.graph import StateGraph, END
from openai import OpenAI, RateLimitError  # OpenRouter speaks the OpenAI API.


# --- LLM helper ------------------------------------------------------------
# A free model on OpenRouter so this runs without paid credits (override via .env).
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free").strip()


def get_client() -> OpenAI:
    """Build the client lazily (only when we actually call the model).

    OpenRouter is OpenAI-API-compatible: same client, different base_url + key.
    """
    # .strip() guards against trailing newlines/spaces (common when a key is
    # pasted into a hosting dashboard's env-var box) — those make an illegal
    # HTTP Authorization header and cause a confusing "Connection error".
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY (get a free key at openrouter.ai/keys).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def call_llm(system: str, user: str, retries: int = 3) -> str:
    """Send one prompt to the model and return the text reply.

    Free OpenRouter models are sometimes rate-limited upstream (HTTP 429), so we
    retry a few times, honoring the server's suggested wait.
    """
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == retries - 1:
                raise
            wait = getattr(e.response, "headers", {}).get("Retry-After")
            wait = int(wait) if wait else 5
            print(f"  ...rate-limited, retrying in {wait}s")
            time.sleep(wait)


# --- Shared state ----------------------------------------------------------
# This dict is the "memory" that flows through the graph. Each node reads the
# keys it needs and returns the keys it wants to update. LangGraph merges them.
class State(TypedDict):
    topic: str      # input:  what to research
    research: str   # set by researcher, read by writer
    summary: str    # set by writer, the final output


# --- Agents (graph nodes) --------------------------------------------------
def researcher(state: State) -> dict:
    """Agent 1: gather raw facts and notes about the topic."""
    print(f"\n[researcher] gathering info on: {state['topic']}")
    research = call_llm(
        system="You are a research assistant. Given a topic, list the key "
               "facts, context, and interesting points as concise bullet notes.",
        user=f"Topic: {state['topic']}",
    )
    # Return only the key we changed; LangGraph merges it into the state.
    return {"research": research}


def writer(state: State) -> dict:
    """Agent 2: turn the research notes into a clean summary."""
    print("[writer] writing summary from research...")
    summary = call_llm(
        system="You are a writer. Turn the given research notes into a clear, "
               "engaging summary of about one paragraph.",
        user=f"Research notes:\n{state['research']}",
    )
    return {"summary": summary}


# --- Build the graph -------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    # Register each agent as a node.
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)

    # Wire the flow: start -> researcher -> writer -> end.
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


# --- Run -------------------------------------------------------------------
def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "The history of coffee"

    app = build_graph()
    # invoke() runs the whole flow and returns the final state.
    final_state = app.invoke({"topic": topic})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(final_state["summary"])


if __name__ == "__main__":
    main()
