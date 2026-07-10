"""
Vercel serverless function: POST a topic, run the LangGraph researcher->writer
flow, and return the research + summary as JSON.

The two-agent flow itself lives in graph.py at the repo root (reused as-is). We
add the repo root to sys.path so this function can import it, and vercel.json's
includeFiles makes sure graph.py is bundled alongside this function.

(The module is named graph.py, not app.py, because Vercel's Python preset treats
a root app.py as a web-server entrypoint and expects a top-level `app` object.)
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Make the repo-root modules importable from inside the api/ folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph  # noqa: E402  -- our researcher/writer graph

# Build the graph once at cold start; Fluid Compute reuses it across requests.
_graph = graph.build_graph()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            topic = json.loads(raw).get("topic", "").strip()
        except (json.JSONDecodeError, AttributeError):
            topic = ""

        if not topic:
            return self._send(400, {"error": "Send JSON like {\"topic\": \"...\"}."})

        try:
            state = _graph.invoke({"topic": topic})
            self._send(200, {"research": state["research"], "summary": state["summary"]})
        except Exception as e:  # surface upstream/model errors to the caller
            self._send(500, {"error": str(e)})

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)
