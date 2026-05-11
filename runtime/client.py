"""Lightweight Python client for the runtime engine.

A coordination engine talking to a remote runtime can use this client instead
of writing requests by hand. Stdlib-only (urllib + http.client), so no extra
dependencies for downstream coordinators.

    from runtime.client import Client
    c = Client("http://localhost:8765", token=os.environ.get("AGI_RUNTIME_TOKEN"))
    sid = c.create_session(budget={"max_usd": 1.0})["id"]
    result = c.chat(sid, "2 + 2")  # blocking
    job = c.submit_job(sid, "summarize ./README.md")  # async
    for ev in c.stream(job["id"]):
        print(ev)
    final = c.wait_job(job["id"])
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator
from urllib import error, parse, request


class ClientError(RuntimeError):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class Client:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 60.0) -> None:
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _req(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = request.Request(self.base + path, data=data, method=method, headers=self._headers())
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode() or "null"
                return json.loads(raw)
        except error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode() or "null")
            except Exception:
                payload = None
            raise ClientError(e.code, payload) from None

    # --- public API ---

    def health(self) -> dict:
        return self._req("GET", "/v1/health")

    def capabilities(self) -> dict:
        return self._req("GET", "/v1/capabilities")

    def metrics(self) -> dict:
        return self._req("GET", "/v1/metrics")

    def create_session(self, *, model: str | None = None, budget: dict | None = None, agent_kwargs: dict | None = None) -> dict:
        body: dict[str, Any] = {}
        if model is not None:
            body["model"] = model
        if budget is not None:
            body["budget"] = budget
        if agent_kwargs is not None:
            body["agent_kwargs"] = agent_kwargs
        return self._req("POST", "/v1/sessions", body)

    def list_sessions(self) -> list[dict]:
        return self._req("GET", "/v1/sessions")["sessions"]

    def get_session(self, sid: str) -> dict:
        return self._req("GET", f"/v1/sessions/{sid}")

    def delete_session(self, sid: str) -> dict:
        return self._req("DELETE", f"/v1/sessions/{sid}")

    def chat(self, sid: str, content: str, *, max_iterations: int = 25) -> dict:
        return self._req("POST", f"/v1/sessions/{sid}/messages", {"content": content, "max_iterations": max_iterations})

    def submit_job(self, sid: str, content: str, *, max_iterations: int = 25, metadata: dict | None = None) -> dict:
        body: dict[str, Any] = {"content": content, "max_iterations": max_iterations}
        if metadata is not None:
            body["metadata"] = metadata
        return self._req("POST", f"/v1/sessions/{sid}/jobs", body)

    def get_job(self, jid: str) -> dict:
        return self._req("GET", f"/v1/jobs/{jid}")

    def cancel_job(self, jid: str) -> dict:
        return self._req("POST", f"/v1/jobs/{jid}/cancel")

    def wait_job(self, jid: str, *, timeout: float = 120.0, poll: float = 0.1) -> dict:
        """Poll until the job reaches a terminal state. Returns the final job dict."""
        deadline = time.time() + timeout
        while True:
            job = self.get_job(jid)
            if job["state"] in {"succeeded", "failed", "cancelled"}:
                return job
            if time.time() > deadline:
                raise TimeoutError(f"job {jid} did not finish within {timeout}s")
            time.sleep(poll)

    def stream(self, jid: str) -> Iterator[dict]:
        """Yield SSE events as parsed dicts until the job emits `done`."""
        url = self.base + f"/v1/jobs/{jid}/stream"
        req = request.Request(url, headers=self._headers())
        with request.urlopen(req, timeout=self.timeout) as resp:
            event_name = None
            data_lines: list[str] = []
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n").rstrip("\r")
                if line == "":
                    if data_lines:
                        payload = "\n".join(data_lines)
                        try:
                            parsed = json.loads(payload)
                        except json.JSONDecodeError:
                            parsed = {"raw": payload}
                        if event_name:
                            parsed.setdefault("kind", event_name)
                        yield parsed
                        if event_name == "done":
                            return
                    event_name = None
                    data_lines = []
                    continue
                if line.startswith(":"):
                    # comment / keepalive
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:") :].strip())

    def get_memory(self, sid: str, *, q: str | None = None, k: int = 10) -> list[dict]:
        qs = []
        if q is not None:
            qs.append(("q", q))
        qs.append(("k", str(k)))
        path = f"/v1/sessions/{sid}/memory?{parse.urlencode(qs)}"
        return self._req("GET", path)["notes"]

    def save_memory(self, sid: str, text: str, tags: list[str] | None = None) -> dict:
        return self._req("POST", f"/v1/sessions/{sid}/memory", {"text": text, "tags": tags or []})
