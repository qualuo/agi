"""Tiny Python client for the runtime HTTP server.

A coordination engine running in another process (or another language)
talks to the runtime over HTTP. This module exists so Python coordinators
don't have to handcraft requests.

    from runtime.client import RuntimeClient
    rc = RuntimeClient("http://127.0.0.1:8765")
    task = rc.submit("summarize ./README.md")
    final = rc.wait(task["id"], timeout=60)
    for ev in rc.events(task["id"]):
        print(ev["kind"], ev["data"])

Stdlib only (urllib) — same constraint as the server.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional
from urllib import error, request
from urllib.parse import urlencode


class RuntimeError_(RuntimeError):
    """Raised when the server returns a non-2xx status."""
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class RuntimeClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- low-level ---------------------------------------------------------

    def _req(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict] = None,
        query: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout if timeout is not None else self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except error.HTTPError as e:
            raise RuntimeError_(e.code, e.read().decode("utf-8", errors="replace"))

    # --- tasks -------------------------------------------------------------

    def submit(
        self,
        instruction: str,
        *,
        budget: Optional[dict] = None,
        metadata: Optional[dict] = None,
        parent_id: Optional[str] = None,
    ) -> dict:
        body: dict[str, Any] = {"instruction": instruction}
        if budget is not None:
            body["budget"] = budget
        if metadata is not None:
            body["metadata"] = metadata
        if parent_id is not None:
            body["parent_id"] = parent_id
        return self._req("POST", "/tasks", body=body)

    def get(self, task_id: str) -> dict:
        return self._req("GET", f"/tasks/{task_id}")

    def tree(self, task_id: str) -> dict:
        return self._req("GET", f"/tasks/{task_id}/tree")

    def events(self, task_id: str) -> list[dict]:
        return self._req("GET", f"/tasks/{task_id}/events")["events"]

    def list(self, *, status: Optional[str] = None) -> list[dict]:
        query = {"status": status} if status else None
        return self._req("GET", "/tasks", query=query)["tasks"]

    def cancel(self, task_id: str) -> dict:
        return self._req("POST", f"/tasks/{task_id}/cancel")

    def wait(self, task_id: str, *, timeout: float = 60.0) -> dict:
        return self._req(
            "POST",
            f"/tasks/{task_id}/wait",
            query={"timeout": str(timeout)},
            timeout=timeout + 5.0,
        )

    def stream(self, task_id: str) -> Iterator[dict]:
        """Iterate over SSE events for a task. Yields parsed JSON event dicts.
        Stops when the server closes the connection (task reached terminal state)."""
        url = f"{self.base_url}/tasks/{task_id}/stream"
        req = request.Request(url, headers={"Accept": "text/event-stream"})
        with request.urlopen(req, timeout=self.timeout) as resp:
            buf = b""
            for chunk in resp:
                buf += chunk
                while b"\n\n" in buf:
                    raw_event, buf = buf.split(b"\n\n", 1)
                    for line in raw_event.split(b"\n"):
                        if line.startswith(b"data: "):
                            payload = line[len(b"data: "):]
                            try:
                                yield json.loads(payload.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue

    # --- skills ------------------------------------------------------------

    def list_skills(self) -> list[dict]:
        return self._req("GET", "/skills")["skills"]

    def get_skill(self, name: str) -> dict:
        return self._req("GET", f"/skills/{name}")

    def add_skill(self, *, name: str, when: str, body: str, tags: Optional[list[str]] = None) -> dict:
        return self._req("POST", "/skills", body={
            "name": name, "when": when, "body": body, "tags": tags or [],
        })

    def remove_skill(self, name: str) -> dict:
        return self._req("DELETE", f"/skills/{name}")

    # --- ops ---------------------------------------------------------------

    def health(self) -> dict:
        return self._req("GET", "/health")

    def metrics(self) -> dict:
        return self._req("GET", "/metrics")
