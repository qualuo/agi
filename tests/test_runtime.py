"""Runtime smoke tests — exercise the task store, event bus, graph executor,
and HTTP server without requiring the Anthropic API.

We do this by registering only the `noop` task kind plus a small synthetic
handler that doesn't call out. The runtime treats agents as just one kind of
handler, so this isolates the orchestration layer.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.runtime.capabilities import RUNTIME_CAPABILITIES
from agi.runtime.events import EventBus
from agi.runtime.graph import GraphExecutor, GraphSpec, NodeSpec
from agi.runtime.tasks import TaskSpec, TaskStatus, TaskStore
from agi.runtime.worker import Worker, WorkerContext


# ---------- bus + store ----------

def test_event_bus_publish_and_subscribe_with_history():
    bus = EventBus()
    bus.publish("task.a", "task.queued", {"v": 1})
    received = []
    sub = bus.subscribe(prefix="task.")
    try:
        bus.publish("task.a", "task.started", {"v": 2})
        bus.publish("graph.x", "graph.started", {"v": 3})
        # Drain queue with a small sleep to allow delivery on this thread.
        while True:
            try:
                received.append(sub.queue.get_nowait())
            except Exception:
                break
        kinds = [e.kind for e in received]
        assert "task.started" in kinds
        assert "graph.started" not in kinds  # different prefix
    finally:
        bus.unsubscribe(sub)
    hist = bus.history("task.")
    assert any(e.kind == "task.queued" for e in hist)


def test_task_store_dedup_and_state_transitions():
    store = TaskStore()
    s = TaskSpec(kind="noop", input={"result": 1}, dedup_key="x")
    t1 = store.submit(s)
    t2 = store.submit(s)
    assert t1.id == t2.id  # dedup
    store.mark_running(t1.id)
    assert store.get(t1.id).status == TaskStatus.RUNNING
    store.mark_succeeded(t1.id, result={"ok": True})
    assert store.get(t1.id).status == TaskStatus.SUCCEEDED
    assert store.get(t1.id).status.terminal


# ---------- worker ----------

def test_worker_executes_noop_and_publishes_events():
    store = TaskStore()
    bus = EventBus()
    registry = {
        "noop": lambda ctx, t: {"result": t.spec.input.get("payload", "ok")},
        "boom": lambda ctx, t: (_ for _ in ()).throw(RuntimeError("expected")),
    }
    worker = Worker(store, bus, registry, name="w0")
    worker.start()
    try:
        # success path
        t = store.submit(TaskSpec(kind="noop", input={"payload": 42}))
        worker.enqueue(t.id)
        for _ in range(50):
            time.sleep(0.05)
            if store.get(t.id).status.terminal:
                break
        assert store.get(t.id).status == TaskStatus.SUCCEEDED
        assert store.get(t.id).result == 42

        # failure path
        t = store.submit(TaskSpec(kind="boom", input={}))
        worker.enqueue(t.id)
        for _ in range(50):
            time.sleep(0.05)
            if store.get(t.id).status.terminal:
                break
        assert store.get(t.id).status == TaskStatus.FAILED
        assert "expected" in (store.get(t.id).error or "")

        # unknown kind
        t = store.submit(TaskSpec(kind="nope", input={}))
        worker.enqueue(t.id)
        for _ in range(50):
            time.sleep(0.05)
            if store.get(t.id).status.terminal:
                break
        assert store.get(t.id).status == TaskStatus.FAILED
    finally:
        worker.stop(drain=True)


# ---------- graph executor ----------

def _graph_runtime():
    store = TaskStore()
    bus = EventBus()
    # Custom registry: noop + an "uppercase" task that reads input["text"].
    def uppercase(ctx, task):
        return {"result": {"text": str(task.spec.input.get("text", "")).upper()}}
    registry = {
        "noop": lambda ctx, t: {"result": t.spec.input.get("result", "ok")},
        "uppercase": uppercase,
        "wrap": lambda ctx, t: {"result": {"text": f"[{t.spec.input['text']}]"}},
    }
    workers = [Worker(store, bus, registry, name=f"w{i}") for i in range(2)]
    for w in workers:
        w.start()
    executor = GraphExecutor(store, bus, workers)
    return store, bus, workers, executor


def test_graph_executes_linear_chain_with_substitution():
    store, bus, workers, ex = _graph_runtime()
    try:
        graph = GraphSpec(name="t", nodes=[
            NodeSpec(id="a", kind="uppercase", input={"text": "hello"}),
            NodeSpec(id="b", kind="wrap", input={"text": "${a.text}"},
                     depends_on=["a"]),
        ])
        result = ex.run(graph, timeout=10.0)
        assert result.status == "succeeded"
        assert result.outputs["a"]["text"] == "HELLO"
        assert result.outputs["b"]["text"] == "[HELLO]"
    finally:
        for w in workers:
            w.stop(drain=True)


def test_graph_parallelizes_independent_nodes():
    store, bus, workers, ex = _graph_runtime()
    try:
        graph = GraphSpec(name="t", nodes=[
            NodeSpec(id="a", kind="uppercase", input={"text": "one"}),
            NodeSpec(id="b", kind="uppercase", input={"text": "two"}),
            NodeSpec(id="c", kind="wrap",
                     input={"text": "${a.text}+${b.text}"},
                     depends_on=["a", "b"]),
        ])
        result = ex.run(graph, timeout=10.0)
        assert result.status == "succeeded"
        assert result.outputs["c"]["text"] == "[ONE+TWO]"
    finally:
        for w in workers:
            w.stop(drain=True)


def test_graph_fail_graph_policy_aborts_downstream():
    store, bus, workers, ex = _graph_runtime()
    try:
        graph = GraphSpec(name="t", nodes=[
            NodeSpec(id="a", kind="nope_kind", input={}),
            NodeSpec(id="b", kind="uppercase", input={"text": "x"},
                     depends_on=["a"]),
        ])
        result = ex.run(graph, timeout=10.0)
        assert result.status == "failed"
        assert "a" in result.errors
    finally:
        for w in workers:
            w.stop(drain=True)


def test_graph_skip_policy_continues_past_failure():
    store, bus, workers, ex = _graph_runtime()
    try:
        graph = GraphSpec(name="t", nodes=[
            NodeSpec(id="a", kind="nope_kind", input={}, on_failure="skip"),
            NodeSpec(id="b", kind="uppercase", input={"text": "x"}),
        ])
        result = ex.run(graph, timeout=10.0)
        assert result.status == "succeeded"
        assert result.outputs["b"]["text"] == "X"
        assert "skipped" in result.outputs["a"]
    finally:
        for w in workers:
            w.stop(drain=True)


# ---------- HTTP server ----------

def test_http_server_capabilities_and_task_submission():
    # Use a non-default port to avoid clashes with a running runtime.
    from agi.runtime.server import Runtime, serve
    runtime = Runtime(num_workers=1)
    # Override the chat handler registry with a deterministic noop so we don't
    # need ANTHROPIC_API_KEY for HTTP-layer tests.
    for w in runtime.workers:
        w.registry["chat"] = lambda ctx, t: {"result": {"text": "stub"}}
    server, _ = serve(runtime, host="127.0.0.1", port=0)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(f"{base}/capabilities", timeout=5) as r:
            caps = json.loads(r.read())
        assert caps["name"] == RUNTIME_CAPABILITIES["name"]
        assert "graph.submit" in [m["name"] for m in caps["methods"]]

        # Submit a noop task via HTTP.
        body = json.dumps({"kind": "noop", "input": {"result": "hello"}}).encode()
        req = urllib.request.Request(f"{base}/tasks", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            sub = json.loads(r.read())
        tid = sub["task_id"]
        # Poll until terminal.
        for _ in range(50):
            time.sleep(0.05)
            with urllib.request.urlopen(f"{base}/tasks/{tid}", timeout=5) as r:
                t = json.loads(r.read())
            if t["status"] in ("succeeded", "failed", "cancelled"):
                break
        assert t["status"] == "succeeded"
        assert t["result"] == "hello"

        # Submit a small graph via HTTP.
        graph_body = json.dumps({
            "name": "g",
            "nodes": [
                {"id": "x", "kind": "noop", "input": {"result": 1}, "depends_on": []},
                {"id": "y", "kind": "noop", "input": {"result": 2}, "depends_on": ["x"]},
            ],
        }).encode()
        req = urllib.request.Request(f"{base}/graphs", data=graph_body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            gid = json.loads(r.read())["graph_id"]
        # Poll via /graphs/{id}.
        for _ in range(50):
            time.sleep(0.05)
            with urllib.request.urlopen(f"{base}/graphs/{gid}", timeout=5) as r:
                latest = json.loads(r.read())
            if latest["latest"]["kind"] in ("graph.completed", "graph.failed"):
                break
        assert latest["latest"]["kind"] == "graph.completed"
    finally:
        server.shutdown()
        runtime.shutdown()
