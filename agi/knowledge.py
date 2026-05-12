"""KnowledgeGraph — typed entities, relations, and facts.

`Memory` is keyword search over free-text notes. `WorldModel` is a per-entity
log of (read/write/fetch) interactions. Neither lets a coordinator answer
questions like "what files does project foo depend on?" or "which URLs has
the agent successfully fetched for entity X?"

`KnowledgeGraph` is the third leg of long-term memory: a typed graph of
nodes (entities) and edges (relations) with append-only facts attached to
each. It's intentionally minimal — not a full RDF store, not a vector
index — but enough that a coordinator can:

  - **Ground** a prompt: pull every fact about an entity the user mentioned
    so the agent doesn't have to rediscover it.
  - **Retrieve neighborhoods**: walk relations N hops out from a seed node
    to build a prompt context with related work.
  - **Pose graph queries**: "every file written in the last hour", "every
    URL that succeeded for entity X", "the shortest path from A to B".
  - **Stream-update**: events from the bus auto-ingest (file read/write,
    URL fetch, subagent completion, skill loaded) so the graph grows
    organically as the runtime runs.

Storage is JSONL append-only on disk, indexed in memory on load. Edges
carry a kind ("depends_on", "wrote", "fetched", "child_of", "tagged"...)
and free-form metadata. Facts are timestamped strings.

Investors care because this is the difference between "the agent
remembers raw text" and "the agent has a graph of the world it's
operated on, queryable in O(1) per hop". A coordination engine that
wants to *plan over real state* needs this layer.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from agi.events import (
    CHAT_COMPLETED,
    SKILL_LOADED,
    SUBAGENT_COMPLETED,
    TOOL_RESULT,
    Event,
    EventBus,
)


@dataclass
class Node:
    """An entity in the knowledge graph.

    `key` is the unique identifier within `kind` (e.g., kind="file", key="/etc/foo").
    `attrs` is a free-form dict for typed attributes.
    """
    kind: str
    key: str
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.key}"


@dataclass
class Edge:
    """A directed, typed relation between two nodes."""
    src: str            # source node id ("kind:key")
    dst: str            # destination node id
    rel: str            # relation kind, e.g. "depends_on", "wrote", "fetched"
    ts: float = field(default_factory=time.time)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """A timestamped assertion about a node."""
    node_id: str
    text: str
    ts: float = field(default_factory=time.time)
    source: str = "user"   # "user" | "observation" | "reflection" | "subagent"
    confidence: float = 1.0


@dataclass
class GraphQuery:
    """A neighborhood query result."""
    seed: Node
    nodes: list[Node]
    edges: list[Edge]
    facts: list[Fact]


class KnowledgeGraph:
    """Append-only typed graph with neighborhood + path retrieval.

    Thread-safe. All writes hit disk before returning (cheap because they're
    line appends). On load, the in-memory index is rebuilt from disk in one
    sequential pass.

    The graph is *namespaced*: pass `namespace="tenant-a"` to scope a view
    to one tenant. A coordinator running multi-tenant federates over many
    namespaces by using one KG file per tenant or by filtering on read.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        self.path = Path(path) if path else Path.home() / ".agi" / "knowledge.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.namespace = namespace
        self._lock = threading.Lock()
        self._nodes: dict[str, Node] = {}
        self._edges_out: dict[str, list[Edge]] = {}
        self._edges_in: dict[str, list[Edge]] = {}
        self._facts: dict[str, list[Fact]] = {}
        self._all_edges: list[Edge] = []
        self._load()

    # --- I/O ---------------------------------------------------------

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("type")
                if t == "node":
                    n = Node(
                        kind=d["kind"],
                        key=d["key"],
                        created_ts=d.get("created_ts", time.time()),
                        updated_ts=d.get("updated_ts", time.time()),
                        attrs=d.get("attrs", {}),
                    )
                    if self.namespace and n.attrs.get("namespace") != self.namespace:
                        continue
                    self._nodes[n.id] = n
                elif t == "edge":
                    e = Edge(
                        src=d["src"],
                        dst=d["dst"],
                        rel=d["rel"],
                        ts=d.get("ts", time.time()),
                        attrs=d.get("attrs", {}),
                    )
                    if self.namespace and e.attrs.get("namespace") != self.namespace:
                        continue
                    self._index_edge(e)
                elif t == "fact":
                    fact = Fact(
                        node_id=d["node_id"],
                        text=d["text"],
                        ts=d.get("ts", time.time()),
                        source=d.get("source", "user"),
                        confidence=d.get("confidence", 1.0),
                    )
                    self._facts.setdefault(fact.node_id, []).append(fact)

    def _append(self, record: dict[str, Any]) -> None:
        if self.namespace:
            attrs = record.get("attrs")
            if isinstance(attrs, dict):
                attrs.setdefault("namespace", self.namespace)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _index_edge(self, e: Edge) -> None:
        self._edges_out.setdefault(e.src, []).append(e)
        self._edges_in.setdefault(e.dst, []).append(e)
        self._all_edges.append(e)

    # --- writes ------------------------------------------------------

    def upsert_node(
        self,
        kind: str,
        key: str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> Node:
        nid = f"{kind}:{key}"
        with self._lock:
            existing = self._nodes.get(nid)
            now = time.time()
            if existing is None:
                node = Node(kind=kind, key=key, attrs=dict(attrs or {}))
                if self.namespace:
                    node.attrs.setdefault("namespace", self.namespace)
                self._nodes[nid] = node
                self._append({"type": "node", **asdict(node)})
                return node
            else:
                if attrs:
                    existing.attrs.update(attrs)
                existing.updated_ts = now
                self._append({"type": "node", **asdict(existing)})
                return existing

    def add_edge(
        self,
        src: str | Node,
        dst: str | Node,
        rel: str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> Edge:
        s = src.id if isinstance(src, Node) else src
        d = dst.id if isinstance(dst, Node) else dst
        with self._lock:
            if s not in self._nodes:
                raise KeyError(f"unknown source node: {s}")
            if d not in self._nodes:
                raise KeyError(f"unknown destination node: {d}")
            e = Edge(src=s, dst=d, rel=rel, attrs=dict(attrs or {}))
            if self.namespace:
                e.attrs.setdefault("namespace", self.namespace)
            self._index_edge(e)
            self._append({"type": "edge", **asdict(e)})
            return e

    def add_fact(
        self,
        node: str | Node,
        text: str,
        *,
        source: str = "user",
        confidence: float = 1.0,
    ) -> Fact:
        nid = node.id if isinstance(node, Node) else node
        with self._lock:
            if nid not in self._nodes:
                raise KeyError(f"unknown node: {nid}")
            fact = Fact(node_id=nid, text=text, source=source, confidence=confidence)
            self._facts.setdefault(nid, []).append(fact)
            self._append({"type": "fact", **asdict(fact)})
            return fact

    # --- reads -------------------------------------------------------

    def get_node(self, kind: str, key: str) -> Node | None:
        return self._nodes.get(f"{kind}:{key}")

    def nodes(self, kind: str | None = None) -> list[Node]:
        if kind is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.kind == kind]

    def edges_from(self, node: str | Node, rel: str | None = None) -> list[Edge]:
        nid = node.id if isinstance(node, Node) else node
        es = self._edges_out.get(nid, [])
        return [e for e in es if rel is None or e.rel == rel]

    def edges_to(self, node: str | Node, rel: str | None = None) -> list[Edge]:
        nid = node.id if isinstance(node, Node) else node
        es = self._edges_in.get(nid, [])
        return [e for e in es if rel is None or e.rel == rel]

    def facts(self, node: str | Node) -> list[Fact]:
        nid = node.id if isinstance(node, Node) else node
        return list(self._facts.get(nid, []))

    def neighborhood(
        self,
        node: str | Node,
        *,
        hops: int = 1,
        rel: str | None = None,
        max_nodes: int = 64,
    ) -> GraphQuery:
        """BFS out from a seed node, up to `hops` away."""
        nid = node.id if isinstance(node, Node) else node
        seed = self._nodes.get(nid)
        if seed is None:
            raise KeyError(f"unknown node: {nid}")
        visited: set[str] = {nid}
        nodes_out: list[Node] = [seed]
        edges_out: list[Edge] = []
        frontier: deque[tuple[str, int]] = deque([(nid, 0)])
        while frontier and len(nodes_out) < max_nodes:
            cur, depth = frontier.popleft()
            if depth >= hops:
                continue
            for e in self._edges_out.get(cur, []) + self._edges_in.get(cur, []):
                if rel is not None and e.rel != rel:
                    continue
                edges_out.append(e)
                for endpoint in (e.src, e.dst):
                    if endpoint not in visited and endpoint in self._nodes:
                        visited.add(endpoint)
                        nodes_out.append(self._nodes[endpoint])
                        frontier.append((endpoint, depth + 1))
                        if len(nodes_out) >= max_nodes:
                            break
        facts_out: list[Fact] = []
        for n in nodes_out:
            facts_out.extend(self._facts.get(n.id, []))
        return GraphQuery(seed=seed, nodes=nodes_out, edges=edges_out, facts=facts_out)

    def shortest_path(
        self,
        src: str | Node,
        dst: str | Node,
        *,
        rel: str | None = None,
        max_depth: int = 6,
    ) -> list[Edge] | None:
        """BFS shortest path; returns None if unreachable within max_depth."""
        s = src.id if isinstance(src, Node) else src
        d = dst.id if isinstance(dst, Node) else dst
        if s == d:
            return []
        prev: dict[str, tuple[str, Edge]] = {}
        visited: set[str] = {s}
        frontier: deque[tuple[str, int]] = deque([(s, 0)])
        while frontier:
            cur, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for e in self._edges_out.get(cur, []):
                if rel is not None and e.rel != rel:
                    continue
                if e.dst in visited:
                    continue
                visited.add(e.dst)
                prev[e.dst] = (cur, e)
                if e.dst == d:
                    path: list[Edge] = []
                    cursor = d
                    while cursor != s:
                        p, edge = prev[cursor]
                        path.append(edge)
                        cursor = p
                    path.reverse()
                    return path
                frontier.append((e.dst, depth + 1))
        return None

    def query_text(self, q: str, *, limit: int = 16) -> list[Node]:
        """Trivial substring match over node keys + facts. Embeddings would
        slot in behind this method."""
        q_low = q.lower()
        hits: list[tuple[float, Node]] = []
        for n in self._nodes.values():
            score = 0.0
            if q_low in n.key.lower():
                score += 2.0
            if q_low in n.kind.lower():
                score += 0.5
            for f in self._facts.get(n.id, []):
                if q_low in f.text.lower():
                    score += 1.0
            if score > 0:
                hits.append((score, n))
        hits.sort(key=lambda t: t[0], reverse=True)
        return [n for _, n in hits[:limit]]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        by_rel: dict[str, int] = {}
        for n in self._nodes.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        for e in self._all_edges:
            by_rel[e.rel] = by_rel.get(e.rel, 0) + 1
        return {
            "nodes": len(self._nodes),
            "edges": len(self._all_edges),
            "facts": sum(len(v) for v in self._facts.values()),
            "by_kind": by_kind,
            "by_rel": by_rel,
        }

    def context_for(
        self,
        kind: str,
        key: str,
        *,
        hops: int = 1,
        max_chars: int = 1600,
    ) -> str:
        """Format a neighborhood as a compact prompt context block.

        Coordinators inject this into a session prompt to ground the agent
        on what's already known about an entity.
        """
        node = self.get_node(kind, key)
        if node is None:
            return ""
        q = self.neighborhood(node, hops=hops)
        lines = [f"# Known about {node.id}"]
        for f in q.facts[:10]:
            lines.append(f"- {f.text}  (src={f.source}, conf={f.confidence:.2f})")
        rel_buckets: dict[str, list[str]] = {}
        for e in q.edges:
            other = e.dst if e.src == node.id else e.src
            rel_buckets.setdefault(e.rel, []).append(other)
        for rel, others in rel_buckets.items():
            uniq = list(dict.fromkeys(others))[:8]
            lines.append(f"- {rel}: {', '.join(uniq)}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[: max_chars - 1] + "…"
        return out


# --- Event ingestion --------------------------------------------------


def attach_to_bus(kg: KnowledgeGraph, bus: EventBus) -> int:
    """Subscribe to an EventBus and grow the graph from real activity.

    Returns the subscription id so callers can unsubscribe.

    Ingests:
      - `tool.result`: file read/write → file node + edge from session
      - `subagent.completed`: role → child-of edge from parent
      - `skill.loaded`: skill node + "used" edge from session
      - `chat.completed`: session node updated

    Failures during ingest are swallowed — a buggy ingestor must never
    block the agent.
    """

    def handle(ev: Event) -> None:
        try:
            sid = ev.session_id
            if sid:
                kg.upsert_node("session", sid)
            if ev.kind == TOOL_RESULT:
                tool = ev.data.get("tool")
                args = ev.data.get("args") or {}
                if tool in ("read_file", "write_file") and isinstance(args, dict):
                    path = args.get("path")
                    if path:
                        kg.upsert_node("file", str(path))
                        if sid:
                            kg.add_edge(
                                f"session:{sid}",
                                f"file:{path}",
                                rel="wrote" if tool == "write_file" else "read",
                            )
                elif tool in ("web_fetch_20260209", "web_fetch") and isinstance(args, dict):
                    url = args.get("url")
                    if url:
                        kg.upsert_node("url", str(url))
                        if sid:
                            kg.add_edge(f"session:{sid}", f"url:{url}", rel="fetched")
            elif ev.kind == SUBAGENT_COMPLETED:
                child_id = ev.data.get("child_id")
                role = ev.data.get("role")
                if child_id:
                    kg.upsert_node("session", child_id, attrs={"role": role})
                    if sid:
                        kg.add_edge(f"session:{sid}", f"session:{child_id}", rel="spawned")
            elif ev.kind == SKILL_LOADED:
                name = ev.data.get("name")
                if name:
                    kg.upsert_node("skill", name)
                    if sid:
                        kg.add_edge(f"session:{sid}", f"skill:{name}", rel="used")
            elif ev.kind == CHAT_COMPLETED:
                if sid:
                    score = ev.data.get("critic_score")
                    if score is not None:
                        kg.add_fact(
                            f"session:{sid}",
                            f"turn completed with critic_score={score}",
                            source="observation",
                        )
        except Exception:
            return

    return bus.subscribe(handle)
