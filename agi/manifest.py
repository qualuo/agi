r"""Manifest — discoverable runtime-primitive catalog for the coordination engine.

The runtime ships ~100 algorithmic primitives.  A *coordination engine*
that drives the runtime needs a single machine-readable answer to the
question *"what can you do?"*: not a free-text README, not a flat list
of Python imports, but a structured catalog it can filter, rank, and
dispatch against.

``Manifest`` is that catalog.  Every primitive registers a
:class:`PrimitiveSpec` describing what it is, what it ingests, what it
emits, what category of problem it solves, what other primitives it
composes with, and what its complexity / determinism / certificate
guarantees are.  The catalog is pure data (JSON-encodable), purely
introspective (no side effects on import), and built from a single
source of truth.  A coordinator queries the catalog with predicates::

    m = default_manifest()

    # "Give me everything that can perform discrete optimisation
    #  with a PAC certificate and zero NumPy dependency."
    for spec in m.find(
        tag=TAG_DISCRETE_OPT,
        certificate=CERT_PAC,
        dependencies_max="stdlib",
    ):
        ...

    # Or by intent — natural-language → token overlap → ranked specs:
    for spec, score in m.recommend("plan a TSP tour with a confidence interval"):
        ...

    # Or just JSON it out and ship to a remote orchestrator:
    print(m.to_json(pretty=True))


Design contract
---------------

* **Single source of truth.**  ``_PRIMITIVE_TABLE`` at the bottom of
  this module is the canonical list.  Adding a primitive means adding
  one row.  Adding metadata fields means extending :class:`PrimitiveSpec`
  and one row at a time.  No magic discovery, no docstring scraping at
  query time — discovery is at startup or on demand and is cached.

* **Pure stdlib.**  No NumPy, no Pydantic, no third-party
  serialisation.  ``json``-encodable values only.

* **Stable identifiers.**  Each spec has a ``name`` (module name) and a
  ``stable_id`` (``agi.<name>.v<schema_version>``).  External
  orchestrators key off ``stable_id`` so we can evolve metadata without
  breaking integration.

* **No import side-effects.**  Importing ``agi.manifest`` does *not*
  import every primitive module.  The catalog is metadata.  Use
  :func:`Manifest.load_module` to instantiate a primitive when a
  coordinator actually wants to call it.

* **Audit trail.**  Each lookup / query / recommend can publish an
  :class:`Event` to an injected :class:`EventBus` so a coordinator's
  decisions over the catalog are themselves auditable.

* **Versioned schema.**  ``SCHEMA_VERSION`` bumps with backwards-
  incompatible field changes; ``to_json`` embeds it; ``from_json``
  rejects mismatches unless ``allow_schema_skew=True``.


What this primitive ships
-------------------------

* :class:`PrimitiveSpec` — the per-primitive metadata record.  Every
  field is JSON-encodable.

* Constants — categorisation taxonomies kept in stable string form so
  external systems can match without importing Python:

  * ``KIND_*``         — high-level role (algorithm, infrastructure,
                         coordination, observability, safety).
  * ``TAG_*``          — fine-grained capability tags (discrete-opt,
                         bayesian-inference, game-theory, …).
  * ``CERT_*``         — guarantee class (none, pac, anytime, exact,
                         empirical).
  * ``DETERMINISM_*``  — given-seed reproducibility level.
  * ``DEPENDENCY_*``   — heaviest runtime dependency (stdlib, numpy,
                         torch, llm, network).

* :class:`Manifest` — the queryable catalog.  Supports
  :meth:`Manifest.list`, :meth:`Manifest.lookup`, :meth:`Manifest.find`
  (predicate filter), :meth:`Manifest.recommend` (intent-ranked),
  :meth:`Manifest.tags`, :meth:`Manifest.kinds`,
  :meth:`Manifest.depends_graph`, :meth:`Manifest.to_json`,
  :meth:`Manifest.from_json`, :meth:`Manifest.diff` (catalog
  comparison across versions).

* :func:`default_manifest` — the canonical pre-populated catalog for
  the current runtime.  Built lazily and cached.

* :func:`auto_discover` — best-effort fallback that walks the ``agi``
  package and synthesises minimal specs for any module not in the
  curated table (useful during development; CI asserts the curated
  table is exhaustive).

* :class:`PrimitiveLoader` — on-demand module import so a coordinator
  can hold a ``PrimitiveSpec`` cheaply and only instantiate the
  underlying primitive when it dispatches work to it.


Composes with
-------------

* :mod:`agi.coordinator` / :mod:`agi.driver` — the canonical drivers
  consult the manifest to choose which primitive to invoke for a
  ``Goal``.  Recommend ranking feeds the planner.

* :mod:`agi.strategist` — meta-decision API uses ``CERT_*`` and
  ``DEPENDENCY_*`` as inputs to risk / cost tradeoffs.

* :mod:`agi.capabilities` — observed-performance routing updates the
  *recommendation* prior with empirical success rates per primitive.

* :mod:`agi.server` / :mod:`agi.mcp` / :mod:`agi.protocol` — the
  network surfaces export ``Manifest.to_json()`` so an external
  orchestrator gets the catalog by HTTP/SSE, MCP, or stdio JSON-RPC
  with no language coupling.

* :mod:`agi.attest` — every catalog snapshot can be fingerprinted into
  the audit ledger so coordinator decisions can be replayed against
  the exact catalog they saw.


Mathematical notation
---------------------

None — this is pure metadata.  All scoring inside :meth:`recommend` is
elementary token-overlap (Jaccard / IDF-weighted Jaccard), no learned
embeddings, no LLM calls.  External orchestrators that want ML-driven
recommendation should consume ``Manifest`` as a feature store and
train their own ranker.

References
----------

* Wirth, *Algorithms + Data Structures = Programs.*  Prentice-Hall
  1976 — catalog as first-class engineering artifact.
* Parnas, *On the Criteria To Be Used in Decomposing Systems into
  Modules.*  CACM 15(12), 1972 — information hiding as the basis of
  module-level metadata.
* OpenAPI / JSON Schema — design influence for the predicate-based
  query and the JSON-encodable spec format.
* Model Context Protocol (Anthropic 2024) — design influence for the
  tool-discovery surface that an external coordinator binds against.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "PrimitiveSpec",
    "Manifest",
    "ManifestError",
    "PrimitiveLoader",
    # Kinds
    "KIND_INFRA",
    "KIND_COORDINATION",
    "KIND_INFERENCE",
    "KIND_OPTIMIZATION",
    "KIND_DECISION",
    "KIND_GAMES",
    "KIND_SAFETY",
    "KIND_OBSERVABILITY",
    "KIND_LEARNING",
    "KIND_SCIENCE",
    "KIND_MEMORY",
    "KIND_ECONOMICS",
    "KNOWN_KINDS",
    # Tags
    "TAG_BAYESIAN",
    "TAG_DISCRETE_OPT",
    "TAG_CONTINUOUS_OPT",
    "TAG_GAME_THEORY",
    "TAG_RL",
    "TAG_PAC",
    "TAG_ANYTIME",
    "TAG_REPLAY",
    "TAG_STREAMING",
    "TAG_MULTI_AGENT",
    "TAG_LLM",
    "TAG_NUMERICAL",
    "TAG_SYMBOLIC",
    "TAG_CAUSAL",
    "TAG_CALIBRATION",
    "TAG_SAFETY",
    "TAG_NETWORK",
    "TAG_PLANNING",
    "TAG_RETRIEVAL",
    "TAG_GENERATIVE",
    "TAG_MULTI_OBJECTIVE",
    "TAG_INTROSPECTION",
    "TAG_ADAPTIVE",
    # Certificates
    "CERT_NONE",
    "CERT_EMPIRICAL",
    "CERT_PAC",
    "CERT_ANYTIME",
    "CERT_EXACT",
    "CERT_DP",
    "KNOWN_CERTIFICATES",
    # Determinism
    "DETERMINISM_NONE",
    "DETERMINISM_SEEDED",
    "DETERMINISM_PURE",
    "KNOWN_DETERMINISMS",
    # Dependencies
    "DEP_STDLIB",
    "DEP_NUMPY",
    "DEP_TORCH",
    "DEP_LLM",
    "DEP_NETWORK",
    "KNOWN_DEPENDENCIES",
    # Events
    "MANIFEST_BUILT",
    "MANIFEST_QUERIED",
    "MANIFEST_LOOKUP",
    "MANIFEST_RECOMMENDED",
    "MANIFEST_EXPORTED",
    "MANIFEST_DIFFED",
    # Factories
    "default_manifest",
    "auto_discover",
    "reset_default_manifest",
]


SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Taxonomy: high-level kinds.  Kept short and stable; orchestrators key off
# these strings.

KIND_INFRA = "infrastructure"
KIND_COORDINATION = "coordination"
KIND_INFERENCE = "inference"
KIND_OPTIMIZATION = "optimization"
KIND_DECISION = "decision"
KIND_GAMES = "games"
KIND_SAFETY = "safety"
KIND_OBSERVABILITY = "observability"
KIND_LEARNING = "learning"
KIND_SCIENCE = "science"
KIND_MEMORY = "memory"
KIND_ECONOMICS = "economics"

KNOWN_KINDS = frozenset({
    KIND_INFRA, KIND_COORDINATION, KIND_INFERENCE, KIND_OPTIMIZATION,
    KIND_DECISION, KIND_GAMES, KIND_SAFETY, KIND_OBSERVABILITY,
    KIND_LEARNING, KIND_SCIENCE, KIND_MEMORY, KIND_ECONOMICS,
})

# Fine-grained capability tags.  A primitive carries many.

TAG_BAYESIAN = "bayesian"
TAG_DISCRETE_OPT = "discrete-optimization"
TAG_CONTINUOUS_OPT = "continuous-optimization"
TAG_GAME_THEORY = "game-theory"
TAG_RL = "reinforcement-learning"
TAG_PAC = "pac-bound"
TAG_ANYTIME = "anytime-valid"
TAG_REPLAY = "replay-verifiable"
TAG_STREAMING = "streaming"
TAG_MULTI_AGENT = "multi-agent"
TAG_LLM = "llm-driven"
TAG_NUMERICAL = "numerical"
TAG_SYMBOLIC = "symbolic"
TAG_CAUSAL = "causal-inference"
TAG_CALIBRATION = "calibration"
TAG_SAFETY = "safety"
TAG_NETWORK = "network-surface"
TAG_PLANNING = "planning"
TAG_RETRIEVAL = "retrieval"
TAG_GENERATIVE = "generative"
TAG_MULTI_OBJECTIVE = "multi-objective"
TAG_INTROSPECTION = "introspection"
TAG_ADAPTIVE = "adaptive"

# Guarantee class — what kind of correctness story does the primitive ship?
CERT_NONE = "none"
CERT_EMPIRICAL = "empirical"  # benchmarks pass but no formal guarantee
CERT_PAC = "pac"              # PAC / Hoeffding / Bernstein bound
CERT_ANYTIME = "anytime"      # anytime-valid (e-process, mixture mart.)
CERT_EXACT = "exact"          # provably correct (LCF kernel, CDCL UNSAT proof)
CERT_DP = "differential-privacy"

KNOWN_CERTIFICATES = frozenset({
    CERT_NONE, CERT_EMPIRICAL, CERT_PAC, CERT_ANYTIME, CERT_EXACT, CERT_DP,
})

# Given-seed reproducibility level.
DETERMINISM_NONE = "none"        # observable nondeterminism (wall clock, threads)
DETERMINISM_SEEDED = "seeded"    # deterministic given an explicit seed
DETERMINISM_PURE = "pure"        # deterministic full stop (no RNG, no clocks)

KNOWN_DETERMINISMS = frozenset({
    DETERMINISM_NONE, DETERMINISM_SEEDED, DETERMINISM_PURE,
})

# Heaviest runtime dependency.  Ordered roughly by weight; a coordinator
# can ask for "the lightest primitive that solves X" and rank by this.
DEP_STDLIB = "stdlib"
DEP_NUMPY = "numpy"
DEP_TORCH = "torch"
DEP_LLM = "llm"          # requires an Anthropic / OpenAI API client
DEP_NETWORK = "network"  # requires outbound HTTP

KNOWN_DEPENDENCIES = frozenset({
    DEP_STDLIB, DEP_NUMPY, DEP_TORCH, DEP_LLM, DEP_NETWORK,
})

_DEP_WEIGHT = {
    DEP_STDLIB: 0,
    DEP_NUMPY: 1,
    DEP_TORCH: 2,
    DEP_NETWORK: 3,
    DEP_LLM: 4,
}

# Events emitted onto an EventBus when one is injected.
MANIFEST_BUILT = "manifest.built"
MANIFEST_QUERIED = "manifest.queried"
MANIFEST_LOOKUP = "manifest.lookup"
MANIFEST_RECOMMENDED = "manifest.recommended"
MANIFEST_EXPORTED = "manifest.exported"
MANIFEST_DIFFED = "manifest.diffed"


class ManifestError(ValueError):
    """Raised on invalid spec / query / schema-version mismatch."""


# ---------------------------------------------------------------------------
# Spec

@dataclass(frozen=True)
class PrimitiveSpec:
    """One row of the catalog.  Frozen / hashable / JSON-encodable.

    Fields
    ------

    name
        The Python module name (without the ``agi.`` prefix) and the
        catalog key.  Stable.

    kind
        One of :data:`KNOWN_KINDS`.  High-level role.

    summary
        One-line description, present tense, ends with a period.

    tags
        Iterable of :data:`KNOWN_TAGS` (see ``TAG_*``).  Order-irrelevant;
        stored sorted.

    inputs
        Short prose list of the typed-ish inputs the primitive ingests
        (``["observations: Iterable[float]", "prior: Distribution"]``).
        Free-form — the manifest is documentation, not type-checking.

    outputs
        Same for outputs.

    certificate
        One of :data:`KNOWN_CERTIFICATES`.

    determinism
        One of :data:`KNOWN_DETERMINISMS`.

    dependency
        One of :data:`KNOWN_DEPENDENCIES` — the heaviest runtime
        dependency the primitive imports.

    composes_with
        Sister-primitive names that this one is designed to feed into
        or consume from.

    references
        Bibliographic citations (free-form strings).

    demo_path
        Optional path (relative to repo root) to an executable demo.

    events_emitted
        Topics the primitive publishes onto the runtime EventBus.

    complexity
        Asymptotic complexity descriptor (free-form string).

    notes
        Anything else a coordinator should know but that doesn't fit
        elsewhere.

    stable_id
        ``"agi.<name>.v<major>"`` — orchestrators key off this to
        survive metadata schema drift.
    """
    name: str
    kind: str
    summary: str
    tags: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    certificate: str = CERT_NONE
    determinism: str = DETERMINISM_SEEDED
    dependency: str = DEP_STDLIB
    composes_with: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    demo_path: str | None = None
    events_emitted: tuple[str, ...] = ()
    complexity: str = ""
    notes: str = ""
    stable_id: str = ""

    def __post_init__(self) -> None:  # pragma: no cover - frozen dataclass
        # Frozen dataclass: __setattr__ is blocked.  Use object.__setattr__
        # to backfill stable_id and sort tag/input/output for hashability.
        if not self.name:
            raise ManifestError("PrimitiveSpec.name is required")
        if self.kind not in KNOWN_KINDS:
            raise ManifestError(
                f"PrimitiveSpec.kind {self.kind!r} not in KNOWN_KINDS={sorted(KNOWN_KINDS)}"
            )
        if self.certificate not in KNOWN_CERTIFICATES:
            raise ManifestError(
                f"PrimitiveSpec.certificate {self.certificate!r} not in {sorted(KNOWN_CERTIFICATES)}"
            )
        if self.determinism not in KNOWN_DETERMINISMS:
            raise ManifestError(
                f"PrimitiveSpec.determinism {self.determinism!r} not in {sorted(KNOWN_DETERMINISMS)}"
            )
        if self.dependency not in KNOWN_DEPENDENCIES:
            raise ManifestError(
                f"PrimitiveSpec.dependency {self.dependency!r} not in {sorted(KNOWN_DEPENDENCIES)}"
            )
        if not self.summary or not self.summary.endswith("."):
            raise ManifestError(
                f"PrimitiveSpec.summary must be a non-empty sentence ending with '.': {self.summary!r}"
            )
        # Canonicalise tuple ordering for deterministic hashing / JSON.
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        object.__setattr__(self, "composes_with", tuple(sorted(set(self.composes_with))))
        object.__setattr__(self, "events_emitted", tuple(sorted(set(self.events_emitted))))
        if not self.stable_id:
            major = SCHEMA_VERSION.split(".", 1)[0]
            object.__setattr__(self, "stable_id", f"agi.{self.name}.v{major}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Tuples become lists for JSON.
        for k in ("tags", "inputs", "outputs", "composes_with", "references", "events_emitted"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PrimitiveSpec":
        return cls(
            name=d["name"],
            kind=d["kind"],
            summary=d["summary"],
            tags=tuple(d.get("tags", ())),
            inputs=tuple(d.get("inputs", ())),
            outputs=tuple(d.get("outputs", ())),
            certificate=d.get("certificate", CERT_NONE),
            determinism=d.get("determinism", DETERMINISM_SEEDED),
            dependency=d.get("dependency", DEP_STDLIB),
            composes_with=tuple(d.get("composes_with", ())),
            references=tuple(d.get("references", ())),
            demo_path=d.get("demo_path"),
            events_emitted=tuple(d.get("events_emitted", ())),
            complexity=d.get("complexity", ""),
            notes=d.get("notes", ""),
            stable_id=d.get("stable_id", ""),
        )


# ---------------------------------------------------------------------------
# Manifest

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
_STOP = frozenset({
    "the", "and", "for", "with", "into", "from", "that", "this", "have",
    "are", "was", "but", "you", "your", "give", "what", "when", "where",
    "how", "why", "primitive", "runtime", "engine", "system", "module",
    "model", "models", "task", "tasks", "use", "using", "case",
})


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 3 and t.lower() not in _STOP]


def _spec_corpus(spec: PrimitiveSpec) -> list[str]:
    parts = [spec.name, spec.summary, spec.kind, spec.notes]
    parts.extend(spec.tags)
    parts.extend(spec.inputs)
    parts.extend(spec.outputs)
    return _tokens(" ".join(parts))


def _score_idf(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    inter = qset & dset
    if not inter:
        return 0.0
    return sum(idf.get(t, 1.0) for t in inter) / (1.0 + math_log1p(len(dset)))


def math_log1p(x: float) -> float:
    # Local helper to keep one numerical dependency out of the import path.
    import math as _math
    return _math.log1p(x)


class Manifest:
    """Queryable catalog of :class:`PrimitiveSpec` records.

    Thread-safe: all mutating methods take a re-entrant lock; query
    methods take a snapshot reference and read lock-free.  This lets
    high-QPS coordinators recommend without contending against a slow
    registrar.
    """

    def __init__(
        self,
        specs: Iterable[PrimitiveSpec] = (),
        *,
        bus: Any | None = None,  # agi.events.EventBus; typed Any to avoid cycle
    ) -> None:
        self._lock = threading.RLock()
        self._by_name: dict[str, PrimitiveSpec] = {}
        self._tags: dict[str, set[str]] = {}    # tag -> {name, ...}
        self._kinds: dict[str, set[str]] = {}   # kind -> {name, ...}
        self._bus = bus
        for s in specs:
            self._add(s)
        self._maybe_publish(MANIFEST_BUILT, {"count": len(self._by_name)})

    # ----- mutating ops --------------------------------------------------

    def register(self, spec: PrimitiveSpec) -> None:
        """Insert a spec.  Replacing an existing key is allowed and
        emits a ``manifest.lookup`` event for traceability."""
        with self._lock:
            self._add(spec)

    def unregister(self, name: str) -> None:
        with self._lock:
            spec = self._by_name.pop(name, None)
            if spec is None:
                return
            for t in spec.tags:
                self._tags.get(t, set()).discard(name)
            self._kinds.get(spec.kind, set()).discard(name)

    def _add(self, spec: PrimitiveSpec) -> None:
        self._by_name[spec.name] = spec
        for t in spec.tags:
            self._tags.setdefault(t, set()).add(spec.name)
        self._kinds.setdefault(spec.kind, set()).add(spec.name)

    # ----- read-only ops -------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self):
        return iter(sorted(self._by_name.values(), key=lambda s: s.name))

    def list(self) -> list[PrimitiveSpec]:
        """Snapshot of all specs, sorted by name."""
        return sorted(self._by_name.values(), key=lambda s: s.name)

    def lookup(self, name: str) -> PrimitiveSpec:
        spec = self._by_name.get(name)
        if spec is None:
            raise ManifestError(f"no such primitive: {name!r}")
        self._maybe_publish(MANIFEST_LOOKUP, {"name": name})
        return spec

    def tags(self) -> dict[str, int]:
        """Tag → primitive-count histogram (snapshot)."""
        return {t: len(names) for t, names in self._tags.items()}

    def kinds(self) -> dict[str, int]:
        return {k: len(names) for k, names in self._kinds.items()}

    def find(
        self,
        *,
        kind: str | None = None,
        tag: str | Iterable[str] | None = None,
        certificate: str | Iterable[str] | None = None,
        determinism: str | Iterable[str] | None = None,
        dependency: str | Iterable[str] | None = None,
        dependencies_max: str | None = None,
        composes_with: str | None = None,
        name_prefix: str | None = None,
    ) -> list[PrimitiveSpec]:
        """Predicate-filtered query.  All conditions are AND-combined.

        ``dependencies_max="numpy"`` keeps every primitive whose
        heaviest dependency is at most numpy (so stdlib + numpy, no
        torch / llm / network).  Use this to surface the cheapest
        primitives in restricted runtime environments.
        """
        kind_set = _norm_set(kind)
        tag_set = _norm_set(tag)
        cert_set = _norm_set(certificate)
        det_set = _norm_set(determinism)
        dep_set = _norm_set(dependency)
        if dependencies_max is not None and dependencies_max not in _DEP_WEIGHT:
            raise ManifestError(f"unknown dependency: {dependencies_max!r}")
        max_w = _DEP_WEIGHT.get(dependencies_max) if dependencies_max else None
        out: list[PrimitiveSpec] = []
        for spec in self.list():
            if kind_set and spec.kind not in kind_set:
                continue
            if tag_set and not (tag_set & set(spec.tags)):
                continue
            if cert_set and spec.certificate not in cert_set:
                continue
            if det_set and spec.determinism not in det_set:
                continue
            if dep_set and spec.dependency not in dep_set:
                continue
            if max_w is not None and _DEP_WEIGHT.get(spec.dependency, 999) > max_w:
                continue
            if composes_with and composes_with not in spec.composes_with:
                continue
            if name_prefix and not spec.name.startswith(name_prefix):
                continue
            out.append(spec)
        self._maybe_publish(MANIFEST_QUERIED, {
            "kind": list(kind_set), "tag": list(tag_set),
            "certificate": list(cert_set), "determinism": list(det_set),
            "dependency": list(dep_set),
            "dependencies_max": dependencies_max,
            "composes_with": composes_with,
            "name_prefix": name_prefix,
            "hits": len(out),
        })
        return out

    def recommend(self, intent: str, *, k: int = 10) -> list[tuple[PrimitiveSpec, float]]:
        """IDF-weighted token-overlap ranking against an intent string.

        Pure stdlib, no LLM call.  Returns the top-``k`` ``(spec,
        score)`` pairs sorted by descending score.  Score == 0 results
        are dropped.  Order across ties is stable on ``spec.name``.

        A coordinator typically composes this with
        :class:`agi.capabilities.CapabilityRegistry` so the prior gets
        re-weighted by empirical per-primitive success.
        """
        if k <= 0:
            return []
        specs = self.list()
        # IDF over the catalog
        df: dict[str, int] = {}
        corpora: list[tuple[PrimitiveSpec, list[str]]] = []
        for spec in specs:
            toks = _spec_corpus(spec)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
            corpora.append((spec, toks))
        n = max(1, len(specs))
        idf = {t: math_log1p(n / max(1, c)) for t, c in df.items()}
        q = _tokens(intent)
        if not q:
            return []
        scored: list[tuple[PrimitiveSpec, float]] = []
        for spec, doc in corpora:
            s = _score_idf(q, doc, idf)
            if s > 0:
                scored.append((spec, s))
        scored.sort(key=lambda p: (-p[1], p[0].name))
        result = scored[:k]
        self._maybe_publish(MANIFEST_RECOMMENDED, {
            "intent": intent, "k": k,
            "names": [s.name for s, _ in result],
        })
        return result

    def depends_graph(self) -> dict[str, list[str]]:
        """Adjacency dict ``name -> [composes_with names that exist]``.

        Composes-with targets that aren't in the catalog are dropped,
        so the returned graph is closed under lookup.  Use this to
        plan a multi-primitive execution by walking the graph from a
        seed primitive.
        """
        known = set(self._by_name.keys())
        out: dict[str, list[str]] = {}
        for spec in self.list():
            out[spec.name] = sorted(c for c in spec.composes_with if c in known)
        return out

    # ----- serialisation -------------------------------------------------

    def to_json(self, *, pretty: bool = False) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_ts": time.time(),
            "count": len(self._by_name),
            "primitives": [s.to_dict() for s in self.list()],
        }
        s = json.dumps(payload, indent=2 if pretty else None, sort_keys=True)
        self._maybe_publish(MANIFEST_EXPORTED, {"bytes": len(s)})
        return s

    def fingerprint(self) -> str:
        """SHA-256 over the canonicalised JSON encoding.  Stable across
        process restarts because :meth:`to_json` sorts keys and
        primitive ordering."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "primitives": [s.to_dict() for s in self.list()],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, blob: str, *, allow_schema_skew: bool = False) -> "Manifest":
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError as e:
            raise ManifestError(f"malformed JSON: {e}") from e
        sv = payload.get("schema_version")
        if sv != SCHEMA_VERSION and not allow_schema_skew:
            raise ManifestError(
                f"schema version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}; "
                f"pass allow_schema_skew=True to override"
            )
        specs = [PrimitiveSpec.from_dict(d) for d in payload.get("primitives", [])]
        return cls(specs)

    def diff(self, other: "Manifest") -> dict[str, Any]:
        """Catalog-level diff for migration audits.

        Returns ``{"added": [...], "removed": [...], "changed": [{
        "name": ..., "field": ..., "from": ..., "to": ...}]}``.
        """
        added = sorted(set(other._by_name) - set(self._by_name))
        removed = sorted(set(self._by_name) - set(other._by_name))
        changed: list[dict[str, Any]] = []
        for name in sorted(set(self._by_name) & set(other._by_name)):
            a = self._by_name[name].to_dict()
            b = other._by_name[name].to_dict()
            for key in sorted(a.keys() | b.keys()):
                if a.get(key) != b.get(key):
                    changed.append({"name": name, "field": key,
                                    "from": a.get(key), "to": b.get(key)})
        result = {"added": added, "removed": removed, "changed": changed}
        self._maybe_publish(MANIFEST_DIFFED, {
            "added": len(added), "removed": len(removed), "changed": len(changed),
        })
        return result

    # ----- bus glue ------------------------------------------------------

    def attach_bus(self, bus: Any) -> None:
        with self._lock:
            self._bus = bus

    def detach_bus(self) -> None:
        with self._lock:
            self._bus = None

    def _maybe_publish(self, kind: str, data: dict[str, Any]) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            from agi.events import Event
            bus.publish(Event(kind=kind, data=data))
        except Exception:
            # Manifest must not break callers because their bus is broken.
            pass


def _norm_set(v: str | Iterable[str] | None) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, str):
        return {v}
    return set(v)


# ---------------------------------------------------------------------------
# Loader

class PrimitiveLoader:
    """On-demand module import for a :class:`PrimitiveSpec`.

    Holding a ``PrimitiveSpec`` is cheap (pure metadata).  Loading the
    backing module may pull heavy dependencies (torch, anthropic, …);
    do it only when a coordinator actually needs to dispatch.

    Imports are cached per process so repeated dispatch is fast.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def load(self, spec: PrimitiveSpec):
        with self._lock:
            mod = self._cache.get(spec.name)
            if mod is not None:
                return mod
            mod = importlib.import_module(f"agi.{spec.name}")
            self._cache[spec.name] = mod
            return mod

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._cache.keys())


# ---------------------------------------------------------------------------
# Default catalog
#
# Single source of truth.  Adding a primitive means appending one row.
# Style: ``spec(name=..., kind=..., summary=..., tags=(...), ...)``.

def _spec(**kw: Any) -> PrimitiveSpec:
    return PrimitiveSpec(**kw)


# Curated specs for all primitives currently in the agi/ package.  Built
# from module docstrings + on-disk inspection at authoring time.
_PRIMITIVE_TABLE: tuple[PrimitiveSpec, ...] = (
    # --- coordination & runtime infrastructure -------------------------
    _spec(name="runtime", kind=KIND_INFRA,
          summary="The Runtime is the surface a coordination engine drives sessions through.",
          tags=(TAG_NETWORK, TAG_INTROSPECTION),
          inputs=("SessionConfig", "user messages"),
          outputs=("Session", "EventBus stream", "usage / cost"),
          composes_with=("events", "memory", "skills", "toolsynth", "tasks"),
          dependency=DEP_LLM,
          determinism=DETERMINISM_NONE,
          notes="Lazy on the LLM client: Runtime() is free for tests."),
    _spec(name="coordinator", kind=KIND_COORDINATION,
          summary="Reference coordination engine that decomposes a Goal into a Plan and drives the Runtime.",
          tags=(TAG_PLANNING,),
          inputs=("Goal", "Runtime"),
          outputs=("CoordinationResult",),
          composes_with=("runtime", "tasks", "goalc", "events"),
          demo_path="examples/coordinator_demo.py"),
    _spec(name="driver", kind=KIND_COORDINATION,
          summary="The canonical entry point a coordination engine uses to drive the Runtime end-to-end.",
          tags=(TAG_PLANNING,),
          composes_with=("runtime", "coordinator", "governance", "preflight"),
          demo_path="examples/driver_multi_tenant_demo.py"),
    _spec(name="goalc", kind=KIND_COORDINATION,
          summary="Goal compiler: turn a Goal into a Plan via heuristic or LLM-based decomposition.",
          tags=(TAG_LLM, TAG_PLANNING),
          composes_with=("coordinator",),
          dependency=DEP_LLM),
    _spec(name="strategist", kind=KIND_COORDINATION,
          summary="Top-level meta-decision API for the coordination engine — risk-adjusted primitive selection.",
          tags=(TAG_PLANNING, TAG_MULTI_OBJECTIVE),
          composes_with=("policy", "capabilities", "preflight", "portfolio"),
          demo_path="examples/strategist_demo.py"),
    _spec(name="autoloop", kind=KIND_COORDINATION,
          summary="Autonomous goal loop: iterate retry-with-lessons until the goal is met or budget exhausted.",
          tags=(TAG_PLANNING, TAG_ADAPTIVE),
          composes_with=("coordinator", "reflection"),
          dependency=DEP_LLM),
    _spec(name="autonomy", kind=KIND_COORDINATION,
          summary="Continuous closed-loop self-improvement engine across the runtime's medium timescales.",
          tags=(TAG_PLANNING, TAG_ADAPTIVE, TAG_RL),
          composes_with=("evolve", "selfeval", "capabilities", "policy"),
          dependency=DEP_LLM),
    _spec(name="evolve", kind=KIND_LEARNING,
          summary="Evolution engine — closed-loop strategy improvement via population search.",
          tags=(TAG_DISCRETE_OPT, TAG_ADAPTIVE),
          composes_with=("autonomy", "selfeval"),
          dependency=DEP_LLM),
    _spec(name="fork", kind=KIND_COORDINATION,
          summary="Session fork: race N variants of a task and return the best by critic score.",
          tags=(TAG_PLANNING,),
          composes_with=("runtime",),
          dependency=DEP_LLM),
    _spec(name="pool", kind=KIND_INFRA,
          summary="RuntimePool federates over many Runtimes from one coordinator surface.",
          tags=(TAG_NETWORK,),
          composes_with=("runtime", "driver"),
          dependency=DEP_LLM),
    _spec(name="server", kind=KIND_INFRA,
          summary="HTTP+SSE server exposing the Runtime to a coordination engine over the network.",
          tags=(TAG_NETWORK, TAG_INTROSPECTION),
          composes_with=("runtime", "manifest"),
          determinism=DETERMINISM_NONE),
    _spec(name="protocol", kind=KIND_INFRA,
          summary="Coordination protocol — stdio JSON-RPC 2.0 surface for driving the Runtime as a subprocess.",
          tags=(TAG_NETWORK,),
          composes_with=("runtime",)),
    _spec(name="mcp", kind=KIND_INFRA,
          summary="Model Context Protocol adapter — drive the Runtime from Claude Desktop / Code.",
          tags=(TAG_NETWORK, TAG_LLM),
          composes_with=("runtime",),
          dependency=DEP_LLM),
    _spec(name="events", kind=KIND_INFRA,
          summary="EventBus and typed Event kinds — the coordination signal channel.",
          tags=(TAG_INTROSPECTION,),
          determinism=DETERMINISM_PURE),
    _spec(name="tasks", kind=KIND_COORDINATION,
          summary="Task scheduler — coordinator-friendly priority work queue on top of the Runtime.",
          tags=(TAG_PLANNING,),
          composes_with=("runtime", "scheduler")),
    _spec(name="scheduler", kind=KIND_COORDINATION,
          summary="Parallel DAG scheduler — the runtime's contract with a coordination engine for fan-out.",
          tags=(TAG_PLANNING,),
          composes_with=("tasks", "coordinator")),
    _spec(name="governance", kind=KIND_INFRA,
          summary="Multi-tenant budgets, quotas, rate limits, and fair-share for the Runtime.",
          tags=(TAG_SAFETY,),
          composes_with=("market", "economist", "driver")),
    _spec(name="preflight", kind=KIND_COORDINATION,
          summary="Preflight estimator — predict cost, duration, and p_success before dispatch.",
          tags=(TAG_CALIBRATION,),
          composes_with=("capabilities", "calibration", "strategist"),
          demo_path="examples/preflight_demo.py"),
    _spec(name="capabilities", kind=KIND_COORDINATION,
          summary="Observed-performance routing registry for coordinators picking SessionConfig.",
          tags=(TAG_ADAPTIVE, TAG_INTROSPECTION),
          composes_with=("policy", "strategist", "preflight")),
    _spec(name="policy", kind=KIND_DECISION,
          summary="PolicyRouter — Thompson-sampled contextual bandit on top of the CapabilityRegistry.",
          tags=(TAG_BAYESIAN, TAG_RL, TAG_ADAPTIVE),
          composes_with=("capabilities", "bandit")),
    _spec(name="contract", kind=KIND_SAFETY,
          summary="Service-level objectives (SLOs) for tickets — the runtime's contract surface.",
          tags=(TAG_SAFETY,),
          composes_with=("driver", "governance"),
          demo_path="examples/slo_contract_demo.py"),
    _spec(name="costs", kind=KIND_INFRA,
          summary="Token usage and cost accounting per session and per tenant.",
          tags=(TAG_INTROSPECTION,),
          determinism=DETERMINISM_PURE),
    _spec(name="memory", kind=KIND_MEMORY,
          summary="Persistent JSONL memory store with keyword search across sessions.",
          tags=(TAG_RETRIEVAL,),
          composes_with=("knowledge", "reflection")),
    _spec(name="knowledge", kind=KIND_MEMORY,
          summary="KnowledgeGraph — typed entities, relations, and facts with provenance.",
          tags=(TAG_RETRIEVAL, TAG_SYMBOLIC),
          composes_with=("memory", "reasoner")),
    _spec(name="persistence", kind=KIND_INFRA,
          summary="Session persistence — durable state across restarts and crashes.",
          tags=(),
          composes_with=("runtime",)),
    _spec(name="skills", kind=KIND_MEMORY,
          summary="Skill library — durable procedural memory of distilled SOPs.",
          tags=(TAG_RETRIEVAL,),
          composes_with=("skillmine", "memory")),
    _spec(name="skillmine", kind=KIND_LEARNING,
          summary="Skill mining — turn successful traces into reusable named skills.",
          tags=(TAG_LLM, TAG_ADAPTIVE),
          composes_with=("skills", "selfeval"),
          dependency=DEP_LLM),
    _spec(name="reflection", kind=KIND_LEARNING,
          summary="Reflection — turn lived experience into durable memory and lessons.",
          tags=(TAG_LLM,),
          composes_with=("memory", "autoloop"),
          dependency=DEP_LLM),
    _spec(name="tools", kind=KIND_INFRA,
          summary="Tools the agent can call — file, shell, web, and memory primitives.",
          tags=(),
          composes_with=("runtime", "toolsynth"),
          determinism=DETERMINISM_NONE),
    _spec(name="toolsynth", kind=KIND_LEARNING,
          summary="Tool synthesis — the agent extends its own callable surface in a sandbox.",
          tags=(TAG_LLM, TAG_ADAPTIVE),
          composes_with=("tools", "runtime"),
          dependency=DEP_LLM),
    _spec(name="selfeval", kind=KIND_OBSERVABILITY,
          summary="Agent-generated regression eval bank with a promotion gate on the suite.",
          tags=(TAG_INTROSPECTION, TAG_CALIBRATION),
          composes_with=("autonomy", "skillmine")),
    _spec(name="market", kind=KIND_ECONOMICS,
          summary="TicketMarket — multi-tenant marketplace dispatch layer with priced lanes.",
          tags=(TAG_MULTI_AGENT,),
          composes_with=("governance", "economist", "negotiator")),
    _spec(name="economist", kind=KIND_ECONOMICS,
          summary="TicketEconomist — closed-loop margin defender and scenario simulator.",
          tags=(TAG_MULTI_OBJECTIVE,),
          composes_with=("market", "governance", "portfolio")),
    _spec(name="oracle", kind=KIND_OBSERVABILITY,
          summary="TicketOracle — counterfactual replay and auto-tuning policy advisor.",
          tags=(TAG_CAUSAL, TAG_INTROSPECTION),
          composes_with=("policy_lab", "counterfactor")),
    _spec(name="attest", kind=KIND_SAFETY,
          summary="Tamper-evident attestation receipts — proof-of-work chain for ticket execution.",
          tags=(TAG_SAFETY, TAG_REPLAY),
          composes_with=("runtime", "verifier")),
    _spec(name="agent", kind=KIND_INFRA,
          summary="Core streaming agent loop with adaptive thinking and tool dispatch.",
          tags=(TAG_LLM,),
          composes_with=("runtime", "tools", "skills"),
          dependency=DEP_LLM,
          determinism=DETERMINISM_NONE),

    # --- inference / prediction ----------------------------------------
    _spec(name="predictor", kind=KIND_INFERENCE,
          summary="Universal sequence prediction via Context Tree Weighting with regret bounds.",
          tags=(TAG_BAYESIAN, TAG_STREAMING, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("hedger", "forecaster"),
          references=("Willems-Shtarkov-Tjalkens 1995 CTW",)),
    _spec(name="forecaster", kind=KIND_INFERENCE,
          summary="Anytime-valid probabilistic forecasting with calibration tracking.",
          tags=(TAG_BAYESIAN, TAG_CALIBRATION, TAG_ANYTIME),
          certificate=CERT_ANYTIME,
          composes_with=("calibration", "drift", "predictor")),
    _spec(name="hedger", kind=KIND_INFERENCE,
          summary="Universal prediction with experts — online learning with vanishing regret.",
          tags=(TAG_ADAPTIVE, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("policy", "bandit"),
          references=("Cesa-Bianchi & Lugosi 2006",)),
    _spec(name="sampler", kind=KIND_INFERENCE,
          summary="Bayesian probabilistic inference — HMC / NUTS / SMC over a generative model.",
          tags=(TAG_BAYESIAN,),
          composes_with=("filterer", "active_inference"),
          demo_path="examples/sampler_demo.py" if False else None),
    _spec(name="filterer", kind=KIND_INFERENCE,
          summary="Bayesian state-space filtering — Kalman / particle / unscented variants.",
          tags=(TAG_BAYESIAN, TAG_STREAMING),
          composes_with=("sampler", "world_model")),
    _spec(name="embedder", kind=KIND_INFERENCE,
          summary="Distortion-bounded text embedding with formal Johnson-Lindenstrauss guarantee.",
          tags=(TAG_NUMERICAL, TAG_RETRIEVAL, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("memory", "knowledge")),
    _spec(name="reasoner", kind=KIND_INFERENCE,
          summary="Symbolic logical reasoning — forward chaining, resolution, congruence closure.",
          tags=(TAG_SYMBOLIC,),
          certificate=CERT_EXACT,
          determinism=DETERMINISM_PURE,
          composes_with=("knowledge", "verifier")),
    _spec(name="latent_reasoner", kind=KIND_INFERENCE,
          summary="Continuous-space chain-of-thought reasoning over latent thought vectors.",
          tags=(TAG_LLM, TAG_NUMERICAL),
          composes_with=("reasoner", "stepwiser"),
          demo_path="examples/latent_reasoner_demo.py"),
    _spec(name="imaginator", kind=KIND_INFERENCE,
          summary="Learned-world-model rollouts for planning under partial observability.",
          tags=(TAG_BAYESIAN, TAG_GENERATIVE, TAG_RL),
          composes_with=("world_model", "filterer", "active_inference"),
          demo_path="examples/imagine_reconcile_loop_demo.py"),
    _spec(name="world_model", kind=KIND_INFERENCE,
          summary="World-model observed-entity tracker with transition learning.",
          tags=(TAG_STREAMING,),
          composes_with=("imaginator", "filterer")),
    _spec(name="abductor", kind=KIND_INFERENCE,
          summary="Bayesian abductive inference — inference to the best explanation with posterior support.",
          tags=(TAG_BAYESIAN, TAG_SYMBOLIC),
          composes_with=("conjecturer", "refuter")),
    _spec(name="attributor", kind=KIND_OBSERVABILITY,
          summary="Data attribution / influence functions — per-point Cook's distance, LOO, TracIn, decision-flip certificates.",
          tags=(TAG_BAYESIAN, TAG_NUMERICAL, TAG_REPLAY, TAG_SAFETY),
          certificate=CERT_EXACT,
          composes_with=("curator", "conformal", "robustifier", "auditor",
                         "forecaster", "aligner"),
          demo_path="examples/attributor_demo.py"),
    _spec(name="conjecturer", kind=KIND_SCIENCE,
          summary="Automated mathematical conjecture generation with falsifiability gates.",
          tags=(TAG_SYMBOLIC, TAG_LLM),
          composes_with=("abductor", "refuter", "verifier"),
          dependency=DEP_LLM),
    _spec(name="refuter", kind=KIND_SCIENCE,
          summary="Automated falsification — adversarial counter-example search against a hypothesis.",
          tags=(TAG_SYMBOLIC,),
          composes_with=("conjecturer", "verifier")),
    _spec(name="scientist", kind=KIND_SCIENCE,
          summary="Sparse symbolic law discovery from observation — closed-form equation discovery.",
          tags=(TAG_SYMBOLIC, TAG_DISCRETE_OPT),
          composes_with=("conjecturer", "compressor", "inducer")),
    _spec(name="mentalist", kind=KIND_INFERENCE,
          summary="Bayesian theory-of-mind — recursive belief modelling over agents.",
          tags=(TAG_BAYESIAN, TAG_MULTI_AGENT),
          composes_with=("equilibrator", "intender", "persuader")),
    _spec(name="intender", kind=KIND_INFERENCE,
          summary="Inverse reinforcement learning — preference / reward inference from demonstrations.",
          tags=(TAG_RL, TAG_BAYESIAN),
          composes_with=("mentalist", "aligner")),
    _spec(name="analogist", kind=KIND_INFERENCE,
          summary="Structure-mapping analogical reasoning across domain models.",
          tags=(TAG_SYMBOLIC,),
          composes_with=("reasoner", "knowledge")),
    _spec(name="active_inference", kind=KIND_INFERENCE,
          summary="ActiveInferencer — free-energy POMDP planning with expected-free-energy policy selection.",
          tags=(TAG_BAYESIAN, TAG_RL, TAG_PLANNING),
          certificate=CERT_PAC,
          composes_with=("filterer", "sampler", "imaginator"),
          demo_path="examples/active_inference_demo.py"),
    _spec(name="conformal", kind=KIND_SAFETY,
          summary="ConformalPredictor — distribution-free, finite-sample-valid prediction sets.",
          tags=(TAG_CALIBRATION, TAG_PAC, TAG_SAFETY),
          certificate=CERT_PAC,
          composes_with=("calibration", "risk_control", "forecaster"),
          demo_path="examples/conformal_demo.py"),
    _spec(name="calibration", kind=KIND_SAFETY,
          summary="Turn raw p_success forecasts into trustworthy ones via isotonic / Platt fits.",
          tags=(TAG_CALIBRATION,),
          composes_with=("conformal", "preflight", "forecaster")),
    _spec(name="deliberator", kind=KIND_INFERENCE,
          summary="Adaptive sequential sampling kernel with anytime-valid stopping.",
          tags=(TAG_ANYTIME, TAG_ADAPTIVE),
          certificate=CERT_ANYTIME,
          composes_with=("arbiter", "auditor", "experiments")),
    _spec(name="compressor", kind=KIND_INFERENCE,
          summary="Minimum Description Length hypothesis selection across competing models.",
          tags=(TAG_BAYESIAN, TAG_SYMBOLIC),
          composes_with=("scientist", "inducer")),
    _spec(name="ranker", kind=KIND_INFERENCE,
          summary="Paired-comparison and partial-ranking inference — Plackett-Luce / Bradley-Terry.",
          tags=(TAG_BAYESIAN,),
          composes_with=("intender", "aligner")),
    _spec(name="cartographer", kind=KIND_LEARNING,
          summary="Zone-of-proximal-development curriculum kernel for skill acquisition.",
          tags=(TAG_ADAPTIVE,),
          composes_with=("curator", "continualist")),
    _spec(name="curator", kind=KIND_LEARNING,
          summary="Automated curriculum *generation* — open-ended task creation.",
          tags=(TAG_ADAPTIVE, TAG_LLM),
          composes_with=("cartographer", "continualist"),
          dependency=DEP_LLM),
    _spec(name="drift", kind=KIND_OBSERVABILITY,
          summary="DriftSentinel — anytime-valid sequential drift detection (CUSUM / GLR).",
          tags=(TAG_STREAMING, TAG_ANYTIME, TAG_CALIBRATION),
          certificate=CERT_ANYTIME,
          composes_with=("forecaster", "auditor", "selfeval")),
    _spec(name="debater", kind=KIND_SAFETY,
          summary="Multi-agent debate — Irving / Barnes-Christiano / Brown-Cohen / Hubinger / Khan-Hughes / Condorcet jury, with PAC truth-win-rate LCB + Nash exploitability check.",
          tags=(TAG_MULTI_AGENT, TAG_PAC, TAG_ANYTIME, TAG_REPLAY, TAG_SAFETY),
          certificate=CERT_PAC,
          composes_with=("mentalist", "truthserum", "reconciler", "equilibrator",
                         "arbiter", "auditor", "strategist", "attest"),
          demo_path="examples/debater_demo.py",
          references=(
              "Irving-Christiano-Amodei 2018",
              "Barnes-Christiano 2020",
              "Brown-Cohen-Irving-Piliouras 2023",
              "Hubinger 2020",
              "Khan-Hughes 2024",
              "Boland 1989",
          )),

    # --- optimisation ---------------------------------------------------
    _spec(name="annealer", kind=KIND_OPTIMIZATION,
          summary="Combinatorial optimisation — simulated annealing, parallel tempering, LAHC, basin hopping, tabu, Luby restart.",
          tags=(TAG_DISCRETE_OPT, TAG_PAC, TAG_REPLAY),
          certificate=CERT_PAC,
          composes_with=("submodular", "solver", "coalition", "portfolio", "scheduler"),
          demo_path="examples/annealer_demo.py",
          references=(
              "Kirkpatrick-Gelatt-Vecchi 1983",
              "Hukushima-Nemoto 1996",
              "Burke-Bykov 2017",
          )),
    _spec(name="bayesopt", kind=KIND_OPTIMIZATION,
          summary="BayesOpt — Bayesian optimisation with GP surrogates and acquisition functions.",
          tags=(TAG_CONTINUOUS_OPT, TAG_BAYESIAN, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("sampler", "calibration"),
          demo_path="examples/bayesopt_demo.py"),
    _spec(name="solver", kind=KIND_OPTIMIZATION,
          summary="CDCL satisfiability solver with conflict-driven clause learning and certified UNSAT proofs.",
          tags=(TAG_DISCRETE_OPT, TAG_SYMBOLIC, TAG_REPLAY),
          certificate=CERT_EXACT,
          determinism=DETERMINISM_PURE,
          composes_with=("planner", "verifier", "synthesizer")),
    _spec(name="planner", kind=KIND_OPTIMIZATION,
          summary="Classical planning via SAT compilation — STRIPS / PDDL.",
          tags=(TAG_PLANNING, TAG_SYMBOLIC, TAG_DISCRETE_OPT),
          certificate=CERT_EXACT,
          composes_with=("solver", "reasoner", "scheduler")),
    _spec(name="inducer", kind=KIND_OPTIMIZATION,
          summary="Levin Universal Search for program induction with optimal time bound.",
          tags=(TAG_DISCRETE_OPT, TAG_SYMBOLIC),
          certificate=CERT_EXACT,
          composes_with=("synthesizer", "scientist", "compressor")),
    _spec(name="synthesizer", kind=KIND_OPTIMIZATION,
          summary="Program synthesis — typed enumerative + neural-guided search over a DSL.",
          tags=(TAG_DISCRETE_OPT, TAG_SYMBOLIC, TAG_LLM),
          composes_with=("inducer", "verifier", "solver"),
          dependency=DEP_LLM),
    _spec(name="submodular", kind=KIND_OPTIMIZATION,
          summary="Discrete subset selection with provable (1-1/e) approximation — greedy / lazy-greedy / stochastic-greedy.",
          tags=(TAG_DISCRETE_OPT,),
          certificate=CERT_PAC,
          composes_with=("annealer", "coalition", "portfolio")),
    _spec(name="pareto", kind=KIND_OPTIMIZATION,
          summary="Multi-objective optimisation — Pareto-frontier discovery with hypervolume metric.",
          tags=(TAG_MULTI_OBJECTIVE, TAG_CONTINUOUS_OPT),
          composes_with=("bayesopt", "annealer", "robustifier"),
          demo_path="examples/pareto_coordination_demo.py"),
    _spec(name="transporter", kind=KIND_OPTIMIZATION,
          summary="Optimal transport — Sinkhorn / entropic regularisation for distributional matching.",
          tags=(TAG_NUMERICAL, TAG_CONTINUOUS_OPT),
          composes_with=("embedder", "diffuser")),
    _spec(name="robustifier", kind=KIND_OPTIMIZATION,
          summary="Distributionally Robust Optimisation — worst-case over an ambiguity ball.",
          tags=(TAG_CONTINUOUS_OPT, TAG_SAFETY, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("pareto", "annealer", "policy_improver")),
    _spec(name="distiller", kind=KIND_OPTIMIZATION,
          summary="Amortised policy / value approximation via behaviour cloning + distillation.",
          tags=(TAG_RL, TAG_NUMERICAL),
          composes_with=("policy_improver", "policy_lab", "imaginator")),
    _spec(name="diffuser", kind=KIND_OPTIMIZATION,
          summary="Score-based generative modelling — diffusion sampling under arbitrary score functions.",
          tags=(TAG_GENERATIVE, TAG_NUMERICAL),
          composes_with=("sampler", "transporter")),
    _spec(name="flower", kind=KIND_OPTIMIZATION,
          summary="Generative Flow Networks — flow-balanced policy gradient over combinatorial spaces.",
          tags=(TAG_DISCRETE_OPT, TAG_GENERATIVE, TAG_RL),
          composes_with=("policy_improver", "annealer"),
          demo_path="examples/flower_coordination_demo.py"),
    _spec(name="continualist", kind=KIND_LEARNING,
          summary="Continual / lifelong learning with elastic-weight-consolidation-style drift bounds.",
          tags=(TAG_ADAPTIVE, TAG_RL),
          composes_with=("cartographer", "curator", "distiller"),
          demo_path="examples/continualist_coordination_demo.py"),
    _spec(name="empowerer", kind=KIND_LEARNING,
          summary="Empowerment & intrinsic motivation — mutual-information-driven action selection.",
          tags=(TAG_RL, TAG_ADAPTIVE),
          composes_with=("active_inference", "imaginator", "policy_improver"),
          demo_path="examples/empowerer_demo.py"),
    _spec(name="stepwiser", kind=KIND_LEARNING,
          summary="Process-reward modelling — step-level credit assignment for reasoning chains.",
          tags=(TAG_RL, TAG_LLM, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("latent_reasoner", "aligner", "policy_improver"),
          demo_path="examples/stepwiser_demo.py"),
    _spec(name="pretunist", kind=KIND_LEARNING,
          summary="Test-time training — closed-form ridge adapter on the current task's support with PAC-Bayes generalisation bound, KL-budget projection, leverage-based abstention.",
          tags=(TAG_ADAPTIVE, TAG_PAC, TAG_ANYTIME, TAG_CALIBRATION),
          certificate=CERT_PAC,
          determinism=DETERMINISM_PURE,
          composes_with=("continualist", "stepwiser", "distiller", "aligner", "preflight", "selfeval"),
          demo_path="examples/pretunist_demo.py",
          references=(
              "Sun-Wang-Liu-Held-Efros-Hardt 2020",
              "Akyürek-Damani-Qiu-Guo-Suzgun-Kim-Andreas-Kim 2024 (ARC-AGI)",
              "Akyürek-Schuurmans-Andreas-Zhou-Ma 2023",
              "McAllester 1999",
              "Catoni 2007",
              "Howard-Ramdas-McAuliffe-Sekhon 2021",
          ),
          complexity="O(n d² + d³) per adapt; O(d²) per observe",
          notes="Operationalises the cell of the architecture's learning-timescale table left blank by frozen LLMs."),
    _spec(name="mechanizer", kind=KIND_OBSERVABILITY,
          summary="Mechanistic interpretability via over-complete sparse-autoencoder / K-SVD dictionary learning with Donoho-Elad identifiability certificate, activation patching, feature steering, and circuit graph.",
          tags=(TAG_NUMERICAL, TAG_INTROSPECTION, TAG_SAFETY, TAG_PAC, TAG_REPLAY),
          certificate=CERT_PAC,
          determinism=DETERMINISM_SEEDED,
          composes_with=(
              "attributor", "aligner", "debater", "conformal", "forecaster",
              "topologist", "knowledge", "pretunist", "verifier", "curator",
              "continualist", "driver", "strategist",
          ),
          demo_path="examples/mechanizer_demo.py",
          references=(
              "Olshausen-Field 1996",
              "Donoho-Elad 2003",
              "Tropp 2004",
              "Aharon-Elad-Bruckstein 2006",
              "Beck-Teboulle 2009",
              "Bricken-Templeton et al. (Anthropic) 2023",
              "Cunningham-Ewart-Riggs-Huben-Sharkey 2023",
              "Templeton et al. (Anthropic) 2024",
              "Gao-Goh-Kingma-Nichol (OpenAI) 2024",
              "Conmy-Mavor-Parker et al. 2023",
              "Maurer-Pontil 2009",
          ),
          complexity="O(K d max_iter) per fit; O(K d) per encode; O(K²) per circuit",
          notes="Pure stdlib; no NumPy.  Operationalises the alignment-research interpretability stack as a coordination-engine dispatchable runtime primitive."),
    _spec(name="policy_improver", kind=KIND_LEARNING,
          summary="Safe off-policy policy optimisation with conservative-step guarantees.",
          tags=(TAG_RL, TAG_SAFETY, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("policy_lab", "counterfactor", "robustifier"),
          demo_path="examples/policy_improver_demo.py"),
    _spec(name="policy_lab", kind=KIND_OBSERVABILITY,
          summary="Off-policy evaluation lab — importance-weighted return estimates with bounds.",
          tags=(TAG_RL, TAG_CAUSAL, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("counterfactor", "policy_improver"),
          demo_path="examples/policy_lab_demo.py"),

    # --- decision / game-theory ----------------------------------------
    _spec(name="bandit", kind=KIND_DECISION,
          summary="Sequential decision under uncertainty — UCB / Thompson / EXP3.",
          tags=(TAG_BAYESIAN, TAG_RL, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("policy", "hedger", "arbiter")),
    _spec(name="arbiter", kind=KIND_DECISION,
          summary="Fixed-confidence Best-Arm Identification — LIL'UCB / Track-and-Stop.",
          tags=(TAG_BAYESIAN, TAG_ANYTIME, TAG_PAC),
          certificate=CERT_ANYTIME,
          composes_with=("bandit", "deliberator", "experiments")),
    _spec(name="equilibrator", kind=KIND_GAMES,
          summary="Non-cooperative game-theoretic equilibria — Nash / correlated / coarse-correlated.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_AGENT),
          composes_with=("diplomat", "mentalist", "negotiator")),
    _spec(name="diplomat", kind=KIND_GAMES,
          summary="Counterfactual Regret Minimisation for extensive-form games.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_AGENT, TAG_RL),
          composes_with=("equilibrator", "annealer")),
    _spec(name="negotiator", kind=KIND_GAMES,
          summary="Multi-party allocation — sealed-bid combinatorial auctions with externality pricing.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_AGENT),
          composes_with=("market", "mechanism", "annealer")),
    _spec(name="mechanism", kind=KIND_GAMES,
          summary="Revenue-optimal mechanism design — Myerson / VCG variants.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_OBJECTIVE),
          composes_with=("negotiator", "market", "economist")),
    _spec(name="coalition", kind=KIND_GAMES,
          summary="Shapley-value credit assignment across cooperating agents.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_AGENT, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("submodular", "annealer", "economist")),
    _spec(name="persuader", kind=KIND_GAMES,
          summary="Bayesian persuasion / information design — sender-optimal signaling.",
          tags=(TAG_BAYESIAN, TAG_GAME_THEORY),
          composes_with=("mentalist", "equilibrator")),
    _spec(name="truthserum", kind=KIND_GAMES,
          summary="Incentive-compatible peer-prediction — Bayesian Truth Serum / surprise-based scoring.",
          tags=(TAG_GAME_THEORY, TAG_MULTI_AGENT),
          composes_with=("market", "mechanism")),
    _spec(name="reconciler", kind=KIND_GAMES,
          summary="Aumann agreement reconciliation across belief-disagreeing agents.",
          tags=(TAG_BAYESIAN, TAG_MULTI_AGENT),
          composes_with=("mentalist", "equilibrator"),
          demo_path="examples/reconciler_demo.py"),

    # --- safety / alignment --------------------------------------------
    _spec(name="aligner", kind=KIND_SAFETY,
          summary="Direct Preference Optimisation — reward-free preference learning over response pairs.",
          tags=(TAG_RL, TAG_SAFETY, TAG_LLM),
          composes_with=("intender", "ranker", "policy_improver"),
          dependency=DEP_LLM),
    _spec(name="auditor", kind=KIND_SAFETY,
          summary="Multiple-hypothesis testing with FDR / FWER control — Benjamini-Hochberg / Bonferroni.",
          tags=(TAG_SAFETY, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("experiments", "deliberator", "drift")),
    _spec(name="risk_control", kind=KIND_SAFETY,
          summary="Distribution-free finite-sample risk control via conformal risk certificates.",
          tags=(TAG_SAFETY, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("conformal", "calibration")),
    _spec(name="quantilizer", kind=KIND_SAFETY,
          summary="Safety-bounded optimisation — top-quantile sampling instead of argmax.",
          tags=(TAG_SAFETY, TAG_BAYESIAN),
          composes_with=("policy_improver", "robustifier")),
    _spec(name="privacy", kind=KIND_SAFETY,
          summary="Differential privacy as a runtime primitive with (ε,δ) accounting.",
          tags=(TAG_SAFETY,),
          certificate=CERT_DP,
          composes_with=("governance", "auditor")),
    _spec(name="verifier", kind=KIND_SAFETY,
          summary="LCF-style proof-certificate kernel — small trusted core, large verified theorems.",
          tags=(TAG_SYMBOLIC, TAG_SAFETY),
          certificate=CERT_EXACT,
          determinism=DETERMINISM_PURE,
          composes_with=("solver", "reasoner", "synthesizer")),

    # --- observability / experimentation -------------------------------
    _spec(name="experiments", kind=KIND_OBSERVABILITY,
          summary="A/B experiments as a first-class runtime primitive with sequential safety.",
          tags=(TAG_CALIBRATION, TAG_ANYTIME, TAG_PAC),
          certificate=CERT_ANYTIME,
          composes_with=("auditor", "deliberator", "experiment_design"),
          demo_path="examples/experiments_demo.py"),
    _spec(name="experiment_design", kind=KIND_OBSERVABILITY,
          summary="Bayesian Optimal Experiment Design — maximise expected information gain.",
          tags=(TAG_BAYESIAN, TAG_ADAPTIVE),
          composes_with=("experiments", "bayesopt")),
    _spec(name="causal", kind=KIND_OBSERVABILITY,
          summary="Heterogeneous treatment effects — doubly-robust + meta-learner estimators.",
          tags=(TAG_CAUSAL,),
          composes_with=("causal_discovery", "counterfactor"),
          demo_path="examples/causal_demo.py"),
    _spec(name="causal_discovery", kind=KIND_SCIENCE,
          summary="Causal structure learning from observational data — PC / GES / NOTEARS.",
          tags=(TAG_CAUSAL,),
          composes_with=("causal", "scientist"),
          demo_path="examples/causal_discovery_demo.py"),
    _spec(name="counterfactor", kind=KIND_OBSERVABILITY,
          summary="Sequential off-policy evaluation with importance-weighted bounds.",
          tags=(TAG_CAUSAL, TAG_RL, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("policy_lab", "policy_improver", "oracle")),
    _spec(name="topologist", kind=KIND_SCIENCE,
          summary="Topological data analysis — persistent homology and Mapper.",
          tags=(TAG_NUMERICAL,),
          composes_with=("embedder", "scientist"),
          demo_path="examples/topologist_demo.py"),
    _spec(name="sketcher", kind=KIND_OBSERVABILITY,
          summary="Bounded-memory streaming sketches — Count-Min / HyperLogLog / Bloom / KLL.",
          tags=(TAG_STREAMING, TAG_PAC),
          certificate=CERT_PAC,
          composes_with=("drift", "embedder")),
    _spec(name="speculator", kind=KIND_OPTIMIZATION,
          summary="Speculative execution — draft-and-verify decoding for cheaper LLM inference.",
          tags=(TAG_LLM, TAG_ADAPTIVE),
          composes_with=("runtime", "stepwiser"),
          dependency=DEP_LLM,
          demo_path="examples/speculator_demo.py"),
    _spec(name="composer", kind=KIND_COORDINATION,
          summary="Typed, certified compositional planning — guarantee-preserving pipeline composition.",
          tags=(TAG_PLANNING, TAG_SYMBOLIC),
          certificate=CERT_EXACT,
          composes_with=("verifier", "planner", "coordinator")),
    _spec(name="portfolio", kind=KIND_DECISION,
          summary="Allocate a fixed agent budget across many tickets — risk-adjusted allocation.",
          tags=(TAG_DISCRETE_OPT, TAG_MULTI_OBJECTIVE),
          composes_with=("submodular", "annealer", "economist", "strategist")),
    _spec(name="searcher", kind=KIND_OPTIMIZATION,
          summary="Bounded-anytime certified tree search — MCTS / A*-with-bound for question-answering.",
          tags=(TAG_PLANNING, TAG_ANYTIME, TAG_PAC),
          certificate=CERT_ANYTIME,
          composes_with=("planner", "solver", "verifier", "reasoner"),
          demo_path="examples/searcher_demo.py"),
    _spec(name="manifest", kind=KIND_INFRA,
          summary="Discoverable runtime-primitive catalog for the coordination engine.",
          tags=(TAG_INTROSPECTION, TAG_NETWORK),
          inputs=("intent string", "filter predicates"),
          outputs=("PrimitiveSpec list", "JSON catalog", "fingerprint"),
          composes_with=("server", "mcp", "protocol", "capabilities", "strategist"),
          determinism=DETERMINISM_PURE,
          notes="Single source of truth for primitive metadata; auto-discovery is a dev-time safety net."),
    _spec(name="scaler", kind=KIND_SCIENCE,
          summary="Scaling-law inference: predict loss at unseen (N, D) and the compute-optimal allocation.",
          tags=(TAG_NUMERICAL, TAG_PAC, TAG_REPLAY, TAG_CALIBRATION),
          inputs=("(N, D, loss) observations",),
          outputs=("FitResult", "ExtrapolatePoint", "ComputeOptimal", "ScalerCertificate"),
          composes_with=("economist", "stepwiser", "curator", "continualist",
                         "pretunist", "strategist", "portfolio", "selfeval",
                         "attest", "conformal"),
          certificate=CERT_PAC,
          determinism=DETERMINISM_SEEDED,
          dependency=DEP_STDLIB,
          demo_path="examples/scaler_demo.py",
          notes="Closed-form Chinchilla compute-optimal allocation; "
                "Kaplan / BNSL / Bahri alternative families; bootstrap-percentile CI; "
                "Hoeffding + empirical-Bernstein held-out RMSE LCB."),
    _spec(name="schemer", kind=KIND_SAFETY,
          summary="Strategic-deception / sandbagging detection: anytime-valid e-process tests over a model's behavioural ledger.",
          tags=(TAG_SAFETY, TAG_ANYTIME, TAG_REPLAY, TAG_BAYESIAN,
                TAG_INTROSPECTION),
          inputs=("Trial(model_id, task_id, context, outcome, latent_features)",),
          outputs=("TestResult", "SchemerVerdict", "SchemerCertificate",
                   "REC_TRUST|RESTRICT|QUARANTINE|ESCALATE_HUMAN"),
          composes_with=("auditor", "deliberator", "truthserum", "aligner",
                         "arbiter", "governance", "attest", "mentalist",
                         "mechanizer", "strategist", "portfolio"),
          certificate=CERT_ANYTIME,
          determinism=DETERMINISM_SEEDED,
          dependency=DEP_STDLIB,
          demo_path="examples/schemer_demo.py",
          notes="Honeypot SPRT / universal-portfolio e-process; paired "
                "Eval-Observed-vs-Unobserved sign test; high-vs-low-stakes "
                "refusal gap; deceptive-marker rate; strategic-refusal gap; "
                "Mantel permutation consistency. Product of independent "
                "e-values is e-valid under any dependence (Vovk-Wang 2021); "
                "Holm step-down for FWER across tests."),
    _spec(name="anticipator", kind=KIND_COORDINATION,
          summary="Sleep-time / anticipatory compute: pre-compute likely-next queries on idle GPU and serve real ones from cache with a hit-rate certificate.",
          tags=(TAG_ADAPTIVE, TAG_STREAMING, TAG_PAC, TAG_REPLAY,
                TAG_RETRIEVAL, TAG_PLANNING),
          inputs=("ContextRecord(ctx_id, ctx, deadline_hint)",
                  "Forecaster -> Iterable[Candidate]",
                  "Answerer -> (answer, realised_cost)"),
          outputs=("Plan", "PrecomputeResult", "ServeResult",
                   "AnticipatorCertificate", "AnticipatorReport"),
          composes_with=("forecaster", "predictor", "embedder", "costs",
                         "economist", "scaler", "scheduler", "memory",
                         "attest", "governance", "coordinator"),
          certificate=CERT_PAC,
          determinism=DETERMINISM_SEEDED,
          dependency=DEP_STDLIB,
          demo_path="examples/anticipator_demo.py",
          notes="0-1 knapsack (greedy ratio + exact branch-and-bound) over "
                "candidate (value, cost) pairs under hard sleep-time "
                "budget; Wilson + Hoeffding hit-rate CIs; empirical-"
                "Bernstein LCB on saved cost per serve; Merkle "
                "fingerprint chain over the entire pre-compute/serve "
                "loop. Companion to Speculator (active-stream "
                "acceleration) and Pretunist (test-time adaptation); "
                "Anticipator shifts compute to idle time. Lin et al. "
                "2025 (Letta) 'Sleep-time Compute' (arXiv:2504.13171)."),
)


# ---------------------------------------------------------------------------
# Auto-discovery (best-effort fallback)

def auto_discover(*, prefix: str = "agi") -> list[PrimitiveSpec]:
    """Walk the ``agi`` package and synthesise minimal specs for any
    module whose docstring fits the ``Name — summary.`` pattern.

    Used as a safety net when developing new primitives that haven't
    been added to ``_PRIMITIVE_TABLE`` yet.  CI should assert the
    curated table is exhaustive.
    """
    import pkgutil
    pkg = importlib.import_module(prefix)
    pkg_path = pkg.__path__  # type: ignore[attr-defined]
    out: list[PrimitiveSpec] = []
    for info in pkgutil.iter_modules(pkg_path):
        name = info.name
        if name.startswith("_") or name in {"manifest"}:
            continue
        try:
            m = importlib.import_module(f"{prefix}.{name}")
        except Exception:
            continue
        doc = (m.__doc__ or "").strip()
        if not doc:
            continue
        # First-line "Name — summary." extraction; tolerate "Name -" too.
        first = doc.splitlines()[0].strip()
        if " — " in first:
            _, _, summary = first.partition(" — ")
        elif " - " in first:
            _, _, summary = first.partition(" - ")
        else:
            summary = first
        summary = summary.strip().rstrip(".") + "."
        try:
            out.append(_spec(
                name=name,
                kind=KIND_INFRA,
                summary=summary,
                notes="auto-discovered from module docstring; add a curated row to _PRIMITIVE_TABLE.",
            ))
        except ManifestError:
            continue
    return out


# ---------------------------------------------------------------------------
# Default manifest factory

_default_lock = threading.RLock()
_default: Manifest | None = None


def default_manifest(*, fresh: bool = False, include_auto: bool = False) -> Manifest:
    """Return the canonical pre-populated catalog.

    Cached per process unless ``fresh=True``.  When ``include_auto`` is
    true, any module in ``agi/`` that isn't already in the curated
    table gets a synthesised entry from its docstring.
    """
    global _default
    with _default_lock:
        if not fresh and _default is not None and not include_auto:
            return _default
        m = Manifest(_PRIMITIVE_TABLE)
        if include_auto:
            for spec in auto_discover():
                if spec.name not in m:
                    m.register(spec)
        if not include_auto:
            _default = m
        return m


def reset_default_manifest() -> None:
    """Drop the cached default manifest.  Useful in tests."""
    global _default
    with _default_lock:
        _default = None
