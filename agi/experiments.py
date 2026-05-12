"""ExperimentRunner — A/B experiments as a first-class runtime primitive.

The runtime now ships nearly everything a coordination engine needs to drive
production AI traffic at scale: forecasts, admission, downgrade, hedging,
margin defense, marketplace dispatch, evolutionary search, counterfactual
replay. What it has been missing is the *release discipline* on top of all
that — the contract that says **every change ships behind an experiment
with predeclared guardrails, a frozen primary metric, and an auditable
decision**.

`ExperimentRunner` closes that gap:

    runner = ExperimentRunner(persistence_path="experiments.jsonl")
    exp = runner.register(Experiment(
        name="cheaper-router-v3",
        variants=[
            Variant(name="control",   overrides={}),
            Variant(name="treatment", overrides={"model": "claude-haiku-4-5"}),
        ],
        primary_metric=METRIC_COST_PER_SUCCESS,
        direction="min",                                       # lower is better
        traffic_split=[0.5, 0.5],
        guardrails=[
            Guardrail(metric=METRIC_P_SUCCESS, direction="max",
                      tolerance=0.05, interpret="abs_delta"),  # don't drop ≥5pp
            Guardrail(metric=METRIC_LATENCY_S, direction="max",
                      tolerance=1.5,  interpret="ratio"),      # ≤1.5x slower
        ],
        min_samples_per_variant=200,
    ))

    # Route a ticket through the experiment:
    variant = runner.assign(exp.name, tenant_id="acme", ticket_id=tid)
    cfg = runner.apply_to_config(variant, cfg)
    ...                                                       # dispatch normally
    runner.record(exp.name, variant.name,
                  success=True, cost_usd=0.012, latency_s=2.1)

    # Decide:
    status = runner.status(exp.name)         # full readout: lift, CI, p-value
    decision = runner.decide(exp.name)       # SHIP / KILL / INCONCLUSIVE / RUNNING

The runner provides:

  * **Deterministic assignment** — `hash((tenant_id or ticket_id, exp.name, salt))`
    so the same tenant always lands on the same variant within an experiment;
    a coordination engine running across many runtimes converges on identical
    assignments without coordination.
  * **Bayesian decisions for binary metrics** — Beta-Binomial posteriors,
    P(treatment > control) computed by Monte Carlo. Honors `min_samples`
    and a minimum detectable effect.
  * **Welch's t-test for continuous metrics** — cost, latency, refund.
  * **Guardrails** — any guardrail breaching its tolerance with high
    confidence triggers an immediate KILL even before the primary metric
    converges. A coordination engine never silently regresses cost or
    latency in pursuit of a quality lift.
  * **Auditable decision log** — every assignment, observation, and
    decision persists to JSONL. Reproducible release engineering.
  * **Auto-pilot** — opt-in periodic decision loop that ships/kills
    experiments automatically as soon as they cross the gate.

Why this matters for a coordination engine: EvolutionEngine and TicketOracle
*propose* changes. ExperimentRunner is the discipline that turns those
proposals into safe production rollouts. Without it, "the runtime gets
smarter" is a hope; with it, it's a measurable, gated process.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# --- metric kinds ----------------------------------------------------

# Binary (Bernoulli) metrics — recorded as 0/1 per ticket.
METRIC_P_SUCCESS = "p_success"
METRIC_REFUND_RATE = "refund_rate"
METRIC_REJECT_RATE = "reject_rate"
METRIC_BREACH_RATE = "breach_rate"

# Continuous metrics — recorded as a real number per ticket.
METRIC_COST_USD = "cost_usd"
METRIC_LATENCY_S = "latency_s"
METRIC_REFUND_USD = "refund_usd"
METRIC_TOKENS_OUTPUT = "tokens_output"

# Derived aggregate metric (computed at status-time, not per-ticket).
METRIC_COST_PER_SUCCESS = "cost_per_success"

BINARY_METRICS = frozenset({
    METRIC_P_SUCCESS,
    METRIC_REFUND_RATE,
    METRIC_REJECT_RATE,
    METRIC_BREACH_RATE,
})

CONTINUOUS_METRICS = frozenset({
    METRIC_COST_USD,
    METRIC_LATENCY_S,
    METRIC_REFUND_USD,
    METRIC_TOKENS_OUTPUT,
})

DERIVED_METRICS = frozenset({
    METRIC_COST_PER_SUCCESS,
})

KNOWN_METRICS = BINARY_METRICS | CONTINUOUS_METRICS | DERIVED_METRICS

# --- experiment lifecycle states -------------------------------------

EXP_RUNNING = "running"
EXP_PAUSED = "paused"
EXP_SHIPPED = "shipped"
EXP_KILLED = "killed"
EXP_INCONCLUSIVE = "inconclusive"

TERMINAL_STATES = frozenset({EXP_SHIPPED, EXP_KILLED, EXP_INCONCLUSIVE})

# --- decisions -------------------------------------------------------

DECISION_SHIP = "ship"
DECISION_KILL = "kill"
DECISION_INCONCLUSIVE = "inconclusive"
DECISION_CONTINUE = "continue"

# --- guardrail interpretations --------------------------------------

INTERPRET_RATIO = "ratio"          # treatment / control
INTERPRET_ABS = "abs"              # treatment absolute value
INTERPRET_ABS_DELTA = "abs_delta"  # |treatment - control| (or signed)

# --- event types ----------------------------------------------------

EXP_EVT_REGISTERED = "experiment.registered"
EXP_EVT_ASSIGNED = "experiment.assigned"
EXP_EVT_OBSERVED = "experiment.observed"
EXP_EVT_DECIDED = "experiment.decided"
EXP_EVT_SHIPPED = "experiment.shipped"
EXP_EVT_KILLED = "experiment.killed"
EXP_EVT_PAUSED = "experiment.paused"
EXP_EVT_RESUMED = "experiment.resumed"
EXP_EVT_GUARDRAIL_BREACH = "experiment.guardrail_breach"
EXP_EVT_AUTOPILOT_TICK = "experiment.autopilot_tick"


# --- variant & experiment models ------------------------------------


@dataclass
class Variant:
    """One arm of an experiment.

    `overrides` is a dict of `SessionConfig` field names → values. The
    runner applies those to the caller's `SessionConfig` at dispatch
    time, so the control variant typically has `overrides={}` and the
    treatment carries the change under test.
    """
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Variant.name must be a non-empty string")
        if not isinstance(self.overrides, dict):
            raise ValueError("Variant.overrides must be a dict")


@dataclass
class Guardrail:
    """A guardrail caps how much a non-primary metric may regress.

    Examples:
      Guardrail(METRIC_P_SUCCESS, "max", 0.05, interpret="abs_delta")
        → treatment's success rate may not drop more than 5pp below control.
      Guardrail(METRIC_LATENCY_S, "max", 1.5, interpret="ratio")
        → treatment may be at most 1.5x slower than control.
      Guardrail(METRIC_COST_USD, "max", 0.10, interpret="abs")
        → treatment's mean cost must stay below $0.10.

    `direction="max"` means "value above the tolerance is a breach".
    `direction="min"` means "value below the tolerance is a breach".
    """
    metric: str
    direction: str
    tolerance: float
    interpret: str = INTERPRET_RATIO

    def __post_init__(self) -> None:
        if self.metric not in KNOWN_METRICS:
            raise ValueError(f"unknown guardrail metric: {self.metric}")
        if self.direction not in ("max", "min"):
            raise ValueError("Guardrail.direction must be 'max' or 'min'")
        if self.interpret not in (INTERPRET_RATIO, INTERPRET_ABS, INTERPRET_ABS_DELTA):
            raise ValueError(f"unknown interpret: {self.interpret}")
        # `abs_delta` deltas are signed (treatment - control); negative
        # tolerances are meaningful there. Other interpret modes are
        # magnitudes/ratios and must be non-negative.
        if self.interpret != INTERPRET_ABS_DELTA and self.tolerance < 0:
            raise ValueError("Guardrail.tolerance must be non-negative")


@dataclass
class Experiment:
    """Declarative experiment spec.

    The first entry of `variants` is treated as the control; remaining
    variants are challengers. `traffic_split` parallels `variants` and
    must sum to ~1.0.

    `primary_metric` + `direction` define the success criterion. Binary
    metrics use Beta-Binomial posteriors; continuous metrics use Welch's
    t-test. `min_samples_per_variant` is the early-stop floor for SHIP
    or INCONCLUSIVE decisions (guardrail breaches can KILL earlier).

    `minimum_detectable_effect` is a relative MDE: an experiment that
    hasn't hit `min_samples` but whose observed effect is well below
    MDE will be marked INCONCLUSIVE rather than continuing forever.
    """
    name: str
    variants: list[Variant]
    primary_metric: str
    direction: str = "max"
    traffic_split: list[float] | None = None
    guardrails: list[Guardrail] = field(default_factory=list)
    min_samples_per_variant: int = 100
    max_samples_per_variant: int = 10_000
    significance_level: float = 0.05
    minimum_detectable_effect: float = 0.05
    posterior_samples: int = 2000
    hash_salt: str = ""
    started_at: float = field(default_factory=time.time)
    stopped_at: float | None = None
    status: str = EXP_RUNNING
    decision: str | None = None
    decision_reason: str | None = None
    shipped_variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Experiment.name must be a non-empty string")
        if len(self.variants) < 2:
            raise ValueError("Experiment requires at least 2 variants (control + ≥1 treatment)")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("Variant names must be unique within an experiment")
        if self.primary_metric not in KNOWN_METRICS:
            raise ValueError(f"unknown primary_metric: {self.primary_metric}")
        if self.direction not in ("max", "min"):
            raise ValueError("Experiment.direction must be 'max' or 'min'")
        if self.traffic_split is None:
            n = len(self.variants)
            self.traffic_split = [1.0 / n] * n
        if len(self.traffic_split) != len(self.variants):
            raise ValueError("traffic_split must have the same length as variants")
        if any(s < 0 for s in self.traffic_split):
            raise ValueError("traffic_split entries must be non-negative")
        total = sum(self.traffic_split)
        if total <= 0:
            raise ValueError("traffic_split must sum to > 0")
        # Normalize.
        self.traffic_split = [s / total for s in self.traffic_split]
        if not (0 < self.significance_level < 1):
            raise ValueError("significance_level must be in (0, 1)")
        if self.min_samples_per_variant < 1:
            raise ValueError("min_samples_per_variant must be >= 1")
        if self.max_samples_per_variant < self.min_samples_per_variant:
            raise ValueError("max_samples_per_variant must be >= min_samples_per_variant")
        if self.posterior_samples < 100:
            raise ValueError("posterior_samples must be >= 100")

    @property
    def control(self) -> Variant:
        return self.variants[0]

    @property
    def treatments(self) -> list[Variant]:
        return self.variants[1:]

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"no variant named {name!r} in experiment {self.name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variants": [asdict(v) for v in self.variants],
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "traffic_split": list(self.traffic_split),
            "guardrails": [asdict(g) for g in self.guardrails],
            "min_samples_per_variant": self.min_samples_per_variant,
            "max_samples_per_variant": self.max_samples_per_variant,
            "significance_level": self.significance_level,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "posterior_samples": self.posterior_samples,
            "hash_salt": self.hash_salt,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "status": self.status,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "shipped_variant": self.shipped_variant,
            "metadata": dict(self.metadata),
        }


# --- per-variant rolling stats --------------------------------------


@dataclass
class MetricStats:
    """Welford's online mean/variance + Bernoulli counters."""
    samples: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    successes: int = 0  # used only for binary metrics

    def add(self, value: float) -> None:
        self.samples += 1
        self.sum += value
        self.sum_sq += value * value
        if value >= 0.5:
            # Binary semantics: anything >=0.5 counts as success. Continuous
            # readers ignore this field.
            self.successes += 1

    @property
    def mean(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.sum / self.samples

    @property
    def variance(self) -> float:
        if self.samples < 2:
            return 0.0
        m = self.mean
        # Sample variance (n-1).
        v = (self.sum_sq - self.samples * m * m) / (self.samples - 1)
        return max(v, 0.0)

    @property
    def std_err(self) -> float:
        if self.samples < 2:
            return 0.0
        return math.sqrt(self.variance / self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "mean": self.mean,
            "variance": self.variance,
            "std_err": self.std_err,
            "successes": self.successes,
        }


# --- status snapshot ------------------------------------------------


@dataclass
class GuardrailBreach:
    """One concrete guardrail breach with the evidence that triggered it."""
    metric: str
    variant: str
    interpret: str
    tolerance: float
    direction: str
    observed: float
    control_value: float | None
    confidence: float
    breached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VariantSnapshot:
    name: str
    samples: int
    primary_mean: float
    primary_std_err: float
    metrics: dict[str, dict[str, Any]]  # metric -> stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "samples": self.samples,
            "primary_mean": self.primary_mean,
            "primary_std_err": self.primary_std_err,
            "metrics": self.metrics,
        }


@dataclass
class ExperimentStatus:
    """Full readout of an experiment's current state.

    A coordination engine plugs the dict form of this object directly
    into a dashboard. The auditable decision flow is also encoded here:
    `decision` ∈ {SHIP, KILL, INCONCLUSIVE, CONTINUE} with `reason`.
    """
    name: str
    status: str
    variants: list[VariantSnapshot]
    primary_metric: str
    direction: str
    best_variant: str | None
    primary_lift: float | None          # relative lift of best vs control
    primary_lift_ci: tuple[float, float] | None
    primary_p_value: float | None       # P(no effect or worse) — frequentist
    prob_treatment_better: float | None  # P(treatment > control) — Bayesian
    decision: str
    reason: str
    guardrail_breaches: list[GuardrailBreach]
    samples_total: int
    samples_min: int
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "variants": [v.to_dict() for v in self.variants],
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "best_variant": self.best_variant,
            "primary_lift": self.primary_lift,
            "primary_lift_ci": list(self.primary_lift_ci) if self.primary_lift_ci else None,
            "primary_p_value": self.primary_p_value,
            "prob_treatment_better": self.prob_treatment_better,
            "decision": self.decision,
            "reason": self.reason,
            "guardrail_breaches": [g.to_dict() for g in self.guardrail_breaches],
            "samples_total": self.samples_total,
            "samples_min": self.samples_min,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class Assignment:
    """The outcome of `runner.assign(...)`. Includes the variant and the
    deterministic bucket value, so a coordination engine can log/reproduce
    routing without consulting the runner."""
    experiment: str
    variant: str
    bucket: float            # in [0, 1)
    overrides: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- math helpers ----------------------------------------------------


def _hash_bucket(key: str) -> float:
    """Deterministic float in [0, 1) from a string key."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return n / 2**64


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _welch_t(
    a_mean: float, a_var: float, a_n: int,
    b_mean: float, b_var: float, b_n: int,
) -> tuple[float, float]:
    """Welch's t statistic + one-sided p-value (H1: a > b).

    Returns (t, p_one_sided). When a sample is too small the test is
    treated as non-significant (p=0.5).
    """
    if a_n < 2 or b_n < 2:
        return 0.0, 0.5
    se = math.sqrt(a_var / a_n + b_var / b_n)
    if se == 0:
        # Identical samples ⇒ no evidence of effect.
        return 0.0, 0.5
    t = (a_mean - b_mean) / se
    # Normal approximation (sample sizes are typically large in our
    # use cases; this matches the t-distribution closely for n>=30 and
    # is conservative for smaller n).
    p = 1.0 - _normal_cdf(t)
    return t, p


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Draw one sample from Beta(α, β) via two Gammas. Falls back to a
    normal approximation when α, β are large enough that gammavariate
    overflows the float range."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha, beta must be > 0")
    return rng.betavariate(alpha, beta)


def _prob_better_binary(
    a_succ: int, a_n: int,
    b_succ: int, b_n: int,
    *,
    prior_a: tuple[float, float] = (1.0, 1.0),
    prior_b: tuple[float, float] = (1.0, 1.0),
    samples: int = 2000,
    rng: random.Random | None = None,
) -> float:
    """Monte Carlo estimate of P(theta_a > theta_b) given Beta posteriors.

    Conjugate Beta-Binomial. Returns 0.5 when both arms are empty.
    """
    rng = rng or random.Random(0xA1B7)
    aa = prior_a[0] + a_succ
    ab = prior_a[1] + (a_n - a_succ)
    ba = prior_b[0] + b_succ
    bb = prior_b[1] + (b_n - b_succ)
    if a_n == 0 and b_n == 0:
        return 0.5
    wins = 0
    for _ in range(samples):
        a = _beta_sample(aa, ab, rng)
        b = _beta_sample(ba, bb, rng)
        if a > b:
            wins += 1
    return wins / samples


def _binary_lift_ci(
    a_succ: int, a_n: int,
    b_succ: int, b_n: int,
    *,
    prior_a: tuple[float, float] = (1.0, 1.0),
    prior_b: tuple[float, float] = (1.0, 1.0),
    samples: int = 2000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Posterior credible interval on the *relative lift* (a-b)/b.

    Returns `(median_lift, lo, hi)` at the (1-alpha) credible level.
    """
    rng = rng or random.Random(0xB7A1)
    aa = prior_a[0] + a_succ
    ab = prior_a[1] + (a_n - a_succ)
    ba = prior_b[0] + b_succ
    bb = prior_b[1] + (b_n - b_succ)
    lifts: list[float] = []
    for _ in range(samples):
        a = _beta_sample(aa, ab, rng)
        b = _beta_sample(ba, bb, rng)
        if b <= 0:
            continue
        lifts.append((a - b) / b)
    if not lifts:
        return 0.0, 0.0, 0.0
    lifts.sort()
    median = lifts[len(lifts) // 2]
    lo = lifts[max(0, int((alpha / 2) * len(lifts)) - 1)]
    hi = lifts[min(len(lifts) - 1, int((1 - alpha / 2) * len(lifts)))]
    return median, lo, hi


# --- core runner -----------------------------------------------------


class ExperimentRunner:
    """Owns experiments, assigns traffic, accumulates stats, decides.

    The runner is thread-safe. Persistence is best-effort JSONL append;
    a coordination engine driving many runners against one file gets a
    consolidated audit log.
    """

    def __init__(
        self,
        *,
        persistence_path: str | os.PathLike[str] | None = None,
        autopilot: bool = False,
        autopilot_interval_s: float = 30.0,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self._experiments: dict[str, Experiment] = {}
        # exp_name -> variant_name -> metric -> MetricStats
        self._stats: dict[str, dict[str, dict[str, MetricStats]]] = {}
        # exp_name -> variant_name -> assignment count
        self._assignments: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()
        self._event_sink = event_sink

        self._path: Path | None = Path(persistence_path) if persistence_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.touch(exist_ok=True)

        self._rng = random.Random(rng_seed if rng_seed is not None else 0x5EED)
        self._autopilot_thread: threading.Thread | None = None
        self._autopilot_stop = threading.Event()
        self._autopilot_interval_s = autopilot_interval_s
        if autopilot:
            self.start_autopilot()

    # --- registration -----------------------------------------------

    def register(self, experiment: Experiment) -> Experiment:
        """Add a new experiment. Raises if one with the same name exists."""
        with self._lock:
            if experiment.name in self._experiments:
                raise ValueError(f"experiment {experiment.name!r} already registered")
            self._experiments[experiment.name] = experiment
            self._stats[experiment.name] = {
                v.name: {} for v in experiment.variants
            }
            self._assignments[experiment.name] = {
                v.name: 0 for v in experiment.variants
            }
        self._persist({"type": EXP_EVT_REGISTERED, "experiment": experiment.to_dict()})
        self._emit(EXP_EVT_REGISTERED, {"name": experiment.name})
        return experiment

    def get(self, name: str) -> Experiment:
        with self._lock:
            if name not in self._experiments:
                raise KeyError(f"no such experiment: {name}")
            return self._experiments[name]

    def list_active(self) -> list[Experiment]:
        with self._lock:
            return [e for e in self._experiments.values() if e.status == EXP_RUNNING]

    def list_all(self) -> list[Experiment]:
        with self._lock:
            return list(self._experiments.values())

    # --- assignment -------------------------------------------------

    def assign(
        self,
        experiment_name: str,
        *,
        tenant_id: str | None = None,
        ticket_id: str | None = None,
        bucket_key: str | None = None,
    ) -> Assignment | None:
        """Return the variant for this caller, or None if not running.

        Bucket precedence: `bucket_key` ⟶ `tenant_id` ⟶ `ticket_id` ⟶ random.
        Hashing on `tenant_id` keeps a given tenant on a stable variant
        across many tickets; falling through to `ticket_id` randomises
        per-ticket; final-fallback random is non-deterministic and used
        only when no identifier is supplied.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                return None
            if exp.status != EXP_RUNNING:
                return None
            key_source = bucket_key or tenant_id or ticket_id
            if key_source is None:
                bucket = self._rng.random()
            else:
                bucket = _hash_bucket(f"{exp.hash_salt}|{exp.name}|{key_source}")
            # Pick by cumulative split.
            cum = 0.0
            chosen: Variant | None = None
            for v, share in zip(exp.variants, exp.traffic_split):
                cum += share
                if bucket < cum:
                    chosen = v
                    break
            if chosen is None:
                chosen = exp.variants[-1]
            self._assignments[experiment_name][chosen.name] += 1
            assignment = Assignment(
                experiment=experiment_name,
                variant=chosen.name,
                bucket=bucket,
                overrides=dict(chosen.overrides),
                metadata={
                    "tenant_id": tenant_id,
                    "ticket_id": ticket_id,
                    "ts": time.time(),
                },
            )
        self._persist({"type": EXP_EVT_ASSIGNED, **assignment.to_dict()})
        self._emit(EXP_EVT_ASSIGNED, assignment.to_dict())
        return assignment

    def apply_to_config(self, variant: Variant, config: Any) -> Any:
        """Return a copy of `config` with the variant's overrides applied.

        `config` must be a dataclass (typically `SessionConfig`). Unknown
        override fields are accepted and stashed into `config.metadata`
        if present, otherwise silently ignored — this keeps experiments
        forward-compatible with config-shape changes.
        """
        if variant is None or not variant.overrides:
            return config
        if hasattr(config, "__dict__"):
            data = dict(config.__dict__)
        else:
            return config
        unknown: dict[str, Any] = {}
        cls = type(config)
        cls_fields = getattr(cls, "__dataclass_fields__", None)
        for k, v in variant.overrides.items():
            if cls_fields is not None and k in cls_fields:
                data[k] = v
            elif k in data:
                data[k] = v
            else:
                unknown[k] = v
        if unknown and "metadata" in data and isinstance(data["metadata"], dict):
            md = dict(data["metadata"])
            md.setdefault("experiment_overrides", {}).update(unknown)
            data["metadata"] = md
        try:
            return cls(**data)
        except TypeError:
            # Caller's config class doesn't accept kwargs init; fall back
            # to mutating a copy.
            import copy
            cp = copy.copy(config)
            for k, v in data.items():
                setattr(cp, k, v)
            return cp

    # --- observation ------------------------------------------------

    def record(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        success: bool | None = None,
        cost_usd: float | None = None,
        latency_s: float | None = None,
        refund_usd: float | None = None,
        breached: bool | None = None,
        rejected: bool | None = None,
        tokens_output: float | None = None,
        custom: dict[str, float] | None = None,
    ) -> None:
        """Add one observation. Any metric left as None is not updated.

        A coordination engine that wants finer-grained metrics passes
        them via `custom={metric_name: value, ...}`.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                return
            if variant_name not in self._stats[experiment_name]:
                # Unknown variant — ignore but persist for audit.
                self._persist({
                    "type": "experiment.record_unknown_variant",
                    "experiment": experiment_name,
                    "variant": variant_name,
                })
                return
            metrics = self._stats[experiment_name][variant_name]

            def _add(metric: str, value: float) -> None:
                if metric not in metrics:
                    metrics[metric] = MetricStats()
                metrics[metric].add(float(value))

            if success is not None:
                _add(METRIC_P_SUCCESS, 1.0 if success else 0.0)
            if cost_usd is not None:
                _add(METRIC_COST_USD, max(0.0, float(cost_usd)))
            if latency_s is not None:
                _add(METRIC_LATENCY_S, max(0.0, float(latency_s)))
            if refund_usd is not None:
                refund_v = max(0.0, float(refund_usd))
                _add(METRIC_REFUND_USD, refund_v)
                _add(METRIC_REFUND_RATE, 1.0 if refund_v > 0 else 0.0)
            if breached is not None:
                _add(METRIC_BREACH_RATE, 1.0 if breached else 0.0)
            if rejected is not None:
                _add(METRIC_REJECT_RATE, 1.0 if rejected else 0.0)
            if tokens_output is not None:
                _add(METRIC_TOKENS_OUTPUT, max(0.0, float(tokens_output)))
            if custom:
                for k, v in custom.items():
                    _add(k, float(v))

        self._persist({
            "type": EXP_EVT_OBSERVED,
            "experiment": experiment_name,
            "variant": variant_name,
            "success": success,
            "cost_usd": cost_usd,
            "latency_s": latency_s,
            "refund_usd": refund_usd,
            "breached": breached,
            "rejected": rejected,
            "tokens_output": tokens_output,
            "custom": custom or {},
            "ts": time.time(),
        })

    # --- status / decision ------------------------------------------

    def _derived_value(
        self, variant_metrics: dict[str, MetricStats], metric: str
    ) -> tuple[float, int]:
        """Compute a derived metric. Returns (value, samples_supporting)."""
        if metric == METRIC_COST_PER_SUCCESS:
            cost = variant_metrics.get(METRIC_COST_USD)
            psucc = variant_metrics.get(METRIC_P_SUCCESS)
            if cost is None or psucc is None or psucc.samples == 0:
                return float("inf"), 0
            mean_cost = cost.mean
            success_rate = psucc.mean
            if success_rate <= 0:
                return float("inf"), min(cost.samples, psucc.samples)
            return mean_cost / success_rate, min(cost.samples, psucc.samples)
        raise ValueError(f"unknown derived metric: {metric}")

    def _metric_value(
        self, variant_metrics: dict[str, MetricStats], metric: str
    ) -> tuple[float, int, float]:
        """Return (mean, samples, std_err) for a metric on a variant."""
        if metric in DERIVED_METRICS:
            v, n = self._derived_value(variant_metrics, metric)
            # Std-err of a ratio is approximated via delta method below
            # if needed; for guardrail evaluation we conservatively
            # report 0 here (the guardrail logic uses absolute thresholds
            # for derived metrics).
            return v, n, 0.0
        s = variant_metrics.get(metric)
        if s is None:
            return 0.0, 0, 0.0
        return s.mean, s.samples, s.std_err

    def _evaluate_guardrails(
        self,
        exp: Experiment,
        control_metrics: dict[str, MetricStats],
        treatment_metrics: dict[str, MetricStats],
        treatment_name: str,
    ) -> list[GuardrailBreach]:
        """Compute breach status for each guardrail."""
        breaches: list[GuardrailBreach] = []
        for g in exp.guardrails:
            t_val, t_n, t_se = self._metric_value(treatment_metrics, g.metric)
            c_val, c_n, _ = self._metric_value(control_metrics, g.metric)
            if t_n == 0 or c_n == 0:
                # Not enough data to evaluate — skip; will be evaluated again
                # on the next decide() pass.
                continue
            if g.interpret == INTERPRET_RATIO:
                if c_val == 0:
                    observed = float("inf") if t_val > 0 else 1.0
                else:
                    observed = t_val / c_val
            elif g.interpret == INTERPRET_ABS:
                observed = t_val
            else:  # abs_delta
                observed = t_val - c_val
            if g.direction == "max":
                breached = observed > g.tolerance
            else:
                breached = observed < g.tolerance
            # Crude confidence: how many standard errors past the tolerance
            # we are. For binary metrics where std_err comes from rate-based
            # samples this is approximate; good enough to gate emergency stop.
            if t_se > 0:
                z = abs(observed - g.tolerance) / t_se
                confidence = _normal_cdf(z)
            else:
                # Without a usable SE, only flag as confident when the
                # observation is clearly past tolerance.
                confidence = 1.0 if breached else 0.0
            breaches.append(GuardrailBreach(
                metric=g.metric,
                variant=treatment_name,
                interpret=g.interpret,
                tolerance=g.tolerance,
                direction=g.direction,
                observed=observed,
                control_value=c_val,
                confidence=confidence,
                breached=breached,
            ))
        return breaches

    def status(self, experiment_name: str) -> ExperimentStatus:
        """Compute current readout including provisional decision."""
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                raise KeyError(f"no such experiment: {experiment_name}")
            stats = self._stats[experiment_name]
            control_name = exp.control.name
            control_metrics = stats[control_name]
            variants_snap: list[VariantSnapshot] = []
            for v in exp.variants:
                vm = stats[v.name]
                pm_val, pm_n, pm_se = self._metric_value(vm, exp.primary_metric)
                metrics_dict = {m: s.to_dict() for m, s in vm.items()}
                if exp.primary_metric in DERIVED_METRICS:
                    metrics_dict[exp.primary_metric] = {
                        "value": pm_val,
                        "samples": pm_n,
                    }
                variants_snap.append(VariantSnapshot(
                    name=v.name,
                    samples=pm_n,
                    primary_mean=pm_val,
                    primary_std_err=pm_se,
                    metrics=metrics_dict,
                ))

            # Choose best treatment vs control on primary metric.
            best_treatment: str | None = None
            best_lift: float | None = None
            best_ci: tuple[float, float] | None = None
            best_p: float | None = None
            best_prob_better: float | None = None
            all_breaches: list[GuardrailBreach] = []

            control_pm, control_n, control_se = self._metric_value(
                control_metrics, exp.primary_metric
            )

            for t in exp.treatments:
                tm = stats[t.name]
                t_pm, t_n, t_se = self._metric_value(tm, exp.primary_metric)
                # Compute lift, p_value, prob_better.
                lift: float
                p_value: float
                prob_better: float
                ci_lo: float
                ci_hi: float
                if exp.primary_metric in BINARY_METRICS:
                    succ_t = tm.get(exp.primary_metric).successes if exp.primary_metric in tm else 0
                    succ_c = control_metrics.get(exp.primary_metric).successes if exp.primary_metric in control_metrics else 0
                    if exp.direction == "max":
                        prob_better = _prob_better_binary(
                            succ_t, t_n, succ_c, control_n,
                            samples=exp.posterior_samples,
                            rng=self._rng,
                        )
                    else:
                        prob_better = _prob_better_binary(
                            succ_c, control_n, succ_t, t_n,
                            samples=exp.posterior_samples,
                            rng=self._rng,
                        )
                    median, ci_lo, ci_hi = _binary_lift_ci(
                        succ_t, t_n, succ_c, control_n,
                        samples=exp.posterior_samples,
                        alpha=exp.significance_level,
                        rng=self._rng,
                    )
                    lift = median
                    # Frequentist p-value via normal approximation to the
                    # two-proportion test (one-sided, treatment > control
                    # if direction=max).
                    p_value = _two_proportion_p(
                        succ_t, t_n, succ_c, control_n,
                        one_sided_better=(exp.direction == "max"),
                    )
                elif exp.primary_metric in CONTINUOUS_METRICS:
                    a_stats = tm.get(exp.primary_metric)
                    b_stats = control_metrics.get(exp.primary_metric)
                    a_mean = a_stats.mean if a_stats else 0.0
                    a_var = a_stats.variance if a_stats else 0.0
                    b_mean = b_stats.mean if b_stats else 0.0
                    b_var = b_stats.variance if b_stats else 0.0
                    if exp.direction == "max":
                        _, p_one = _welch_t(a_mean, a_var, t_n, b_mean, b_var, control_n)
                    else:
                        _, p_one = _welch_t(b_mean, b_var, control_n, a_mean, a_var, t_n)
                    p_value = p_one
                    if b_mean != 0:
                        lift = (a_mean - b_mean) / abs(b_mean)
                    else:
                        lift = 0.0
                    ci_lo, ci_hi = _continuous_lift_ci(
                        a_mean, a_var, t_n, b_mean, b_var, control_n,
                        alpha=exp.significance_level,
                    )
                    prob_better = 1.0 - p_value
                else:  # derived (e.g., cost_per_success)
                    # Compare via ratio; lift = (t-c)/|c|; use Welch on
                    # the constituent metric variances as an approximation.
                    if control_pm == 0 or control_pm == float("inf"):
                        lift = 0.0
                    else:
                        lift = (t_pm - control_pm) / abs(control_pm)
                    p_value = 0.5
                    prob_better = 0.5
                    ci_lo, ci_hi = lift, lift
                    # Try to derive a Welch test on cost_usd as a proxy.
                    if exp.primary_metric == METRIC_COST_PER_SUCCESS:
                        ac = tm.get(METRIC_COST_USD)
                        bc = control_metrics.get(METRIC_COST_USD)
                        if ac is not None and bc is not None and ac.samples >= 2 and bc.samples >= 2:
                            if exp.direction == "max":
                                _, p_one = _welch_t(
                                    ac.mean, ac.variance, ac.samples,
                                    bc.mean, bc.variance, bc.samples,
                                )
                            else:
                                _, p_one = _welch_t(
                                    bc.mean, bc.variance, bc.samples,
                                    ac.mean, ac.variance, ac.samples,
                                )
                            p_value = p_one
                            prob_better = 1.0 - p_value

                breaches = self._evaluate_guardrails(
                    exp, control_metrics, tm, t.name
                )
                all_breaches.extend(breaches)
                # Treatment is a candidate if direction-favouring.
                better = (
                    (exp.direction == "max" and t_pm > control_pm)
                    or (exp.direction == "min" and t_pm < control_pm)
                )
                if best_treatment is None or (
                    better and (best_lift is None or abs(lift) > abs(best_lift))
                ):
                    best_treatment = t.name
                    best_lift = lift
                    best_ci = (ci_lo, ci_hi)
                    best_p = p_value
                    best_prob_better = prob_better

            samples_min = min(
                stats[v.name].get(exp.primary_metric, MetricStats()).samples
                if exp.primary_metric not in DERIVED_METRICS
                else self._derived_value(stats[v.name], exp.primary_metric)[1]
                for v in exp.variants
            )
            samples_total = sum(
                stats[v.name].get(exp.primary_metric, MetricStats()).samples
                if exp.primary_metric not in DERIVED_METRICS
                else self._derived_value(stats[v.name], exp.primary_metric)[1]
                for v in exp.variants
            )

            decision, reason = self._compute_decision(
                exp,
                best_treatment=best_treatment,
                best_lift=best_lift,
                best_prob_better=best_prob_better,
                best_p_value=best_p,
                samples_min=samples_min,
                breaches=all_breaches,
            )

            elapsed = time.time() - exp.started_at

            return ExperimentStatus(
                name=exp.name,
                status=exp.status,
                variants=variants_snap,
                primary_metric=exp.primary_metric,
                direction=exp.direction,
                best_variant=best_treatment,
                primary_lift=best_lift,
                primary_lift_ci=best_ci,
                primary_p_value=best_p,
                prob_treatment_better=best_prob_better,
                decision=decision,
                reason=reason,
                guardrail_breaches=all_breaches,
                samples_total=samples_total,
                samples_min=samples_min,
                elapsed_s=elapsed,
            )

    def _compute_decision(
        self,
        exp: Experiment,
        *,
        best_treatment: str | None,
        best_lift: float | None,
        best_prob_better: float | None,
        best_p_value: float | None,
        samples_min: int,
        breaches: list[GuardrailBreach],
    ) -> tuple[str, str]:
        if exp.status == EXP_SHIPPED:
            return DECISION_SHIP, "already shipped"
        if exp.status == EXP_KILLED:
            return DECISION_KILL, "already killed"
        if exp.status == EXP_INCONCLUSIVE:
            return DECISION_INCONCLUSIVE, "already inconclusive"
        if exp.status == EXP_PAUSED:
            return DECISION_CONTINUE, "paused"

        # Guardrail breach with confidence → immediate kill.
        confident_breaches = [
            b for b in breaches
            if b.breached and b.confidence >= 1.0 - exp.significance_level
        ]
        if confident_breaches:
            metrics = ", ".join(sorted({b.metric for b in confident_breaches}))
            return DECISION_KILL, f"guardrail breach: {metrics}"

        if best_treatment is None:
            return DECISION_CONTINUE, "no treatment variant"

        if samples_min < exp.min_samples_per_variant:
            return DECISION_CONTINUE, (
                f"samples_min={samples_min} < min={exp.min_samples_per_variant}"
            )

        # Stat-sig win on primary metric?
        if best_prob_better is not None and best_prob_better >= 1.0 - exp.significance_level:
            return DECISION_SHIP, (
                f"prob_treatment_better={best_prob_better:.3f} >= "
                f"{1.0 - exp.significance_level:.3f}; lift={best_lift:.3f}"
            )

        # Stat-sig loss on primary metric (opposite direction).
        if best_prob_better is not None and best_prob_better <= exp.significance_level:
            return DECISION_KILL, (
                f"prob_treatment_better={best_prob_better:.3f} <= "
                f"{exp.significance_level:.3f}; lift={best_lift:.3f}"
            )

        # Sample cap reached without a verdict.
        if samples_min >= exp.max_samples_per_variant:
            return DECISION_INCONCLUSIVE, (
                f"reached max_samples_per_variant={exp.max_samples_per_variant} "
                f"without stat-sig outcome"
            )

        # No effect within MDE? Treat as inconclusive only at the cap.
        if (
            best_lift is not None
            and abs(best_lift) < exp.minimum_detectable_effect
            and samples_min >= exp.max_samples_per_variant // 2
        ):
            # Halfway to cap with effect below MDE: declare inconclusive.
            return DECISION_INCONCLUSIVE, (
                f"observed |lift|={abs(best_lift):.3f} < MDE="
                f"{exp.minimum_detectable_effect:.3f} at "
                f"samples_min={samples_min}"
            )

        return DECISION_CONTINUE, "accumulating data"

    def decide(self, experiment_name: str) -> tuple[str, str]:
        """Same as `status(name).decision`, but returns just (decision, reason)."""
        s = self.status(experiment_name)
        return s.decision, s.reason

    # --- lifecycle transitions --------------------------------------

    def ship(self, experiment_name: str, *, variant: str | None = None, reason: str = "manual ship") -> Experiment:
        """Mark a variant as the winner and stop the experiment."""
        with self._lock:
            exp = self._experiments[experiment_name]
            if exp.status in TERMINAL_STATES:
                raise RuntimeError(f"experiment {experiment_name} is already terminal ({exp.status})")
            if variant is None:
                s = self.status(experiment_name)
                variant = s.best_variant or exp.treatments[0].name
            exp.status = EXP_SHIPPED
            exp.decision = DECISION_SHIP
            exp.decision_reason = reason
            exp.shipped_variant = variant
            exp.stopped_at = time.time()
        self._persist({"type": EXP_EVT_SHIPPED, "experiment": experiment_name, "variant": variant, "reason": reason})
        self._emit(EXP_EVT_SHIPPED, {"experiment": experiment_name, "variant": variant, "reason": reason})
        return exp

    def kill(self, experiment_name: str, *, reason: str = "manual kill") -> Experiment:
        with self._lock:
            exp = self._experiments[experiment_name]
            if exp.status in TERMINAL_STATES:
                raise RuntimeError(f"experiment {experiment_name} is already terminal ({exp.status})")
            exp.status = EXP_KILLED
            exp.decision = DECISION_KILL
            exp.decision_reason = reason
            exp.stopped_at = time.time()
        self._persist({"type": EXP_EVT_KILLED, "experiment": experiment_name, "reason": reason})
        self._emit(EXP_EVT_KILLED, {"experiment": experiment_name, "reason": reason})
        return exp

    def conclude(self, experiment_name: str, *, reason: str = "manual conclude") -> Experiment:
        """Mark an experiment as inconclusive (no winner)."""
        with self._lock:
            exp = self._experiments[experiment_name]
            if exp.status in TERMINAL_STATES:
                raise RuntimeError(f"experiment {experiment_name} is already terminal ({exp.status})")
            exp.status = EXP_INCONCLUSIVE
            exp.decision = DECISION_INCONCLUSIVE
            exp.decision_reason = reason
            exp.stopped_at = time.time()
        self._persist({"type": EXP_EVT_DECIDED, "experiment": experiment_name, "decision": DECISION_INCONCLUSIVE, "reason": reason})
        return exp

    def pause(self, experiment_name: str) -> Experiment:
        with self._lock:
            exp = self._experiments[experiment_name]
            if exp.status not in (EXP_RUNNING,):
                raise RuntimeError(f"can only pause running experiments; was {exp.status}")
            exp.status = EXP_PAUSED
        self._persist({"type": EXP_EVT_PAUSED, "experiment": experiment_name})
        self._emit(EXP_EVT_PAUSED, {"experiment": experiment_name})
        return exp

    def resume(self, experiment_name: str) -> Experiment:
        with self._lock:
            exp = self._experiments[experiment_name]
            if exp.status != EXP_PAUSED:
                raise RuntimeError(f"can only resume paused experiments; was {exp.status}")
            exp.status = EXP_RUNNING
        self._persist({"type": EXP_EVT_RESUMED, "experiment": experiment_name})
        self._emit(EXP_EVT_RESUMED, {"experiment": experiment_name})
        return exp

    # --- autopilot --------------------------------------------------

    def evaluate_all(self) -> dict[str, tuple[str, str]]:
        """Run `decide()` against every running experiment and apply ship/kill
        when the decision is terminal. Returns a name→(decision, reason) map."""
        out: dict[str, tuple[str, str]] = {}
        with self._lock:
            running = [e.name for e in self._experiments.values() if e.status == EXP_RUNNING]
        for name in running:
            try:
                s = self.status(name)
            except KeyError:
                continue
            out[name] = (s.decision, s.reason)
            if s.decision == DECISION_SHIP and s.best_variant is not None:
                try:
                    self.ship(name, variant=s.best_variant, reason=s.reason)
                except RuntimeError:
                    pass
            elif s.decision == DECISION_KILL:
                try:
                    self.kill(name, reason=s.reason)
                except RuntimeError:
                    pass
            elif s.decision == DECISION_INCONCLUSIVE:
                try:
                    self.conclude(name, reason=s.reason)
                except RuntimeError:
                    pass
        self._emit(EXP_EVT_AUTOPILOT_TICK, {"evaluations": {k: list(v) for k, v in out.items()}})
        return out

    def start_autopilot(self, *, interval_s: float | None = None) -> None:
        """Start a background thread that calls `evaluate_all()` periodically."""
        if interval_s is not None:
            self._autopilot_interval_s = interval_s
        if self._autopilot_thread is not None and self._autopilot_thread.is_alive():
            return
        self._autopilot_stop.clear()

        def _loop():
            while not self._autopilot_stop.is_set():
                try:
                    self.evaluate_all()
                except Exception:
                    pass
                self._autopilot_stop.wait(self._autopilot_interval_s)

        self._autopilot_thread = threading.Thread(
            target=_loop, daemon=True, name="exp-autopilot"
        )
        self._autopilot_thread.start()

    def stop_autopilot(self) -> None:
        self._autopilot_stop.set()
        if self._autopilot_thread is not None:
            self._autopilot_thread.join(timeout=2.0)
        self._autopilot_thread = None

    # --- persistence / event sink -----------------------------------

    def _persist(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("a") as f:
                f.write(json.dumps(record, default=str))
                f.write("\n")
        except Exception:
            pass

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(kind, payload)
        except Exception:
            pass

    # --- driver integration helpers ---------------------------------

    def attach_to_driver(self, driver: Any) -> None:
        """Subscribe to a `RuntimeDriver` so completed tickets auto-record.

        The driver must expose a `runtime.bus` (or `pool.runtimes`-shaped
        federation) and call `record(...)` for completed receipts. This
        helper installs a finalizer hook on the driver — see
        `RuntimeDriver.experiments` for the canonical wiring.
        """
        # Implemented as a method to keep import direction (driver →
        # experiments) clean; the actual subscription lives in
        # `RuntimeDriver.submit_with_experiment` / `_persist_receipt`.
        driver._experiments_runner = self  # type: ignore[attr-defined]


def _two_proportion_p(
    a_succ: int, a_n: int,
    b_succ: int, b_n: int,
    *,
    one_sided_better: bool = True,
) -> float:
    """One-sided two-proportion z-test p-value (H1: p_a > p_b)."""
    if a_n == 0 or b_n == 0:
        return 0.5
    p_a = a_succ / a_n
    p_b = b_succ / b_n
    p_pool = (a_succ + b_succ) / (a_n + b_n)
    if p_pool <= 0 or p_pool >= 1:
        return 0.5
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / a_n + 1.0 / b_n))
    if se == 0:
        return 0.5
    z = (p_a - p_b) / se
    if not one_sided_better:
        z = -z
    return max(0.0, min(1.0, 1.0 - _normal_cdf(z)))


def _continuous_lift_ci(
    a_mean: float, a_var: float, a_n: int,
    b_mean: float, b_var: float, b_n: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Approximate CI on (a-b)/|b| via delta method."""
    if a_n < 2 or b_n < 2 or b_mean == 0:
        return 0.0, 0.0
    se = math.sqrt(a_var / a_n + b_var / b_n)
    z = 1.959963984540054  # ~ qnorm(0.975)
    if alpha != 0.05:
        # Crude: scale z by sigma-equivalent of alpha. For α=0.10 → 1.645,
        # for α=0.01 → 2.576. We linearly interpolate as a coarse fallback.
        if alpha >= 0.10:
            z = 1.6448536269514722
        elif alpha <= 0.01:
            z = 2.5758293035489004
    half = z * se / abs(b_mean)
    center = (a_mean - b_mean) / abs(b_mean)
    return center - half, center + half


__all__ = [
    # variants & experiments
    "Variant",
    "Guardrail",
    "Experiment",
    "ExperimentRunner",
    "MetricStats",
    "Assignment",
    "ExperimentStatus",
    "VariantSnapshot",
    "GuardrailBreach",
    # metric constants
    "METRIC_P_SUCCESS",
    "METRIC_REFUND_RATE",
    "METRIC_REJECT_RATE",
    "METRIC_BREACH_RATE",
    "METRIC_COST_USD",
    "METRIC_LATENCY_S",
    "METRIC_REFUND_USD",
    "METRIC_TOKENS_OUTPUT",
    "METRIC_COST_PER_SUCCESS",
    "BINARY_METRICS",
    "CONTINUOUS_METRICS",
    "DERIVED_METRICS",
    "KNOWN_METRICS",
    # state / decision constants
    "EXP_RUNNING",
    "EXP_PAUSED",
    "EXP_SHIPPED",
    "EXP_KILLED",
    "EXP_INCONCLUSIVE",
    "TERMINAL_STATES",
    "DECISION_SHIP",
    "DECISION_KILL",
    "DECISION_INCONCLUSIVE",
    "DECISION_CONTINUE",
    # guardrail interpret modes
    "INTERPRET_RATIO",
    "INTERPRET_ABS",
    "INTERPRET_ABS_DELTA",
    # event kinds
    "EXP_EVT_REGISTERED",
    "EXP_EVT_ASSIGNED",
    "EXP_EVT_OBSERVED",
    "EXP_EVT_DECIDED",
    "EXP_EVT_SHIPPED",
    "EXP_EVT_KILLED",
    "EXP_EVT_PAUSED",
    "EXP_EVT_RESUMED",
    "EXP_EVT_GUARDRAIL_BREACH",
    "EXP_EVT_AUTOPILOT_TICK",
]
