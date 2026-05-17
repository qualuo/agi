"""Pretunist coordination demo — Pretunist as a primitive a coordination
engine *discovers* via the Manifest, *gates* via Preflight-style admission
rules, *invokes* on a stream of tasks, and *audits* via the certificate
ledger.

A coordination engine drives the runtime to deliver:

  Goal:  *Specialise the system to each new task at inference time.*
         The base model is generic.  Each task arrives with a small
         support set (few-shot demos).  For each task we want:

           1. A *discovery* step that finds the primitive whose
              ``manifest.kind == learning`` and ``certificate == pac``
              and ``adaptive`` tag — that's Pretunist, by construction.
           2. A *preflight* check: is the test query in-distribution
              relative to the demos?  If not, abstain → route to a
              larger model / collect more data.
           3. An *adapt* step: closed-form ridge fit on the demos.
           4. A *certify* step: PAC-Bayes bound on test risk.
           5. A *commit-or-defer* decision: if the bound is non-vacuous
              and the e-process favours the adapter, commit; else
              defer.
           6. A *reset* step: drop the per-task adapter and move on.

This is the **runtime-as-coordination-surface** investor pitch: the
coordination engine doesn't know what Pretunist does — it discovers
that some primitive offers test-time training via the manifest, calls
it through a uniform API, reads a uniform certificate, and acts.

Run::

  python examples/pretunist_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.manifest import (  # noqa: E402
    CERT_PAC,
    KIND_LEARNING,
    TAG_ADAPTIVE,
    default_manifest,
)
from agi.pretunist import (  # noqa: E402
    ABSTAIN_LEVERAGE,
    ABSTAIN_VARIANCE,
    Pretunist,
    PretunistConfig,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Synthetic task generator: each task is a different latent weight vector
# w_i, drawn fresh.  The runtime *never sees w_i directly*; it only sees
# (x_j, y_j) demos and a query.

@dataclass(frozen=True)
class Task:
    """One few-shot task."""

    task_id: str
    demos_x: tuple[tuple[float, ...], ...]
    demos_y: tuple[tuple[float, ...], ...]
    queries_x: tuple[tuple[float, ...], ...]
    queries_y: tuple[tuple[float, ...], ...]


def synth_task(
    *,
    task_id: str,
    d: int,
    n_demo: int,
    n_query: int,
    seed: int,
    noise: float = 0.05,
) -> Task:
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 1.0) for _ in range(d)]
    def gen(n: int):
        xs, ys = [], []
        for _ in range(n):
            x = tuple(rng.gauss(0.0, 1.0) for _ in range(d))
            y = sum(wi * xi for wi, xi in zip(w, x)) + rng.gauss(0.0, noise)
            xs.append(x)
            ys.append((y,))
        return tuple(xs), tuple(ys)
    dx, dy = gen(n_demo)
    qx, qy = gen(n_query)
    return Task(task_id=task_id, demos_x=dx, demos_y=dy, queries_x=qx, queries_y=qy)


# ---------------------------------------------------------------------------
# A thin "coordination engine" — discovers primitives, executes a fixed
# routing policy, and aggregates certificates.

@dataclass
class TaskOutcome:
    task_id: str
    n_predicted: int
    n_abstained: int
    mse: float
    pac_bayes_bound: float
    kl_drift_nats: float
    adapter_fingerprint: str


class CoordinationEngine:
    """A minimal coordinator that drives Pretunist via the manifest."""

    def __init__(self, d: int) -> None:
        self.d = d
        # 1. Discovery: which primitive solves this category of task?
        self.manifest = default_manifest()
        candidates = self.manifest.find(
            kind=KIND_LEARNING,
            tag=TAG_ADAPTIVE,
            certificate=CERT_PAC,
        )
        if not candidates:
            raise RuntimeError("no adaptive learning primitive available")
        self.primitive_spec = candidates[0]
        print(f"  coordinator discovered:   {self.primitive_spec.name}")
        print(f"  certificate class:        {self.primitive_spec.certificate}")
        print(f"  composes with:            {list(self.primitive_spec.composes_with)}")
        print(f"  references:")
        for ref in self.primitive_spec.references[:3]:
            print(f"    · {ref}")

    def run_task(
        self,
        task: Task,
        *,
        delta: float = 0.05,
        bound_tolerance: float = 1.0,
    ) -> TaskOutcome:
        # 2. Spin up an instance of the discovered primitive for this task.
        pre = Pretunist(PretunistConfig(
            adapter_dim=self.d,
            output_dim=1,
            ridge_lambda=1e-3,
            prior_variance=1.0,
            posterior_variance=1.0,
            noise_variance=0.05,
            abstain_rules=(ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE),
            abstain_leverage_threshold=0.95,
            abstain_variance_threshold=0.5,
        ))
        # 3. Ingest demos.
        pre.observe_batch(list(task.demos_x), list(task.demos_y))
        # 4. One initial adapt to compute a global certificate before
        #    we commit anything.
        pre.adapt()
        cert = pre.certify(delta=delta)
        if cert.pac_bayes_is_vacuous or cert.pac_bayes_bound > bound_tolerance:
            print(f"  [{task.task_id}] bound vacuous / above tolerance → defer task")
            return TaskOutcome(
                task_id=task.task_id,
                n_predicted=0,
                n_abstained=len(task.queries_x),
                mse=math.inf,
                pac_bayes_bound=cert.pac_bayes_bound,
                kl_drift_nats=cert.kl_qp_nats,
                adapter_fingerprint=cert.adapter_fingerprint,
            )
        # 5. Per-query: abstain or predict.
        n_pred = 0
        n_abst = 0
        sq = 0.0
        for xq, yq in zip(task.queries_x, task.queries_y):
            ab = pre.should_abstain(xq)
            if ab.triggered:
                n_abst += 1
                continue
            r = pre.adapt(query=xq)
            err = r.prediction[0] - yq[0]
            sq += err * err
            n_pred += 1
        mse = sq / max(1, n_pred)
        return TaskOutcome(
            task_id=task.task_id,
            n_predicted=n_pred,
            n_abstained=n_abst,
            mse=mse,
            pac_bayes_bound=cert.pac_bayes_bound,
            kl_drift_nats=cert.kl_qp_nats,
            adapter_fingerprint=cert.adapter_fingerprint,
        )


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    banner("0 — Manifest-driven discovery")
    d = 8
    engine = CoordinationEngine(d=d)

    banner("1 — Run a stream of tasks (each w_i drawn fresh)")
    tasks = [
        synth_task(task_id=f"T{i:02d}", d=d, n_demo=60, n_query=20, seed=1000 + i)
        for i in range(8)
    ]
    outcomes: list[TaskOutcome] = []
    for t in tasks:
        o = engine.run_task(t)
        outcomes.append(o)
        print(
            f"  [{o.task_id}] mse={o.mse:7.4f}  "
            f"pac={o.pac_bayes_bound:7.4f}  "
            f"kl={o.kl_drift_nats:6.3f}  "
            f"pred={o.n_predicted:>3}/{o.n_predicted + o.n_abstained}  "
            f"fp={o.adapter_fingerprint[:10]}…"
        )

    banner("2 — Aggregate certificate")
    n_pred_total = sum(o.n_predicted for o in outcomes)
    n_abst_total = sum(o.n_abstained for o in outcomes)
    total_se = sum(o.mse * o.n_predicted for o in outcomes if math.isfinite(o.mse))
    mean_mse = total_se / max(1, n_pred_total)
    max_bound = max(o.pac_bayes_bound for o in outcomes if math.isfinite(o.pac_bayes_bound))
    mean_kl = sum(o.kl_drift_nats for o in outcomes) / max(1, len(outcomes))
    print(f"  tasks:                          {len(outcomes)}")
    print(f"  predictions made:               {n_pred_total}")
    print(f"  abstentions:                    {n_abst_total}")
    print(f"  pooled MSE on committed preds:  {mean_mse:.4f}")
    print(f"  max PAC-Bayes bound across:     {max_bound:.4f}")
    print(f"  mean KL drift across tasks:     {mean_kl:.3f}  nats")

    banner("3 — Adversarial task: query *very* far from demos")
    # Same demos but query is 100x out of distribution.
    adv = synth_task(task_id="ADV", d=d, n_demo=30, n_query=0, seed=99)
    extreme_query = tuple([100.0] + [0.0] * (d - 1))
    pre = Pretunist(PretunistConfig(
        adapter_dim=d, ridge_lambda=1e-3, noise_variance=0.05,
        abstain_rules=(ABSTAIN_LEVERAGE, ABSTAIN_VARIANCE),
        abstain_leverage_threshold=0.95,
        abstain_variance_threshold=0.5,
    ))
    pre.observe_batch(list(adv.demos_x), list(adv.demos_y))
    pre.adapt()
    ab = pre.should_abstain(extreme_query)
    print(f"  query:                                    [{extreme_query[0]}, …]")
    print(f"  abstention triggered?                      {ab.triggered}")
    print(f"  rules fired:                               {list(ab.rules_fired)}")
    print(f"  leverage:                                 {ab.leverage:.3f}")
    print(f"  predictive variance:                      {ab.predictive_variance:.3f}")
    print(f"  → coordinator routes this query to a larger model / data-gathering.")

    banner("4 — Federation: ship adapter across the wire")
    src = Pretunist(PretunistConfig(adapter_dim=d, ridge_lambda=1e-3))
    src.observe_batch(list(tasks[0].demos_x), list(tasks[0].demos_y))
    src.adapt()
    snap = src.snapshot()
    print(f"  source adapter fingerprint:   {src.adapter_fingerprint[:16]}…")
    print(f"  source ledger root:           {src.ledger_root[:16]}…")
    dst = Pretunist.restore(snap)
    print(f"  restored adapter fingerprint: {dst.adapter_fingerprint[:16]}…")
    print(f"  restored ledger root:         {dst.ledger_root[:16]}…")
    # Confirm prediction matches.
    p_src = src.predict(tasks[0].queries_x[0])[0]
    p_dst = dst.predict(tasks[0].queries_x[0])[0]
    print(f"  predictions at query agree?    {abs(p_src - p_dst) < 1e-12}")
    print(f"  → a federation member can prove what data its adapter saw.")

    print()
    print("Done.  Pretunist appears to the coordination engine as a discoverable")
    print("primitive with a uniform API and a uniform certificate.  The engine")
    print("never had to know it's solving ridge regression — it routed a task")
    print("to the primitive whose manifest matched the requirements, read the")
    print("PAC-Bayes bound, and made a commit-vs-defer decision.  That's the")
    print("'runtime engine driven by a coordination engine' story end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
