"""Confabulator coordination demo: routing pool that re-tries hallucinated answers.

The runtime-engine story end-to-end in one runnable script:

  1. The coordination engine has three workers in its routing pool:

       * ``stable``   — usually gives one consistent answer.
       * ``shaky``    — disagrees with itself ~30 % of the time.
       * ``confab``   — confidently confabulates ~70 % of the time.

  2. The engine maintains one :class:`Confabulator` per worker (the
     truthfulness shadow) and one shared cross-worker auditor (the
     fleet-level FDR check, simulated here by Holm-combining the
     individual e-values).  Each worker comes with a labelled
     calibration pool — typically these are TruthfulQA-shaped probes
     held out from production traffic.

  3. As live traffic arrives, the engine queries each worker, scores
     the trial through the worker's Confabulator, and applies the
     returned recommendation:

       * ``trust``         → ship the answer.
       * ``regenerate``    → re-roll with higher diversity on the
                             *same* worker; ship if the second roll
                             scores better.
       * ``restrict``      → ship the answer wrapped in an "I'm not
                             sure" header.
       * ``escalate_human`` → drop the answer; escalate.
       * ``quarantine``    → pull the worker out of rotation pending
                             diagnosis (NaN score = no detector
                             fired).

  4. After ``N`` trials the engine prints a per-worker scorecard,
     re-routes future traffic away from quarantined workers, and the
     auditor reports the fleet-level certificate.

Run:  python examples/confabulator_coordination_demo.py
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.events import Event, EventBus
from agi.confabulator import (
    REC_ESCALATE,
    REC_QUARANTINE,
    REC_REGENERATE,
    REC_RESTRICT,
    REC_TRUST,
    Confabulator,
    ConfabulatorConfig,
    Sample,
    Trial,
    synthetic_trials,
)


def banner(title: str) -> None:
    bar = "-" * (len(title) + 6)
    print(f"\n{bar}\n   {title}\n{bar}")


# ---------------------------------------------------------------------------
# Worker simulator
# ---------------------------------------------------------------------------


@dataclass
class Worker:
    """A simulated LLM worker with a configurable hallucination profile.

    ``halluc_rate`` controls the fraction of trials sampled from the
    high-entropy distribution; ``noise_factor`` perturbs the truthful
    distribution so two trials at the same prompt don't return
    identical samples.
    """
    name: str
    halluc_rate: float
    noise_factor: float = 0.0
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def sample(self, prompt_idx: int, k: int = 5) -> tuple[bool, tuple[Sample, ...]]:
        """Draw K samples for a given prompt.

        Returns ``(is_hallucination, samples)``.  The first field is
        the ground-truth label the coordination engine could only know
        in offline calibration.
        """
        is_halluc = self.rng.random() < self.halluc_rate
        if is_halluc:
            # Wide dispersion across 4–5 plausible answers.
            n_unique = self.rng.randint(3, 5)
            answers = [f"answer_{prompt_idx}_v{j}" for j in range(n_unique)]
            samples = tuple(
                Sample(
                    text=self.rng.choice(answers),
                    mean_logprob=math.log(1.0 / n_unique),
                    n_tokens=2,
                )
                for _ in range(k)
            )
        else:
            # Concentrated on one cluster with optional rare deviation.
            main = f"answer_{prompt_idx}_main"
            samples = []
            for _ in range(k):
                if self.rng.random() < self.noise_factor:
                    txt = f"answer_{prompt_idx}_alt"
                else:
                    txt = main
                samples.append(Sample(
                    text=txt,
                    mean_logprob=-0.1,
                    n_tokens=2,
                ))
            samples = tuple(samples)
        return is_halluc, samples


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass
class WorkerStats:
    n_dispatched: int = 0
    n_trust: int = 0
    n_regenerate: int = 0
    n_restrict: int = 0
    n_escalate: int = 0
    n_quarantine: int = 0
    n_correct_trust: int = 0      # rec=trust AND not halluc
    n_caught_halluc: int = 0      # rec in {restrict, escalate} AND halluc
    n_missed_halluc: int = 0      # rec=trust AND halluc
    n_false_alarms: int = 0       # rec=escalate AND not halluc

    def add(self, rec: str, is_halluc: bool) -> None:
        self.n_dispatched += 1
        if rec == REC_TRUST:
            self.n_trust += 1
            if not is_halluc:
                self.n_correct_trust += 1
            else:
                self.n_missed_halluc += 1
        elif rec == REC_REGENERATE:
            self.n_regenerate += 1
        elif rec == REC_RESTRICT:
            self.n_restrict += 1
            if is_halluc:
                self.n_caught_halluc += 1
        elif rec == REC_ESCALATE:
            self.n_escalate += 1
            if is_halluc:
                self.n_caught_halluc += 1
            else:
                self.n_false_alarms += 1
        elif rec == REC_QUARANTINE:
            self.n_quarantine += 1


def calibrate_worker(worker: Worker, n_truthful: int = 30,
                     n_hallucinated: int = 30, k: int = 5,
                     *, seed: int) -> Confabulator:
    """Spin up a Confabulator for a worker and fit its threshold."""
    bus = EventBus()
    c = Confabulator(
        ConfabulatorConfig(
            seed=seed,
            budget_p0=0.10,
            alpha=0.05,
            bootstrap_b=200,
        ),
        bus=bus,
    )
    # The calibration pool is *known-labelled*: an oracle (e.g., a
    # human-judged held-out TruthfulQA slice) tells us which are
    # hallucinations.  We materialise it by drawing from the worker
    # itself under a fixed seed.
    held_out_rng = random.Random(seed + 1)
    worker.rng = held_out_rng
    prompt_idx = 0
    for _ in range(n_truthful + n_hallucinated):
        is_halluc, samples = worker.sample(prompt_idx, k=k)
        c.submit(f"cal_{worker.name}_{prompt_idx}",
                 samples, truth=(not is_halluc))
        prompt_idx += 1
    c.fit_threshold()
    return c


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> int:
    banner("Coordination engine spinning up three workers")

    workers = {
        "stable":  Worker(name="stable",  halluc_rate=0.05, noise_factor=0.10),
        "shaky":   Worker(name="shaky",   halluc_rate=0.30, noise_factor=0.20),
        "confab":  Worker(name="confab",  halluc_rate=0.70, noise_factor=0.30),
    }

    confabs: dict[str, Confabulator] = {}
    for i, (name, w) in enumerate(workers.items()):
        c = calibrate_worker(w, seed=i * 17)
        confabs[name] = c
        thr = c.threshold
        print(f"   {name:8s}  threshold={thr.threshold:.3f}  "
              f"AUROC={thr.auroc:.2f}  "
              f"CI=[{thr.auroc_lower:.2f}, {thr.auroc_upper:.2f}]")

    banner("Live traffic — 100 prompts dispatched")

    stats = {name: WorkerStats() for name in workers}
    quarantined: set[str] = set()
    routing_rng = random.Random(101)

    # Reset each worker's RNG for the live phase so the live trials are
    # independent of the calibration ones.
    for i, (name, w) in enumerate(workers.items()):
        w.rng = random.Random(1000 + i)

    for prompt_idx in range(100):
        # Pick a worker that's still in rotation.
        live = [n for n in workers if n not in quarantined]
        if not live:
            break
        name = routing_rng.choice(live)
        w = workers[name]
        is_halluc, samples = w.sample(prompt_idx, k=5)
        rep, rec = confabs[name].gate(f"live_{prompt_idx}", samples)
        stats[name].add(rec, is_halluc)

        if rec == REC_QUARANTINE:
            quarantined.add(name)
            print(f"   t={prompt_idx:3d} worker={name:6s} "
                  f"-> QUARANTINE; rerouting future traffic")

        if rec == REC_REGENERATE:
            # The engine re-rolls with a different seed.  In a real
            # deployment this would be a sampling-temperature bump or
            # a fall-back to a stronger model; here we simply ask the
            # SAME worker for a fresh draw and keep whichever rolls
            # lower.
            is_halluc2, samples2 = w.sample(prompt_idx, k=5)
            rep2, rec2 = confabs[name].gate(f"live_{prompt_idx}_retry", samples2)
            if rep2.combined_score < rep.combined_score:
                # The retry was an improvement; replace.
                if rec2 != REC_REGENERATE:
                    stats[name].add(rec2, is_halluc2)
                    stats[name].n_dispatched -= 1  # don't double-count

    banner("Per-worker scorecard")
    print(f"   {'worker':8s} {'disp':>5s} {'trust':>6s} {'regen':>6s} "
          f"{'restr':>6s} {'esc':>5s} {'quar':>5s} "
          f"{'caught':>7s} {'missed':>7s} {'false+':>7s}")
    for name, s in stats.items():
        print(f"   {name:8s} {s.n_dispatched:5d} {s.n_trust:6d} "
              f"{s.n_regenerate:6d} {s.n_restrict:6d} {s.n_escalate:5d} "
              f"{s.n_quarantine:5d} {s.n_caught_halluc:7d} "
              f"{s.n_missed_halluc:7d} {s.n_false_alarms:7d}")

    banner("Audit certificates — one per worker")
    for name, c in confabs.items():
        cert = c.certify()
        print(f"   {name:8s} verdict={cert.verdict:23s} "
              f"rec={cert.recommendation:14s}  "
              f"rate={cert.hallucination_rate:.2f}  "
              f"e={cert.e_value:.1e}  rejected={cert.rejected_h0}")
        print(f"            CI=[{cert.rate_lower_cp:.2f}, "
              f"{cert.rate_upper_cp:.2f}]  "
              f"AUROC={cert.auroc:.2f}  fp={cert.fingerprint_hash[:12]}")

    banner("Fleet-level FDR — Holm step-down across workers")
    p_vals: list[tuple[str, float]] = []
    for name, c in confabs.items():
        cert = c.certify()
        if cert.holm_smallest_p is not None:
            p_vals.append((name, cert.holm_smallest_p))
    sorted_p = sorted(p_vals, key=lambda kv: kv[1])
    print(f"   {'worker':8s} {'min Holm p':>13s}")
    for name, p in sorted_p:
        print(f"   {name:8s} {p:13.3e}")

    banner("Outcome summary")
    n_caught_total = sum(s.n_caught_halluc for s in stats.values())
    n_missed_total = sum(s.n_missed_halluc for s in stats.values())
    n_false_total = sum(s.n_false_alarms for s in stats.values())
    n_dispatched_total = sum(s.n_dispatched for s in stats.values())
    print(f"   total trials served: {n_dispatched_total}")
    print(f"   confabulations caught (restrict / escalate):  {n_caught_total}")
    print(f"   confabulations missed (shipped as trust):     {n_missed_total}")
    print(f"   false-positive escalations (truth + escalate): {n_false_total}")
    if quarantined:
        print(f"   quarantined: {', '.join(sorted(quarantined))}")
    else:
        print(f"   quarantined: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
