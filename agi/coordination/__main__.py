"""CLI: drive a goal through the coordinator -> runtime pipeline.

Usage:
    python -m agi.coordination "Summarize ./README.md in 3 bullets"
    python -m agi.coordination --remote http://127.0.0.1:7777 "..."

Shows a live event stream as the graph executes, then prints the final
report (goal, final_text, cost, iterations, critic score).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from agi.coordination.coordinator import Coordinator, RuntimeClient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("goal", nargs="+", help="Goal to accomplish.")
    p.add_argument("--remote", default=None,
                   help="Use a remote runtime at URL (default: in-process).")
    p.add_argument("--workers", type=int, default=3,
                   help="Worker count for in-process runtime (default 3).")
    p.add_argument("--no-verify", action="store_true",
                   help="Disable critique gate at the end of the graph.")
    p.add_argument("--max-iterations", type=int, default=2,
                   help="Max plan-execute-verify revisions (default 2).")
    p.add_argument("--json", action="store_true",
                   help="Print the full report as JSON at the end.")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and args.remote is None:
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    goal = " ".join(args.goal)

    if args.remote:
        runtime = RuntimeClient(base_url=args.remote)
    else:
        from agi.runtime.server import Runtime
        runtime = Runtime(num_workers=args.workers)

    coord = Coordinator(runtime,
                        verify=not args.no_verify,
                        max_iterations=args.max_iterations)
    print(f"goal: {goal}\n")
    print("---- live events ----", flush=True)
    t_last = time.time()

    def on_event(ev):
        nonlocal t_last
        kind = ev.get("kind") if isinstance(ev, dict) else None
        payload = ev.get("payload", {}) if isinstance(ev, dict) else {}
        # Concise display.
        if kind == "graph.node_ready":
            print(f"  + node ready  {payload.get('node_id')}  (task {payload.get('task_id')})")
        elif kind == "task.started":
            t = payload.get("task", {})
            tags = t.get("spec", {}).get("tags", [])
            print(f"  > task started  {t.get('id')}  kind={t.get('spec', {}).get('kind')}  tags={tags}")
        elif kind == "task.tool_use":
            print(f"      tool  {payload.get('name')}")
        elif kind == "task.succeeded":
            t = payload.get("task", {})
            elapsed = payload.get("elapsed", 0)
            print(f"  + task done    {t.get('id')}  ({elapsed:.1f}s)  cost=${t.get('cost_usd', 0):.4f}")
        elif kind == "task.failed":
            t = payload.get("task", {})
            print(f"  ! task failed  {t.get('id')}  {t.get('error', '')[:100]}")
        elif kind == "graph.completed":
            r = payload.get("result", {})
            print(f"\n---- graph done ----  status={r.get('status')}  "
                  f"elapsed={r.get('elapsed', 0):.1f}s  cost=${r.get('total_cost_usd', 0):.4f}")
        elif kind == "graph.failed":
            r = payload.get("result", {})
            print(f"\n---- graph FAILED ----  errors={r.get('errors')}")
        t_last = time.time()

    report = coord.run(goal, on_event=on_event)
    print("\n---- final text ----")
    print(report.final_text)
    print("\n---- report ----")
    if args.json:
        print(json.dumps(report.to_dict(), default=str, indent=2))
    else:
        print(f"  goal          {report.goal}")
        print(f"  status        {report.status}")
        print(f"  iterations    {report.iterations}")
        print(f"  critic_score  {report.critic_score}")
        print(f"  cost          ${report.total_cost_usd:.4f}")
        print(f"  tokens        {report.total_tokens:,}")
        print(f"  elapsed       {report.elapsed:.1f}s")
    if hasattr(runtime, "shutdown"):
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
