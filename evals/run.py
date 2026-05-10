"""Eval runner.

Reads evals/tasks.jsonl, runs each task with a fresh Agent + isolated memory,
checks the result, prints a summary.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Make `agi` importable when running as `python evals/run.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agi.agent import Agent
from agi.memory import Memory


def check(task: dict, response_text: str) -> tuple[bool, str]:
    spec = task["check"]
    kind = spec["type"]
    if kind == "contains":
        ok = spec["value"].lower() in response_text.lower()
        return ok, f"expected substring {spec['value']!r}"
    if kind == "all_contain":
        missing = [v for v in spec["values"] if v not in response_text]
        return not missing, f"missing substrings: {missing}" if missing else "ok"
    if kind == "file_contents":
        path = Path(spec["path"])
        if not path.exists():
            return False, f"file {path} not created"
        actual = path.read_text().strip()
        ok = spec["value"].strip() in actual
        return ok, f"file content was {actual!r}"
    return False, f"unknown check type {kind!r}"


def run_one(task: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory(path=Path(tmp) / "memory.jsonl")
        agent = Agent(memory=memory, verbose=False)
        t0 = time.time()
        try:
            response = agent.chat(task["prompt"])
            err: str | None = None
        except Exception as e:
            response = ""
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        passed, detail = (False, err) if err else check(task, response)
    return {
        "id": task["id"],
        "passed": passed,
        "elapsed": elapsed,
        "detail": detail,
        "response": response[:300],
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    tasks_path = Path(__file__).parent / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]

    print(f"running {len(tasks)} tasks...\n")
    results = []
    for task in tasks:
        print(f"  {task['id']:<12} ", end="", flush=True)
        result = run_one(task)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}  ({result['elapsed']:.1f}s)  {result['detail']}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
