---
name: code-with-tests
description: Write a small piece of code together with the tests that prove it works.
args: [requirement, language]
tags: [code, tests, verification]
version: 1
---

When you're asked to write code:

1. Restate the requirement in your own words and list 3-5 concrete test cases
   — at least one happy path, one edge case, one failure case.
2. Write the test(s) first, in the user's chosen test framework (or pytest by
   default for Python). Use `write_file` to put them on disk in a sensible
   location.
3. Write the implementation. Keep it minimal — no premature abstraction.
4. Run the tests with `run_bash` (e.g., `python -m pytest -q <file>`). Iterate
   on failures.
5. Report: which tests pass, what the code does in one paragraph, and any
   known limitation.

Failure modes:
- Tests that just import-and-call without asserting outputs: not enough.
- Tests that depend on network or filesystem state we don't control:
  isolate them or skip with reason.
- Tests passing because they trivially can't fail: review each assertion.
