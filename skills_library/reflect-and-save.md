---
name: reflect-and-save
description: After a task ends, distill durable lessons into memory and (optionally) a new skill.
args: [task_summary, outcome]
tags: [memory, learning, post-task]
version: 1
---

After a non-trivial task ends — pass or fail — spend one short pass on
reflection. The goal is durable improvement, not therapy.

1. In one sentence, what did the task actually require? (Often different from
   what was originally asked.)
2. What worked? Save one concrete tactic to memory with the `lesson:worked` tag.
3. What didn't work? Save one concrete trap to memory with the `lesson:trap` tag.
4. If you used a procedure 3+ times across tasks, propose a new skill: write
   the SOP as a markdown body with `name`, `description`, `args`. Save as a
   draft note with the `skill-draft` tag for human review before committing.
5. If the task hit a tool failure, save the failure mode with the `tool-failure`
   tag so future tasks can plan around it.

Failure modes:
- Generic platitudes ("be careful"): useless, don't save them.
- Lessons that contradict objective verification: the verifier is the
  source of truth.
