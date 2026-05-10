---
name: decompose-goal
description: Decompose an ambiguous, multi-step goal into a typed task DAG.
args: [goal, constraints]
tags: [planning, coordination, dag]
version: 1
---

When you're asked to plan, you are producing a graph the coordination engine
will execute. Be concrete; vague nodes produce vague work.

1. Read the goal carefully. List the ambiguities — things a reasonable person
   could interpret two ways. If the user is reachable, ask. Otherwise pick the
   most defensible interpretation and state it.
2. Identify the minimum set of *outputs* needed to satisfy the goal. Each
   output is a node. Don't include "thinking" or "research" as terminal nodes;
   research is in service of an output.
3. For each output node, identify what upstream nodes must produce its
   inputs. Draw the dependency edges.
4. Choose a `kind` for each node from: `chat`, `plan` (further decompose),
   `critique` (verify), `skill.invoke` (named procedure), `tool` (single
   tool call). Most nodes are `chat`.
5. Pick a `role` per node: `planner` for further-decompose nodes, `executor`
   for chats, `critic` for verification gates.
6. Add at least one `critique` node before any irreversible action (writing
   to a shared system, posting to a public channel, sending a message).
7. Output JSON matching `GraphSpec`:

   ```
   {
     "name": "<short slug>",
     "nodes": [
       {"id": "research",  "kind": "chat", "role": "executor",
        "input": {"message": "..."}, "depends_on": []},
       {"id": "draft",     "kind": "chat", "role": "executor",
        "input": {"message": "Draft using ${research.text}"}, "depends_on": ["research"]},
       {"id": "critique",  "kind": "critique",
        "input": {"prompt": "...", "response": "${draft.text}"},
        "depends_on": ["draft"]}
     ]
   }
   ```

Failure modes:
- Single giant node disguised as a plan — not a plan.
- Dependencies that can't be satisfied (cycle / unknown id).
- No verification step before a destructive action.
