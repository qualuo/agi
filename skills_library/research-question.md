---
name: research-question
description: Find a credible, well-sourced answer to a factual question using the open web.
args: [question, depth]
tags: [research, web, fact-check]
version: 1
---

When you receive a research question, do the following:

1. State what you already know about the topic in one sentence. Mark any
   assumption you'd want to verify.
2. Use `web_search` to gather 3-5 candidate sources. Prefer primary sources
   (official sites, peer-reviewed work, original publications) over
   aggregators.
3. For the most promising 2-3 sources, use `web_fetch` to read the underlying
   page rather than relying on the search snippet.
4. Cross-check: at least two independent sources must agree on the core fact.
   If they conflict, surface the conflict instead of picking one.
5. Save any durable fact you learned to long-term memory with the `fact` tag
   and a short citation (URL + access date).
6. Return: a one-paragraph answer, then a "Sources" list with URLs.

Failure modes:
- Single-source claim: never report as confirmed.
- Date-sensitive fact (prices, status, current events): include "as of <date>".
- Behind-paywall source: state that the source is paywalled and what was
  accessible.
