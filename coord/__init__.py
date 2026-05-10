"""Coordination clients.

The runtime in `agi.runtime` is the substrate. Clients in this package
demonstrate how a coordinator drives one or more runtimes as subprocesses
over the JSON-line protocol.

This is *not* the agent loop — it's the layer above. A real coordination
engine would live in its own service and pool runtimes by capability. The
clients here are minimal and self-contained for demonstration.
"""
