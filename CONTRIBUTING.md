# Contributing

Install with `uv sync --extra mujoco`. Run Ruff, mypy and pytest as documented in
the README (`uv run poe ci`). Keep the core free of vendor SDKs, simulator imports and provider-name
branches. Optional integration dependencies belong in extras.

Add a real adapter before generalizing a protocol. Prefer ordinary functions and
explicit composition to registries, services or workflow languages. Preserve
source observation identity and report measured partial dispatch counts.

Tests should cover behavior: partial execution, unknown results, stale frames,
session feedback, cancellation and cleanup. A successful HTTP response is not a
task-success assertion. Keep credentials and generated recordings out of Git.

Environment setup: copy `.env.example` to `.env`, keep real keys local, and use
`--env-file` for an explicit alternate file. Do not make library imports load
credentials. Optional hooks can be installed with `uv run pre-commit install`.
Full-source mypy checks are required; exceptions for untyped third-party libraries
are scoped to those imports, not used to exclude application modules.
