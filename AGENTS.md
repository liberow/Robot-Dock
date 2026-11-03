# Repository instructions

- Public project: Robot Dock. Distribution/CLI: `robot-dock`. Python: `robot_dock`.
- Keep vendor sessions in policy adapters and physical dispatch in robot backends.
- Core imports must not depend on concrete integrations or the CLI.
- Do not automatically retry physical execution or uncertain inference.
- Shipped runs must use remote model actions. Do not add built-in fixed task
  trajectories, scripted policy fallbacks, or use offline test stubs as API validation.
- For model evaluation, do not add uncalibrated scene-specific position guards or
  alter Cartesian policy targets to make a task pass. Keep object ground truth
  evaluator-only; robot control must use behavior that has a real-device equivalent.
- Do not persist keys, tokens or authorization headers.
- Run `uv run poe ci` (format, repository validation, full-source types, tests and coverage).
- Configuration for Ruff/pytest/mypy lives only in pyproject.toml; do not add duplicate ini files.
- .env is local only; .env.example contains no real keys. Never copy another project's credentials.
- Keep the initial changes uncommitted until the owner explicitly approves commit/push.
- Keep development in this checkout. Prefer maintained usage/architecture/adapter
  documentation over one-off review or migration reports. Consolidate tool settings
  in pyproject.toml; do not retain temporary experiments as permanent modules.
- Before finishing each task, remove generated verification images/reports,
  temporary scripts, test outputs and tool caches from this checkout. Do not
  accumulate one-off validation files. Keep maintained regression tests, required
  assets and recorded inference episodes/datasets unless the owner requests removal.
- README.md is Simplified Chinese; README.en.md is English. Keep their content
  aligned and preserve reciprocal language links when updating either README.
