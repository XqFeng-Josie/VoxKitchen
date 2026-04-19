"""Runtime plumbing for multi-env stage execution.

See ``docs/architecture/multi-env.md`` for the big picture. Briefly: each
pipeline stage may run in its own Python environment (core / asr / tts)
with its own installed packages, and data crosses env boundaries as
jsonl.gz files on disk. This package contains:

- :mod:`env_resolver` — maps operator names to envs, detects current env
- :mod:`stage_runner` — subprocess entry point that runs one stage
- :mod:`dump_schemas` — build-time: export this env's operator schemas
- :mod:`merge_schemas` — build-time: combine per-env dumps into the
  unified ``op_schemas.json`` and ``op_env_map.json``
"""
