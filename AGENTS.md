# Guidelines for AI agents

## Code

- New functions and methods need type hints for all parameters and the return value. Use the builtin generics and
  the union syntax (`list[int]`, `float | None`) and import `Callable`, `Iterator` etc. from `collections.abc`.
- Document functions with NumPy-style docstrings (`Parameters` / `Returns` sections), as in the existing code.
- Raise an exception for failed runtime checks instead of using `assert`.
- Do not log a message before raising an exception with the same content.
- Format and lint with `ruff format` and `ruff check` (version pinned in `.pre-commit-config.yaml`) and type-check
  with `pyright` (`uv run pyright`); all three run in CI.

## Tests

- Tests live in `vayesta/tests` and run with `pytest`; CI runs `python .github/workflows/run_tests.py`.
- Test classes derive from `vayesta.tests.common.TestCase` (which provides `assertAllclose`) and use the shared
  systems in `vayesta/tests/testsystems.py`.
- Bug fixes come with a regression test.
