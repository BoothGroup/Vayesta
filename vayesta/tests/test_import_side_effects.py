import logging

import vayesta


def test_standard_log_levels_unchanged():
    assert logging.getLevelName(logging.WARNING) == "WARNING"
    assert logging.getLevelName("WARNING") == logging.WARNING
    assert logging.FATAL == logging.CRITICAL
    assert logging.getLevelName("FATAL") == logging.CRITICAL


def test_other_logger_levelnames():
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    other = logging.getLogger("test_import_side_effects.other")
    handler = Handler()
    other.addHandler(handler)
    try:
        other.warning("a warning")
        other.fatal("a fatal error")
    finally:
        other.removeHandler(handler)
    assert [(r.levelno, r.levelname) for r in records] == [(logging.WARNING, "WARNING"), (logging.CRITICAL, "CRITICAL")]


def test_deprecated_logged_once_as_warning():
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    log = logging.getLogger("test_import_side_effects.deprecated")
    handler = Handler()
    log.addHandler(handler)
    try:
        log.deprecated("this is deprecated")
        log.deprecated("this is deprecated")
    finally:
        log.removeHandler(handler)
    assert [(r.levelno, r.levelname) for r in records] == [(logging.WARNING, "WARNING")]


def test_get_git_hash_without_git(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert vayesta.get_git_hash(str(tmp_path)) == "<Not Found>"
