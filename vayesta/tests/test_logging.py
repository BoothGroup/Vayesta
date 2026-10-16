import logging
import os
import subprocess
import sys

import pytest

import vayesta


@pytest.fixture
def restore_logging():
    """Restore the handlers and level of the Vayesta logger after the test."""
    handlers = vayesta.log.handlers[:]
    tracked = vayesta._handlers[:]
    errhandler = vayesta.errhandler
    level = vayesta.log.level
    yield
    for hdl in vayesta.log.handlers[:]:
        if hdl not in handlers:
            vayesta.log.removeHandler(hdl)
            hdl.close()
    for hdl in handlers:
        if hdl not in vayesta.log.handlers:
            vayesta.log.addHandler(hdl)
    vayesta._handlers[:] = tracked
    vayesta.errhandler = errhandler
    vayesta.log.setLevel(level)


def test_setup_logging_files(tmp_path, restore_logging):
    vayesta.setup_logging(
        log_level=logging.INFO, logfile="log.txt", errfile="err.txt", quiet=True, output_dir=str(tmp_path)
    )
    vayesta.log.info("info message")
    vayesta.log.warning("warning message")
    for hdl in vayesta._handlers:
        hdl.flush()
    log = (tmp_path / "log.txt").read_text()
    assert "info message" in log
    assert "warning message" in log
    # The error log only contains warnings and above
    err = (tmp_path / "err.txt").read_text()
    assert "info message" not in err
    assert "warning message" in err


def test_setup_logging_replaces_handlers(tmp_path, restore_logging):
    vayesta.setup_logging(logfile="log1.txt", errfile=False, quiet=False, output_dir=str(tmp_path))
    first = vayesta._handlers[:]
    assert len(first) == 3
    vayesta.setup_logging(logfile=False, errfile=False, quiet=True, output_dir=str(tmp_path))
    assert vayesta._handlers == []
    assert vayesta.errhandler is None
    assert not any(hdl in vayesta.log.handlers for hdl in first)
    # The NullHandler is kept
    assert any(isinstance(hdl, logging.NullHandler) for hdl in vayesta.log.handlers)


def test_setup_logging_level(restore_logging):
    vayesta.setup_logging(log_level=logging.DEBUG, logfile=False, errfile=False, quiet=True)
    assert vayesta.log.level == logging.DEBUG


def test_no_setup_logging_on_import():
    env = dict(os.environ, VAYESTA_SETUP_LOGGING="0")
    code = "import vayesta; print(len(vayesta._handlers), [type(h).__name__ for h in vayesta.log.handlers])"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0", "['NullHandler']"]
    # Nothing is logged to the terminal
    assert out.stderr == ""
