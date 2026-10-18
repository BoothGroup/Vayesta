import logging

import numpy as np
import pytest

import vayesta


@pytest.fixture
def restore_logging():
    """Restore the handlers and level of the Vayesta logger and NumPy's print options after the test."""
    printoptions = np.get_printoptions()
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
    np.set_printoptions(**printoptions)


def test_setup_files(tmp_path, restore_logging):
    vayesta.setup(
        log_level=logging.INFO, logfile="log.txt", errfile="err.txt", quiet=True, output_dir=str(tmp_path), banner=False
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


def test_setup_replaces_handlers(tmp_path, restore_logging):
    vayesta.setup(logfile="log1.txt", errfile=False, quiet=False, output_dir=str(tmp_path), banner=False)
    first = vayesta._handlers[:]
    assert len(first) == 3
    vayesta.setup(logfile=False, errfile=False, quiet=True, output_dir=str(tmp_path), banner=False)
    assert vayesta._handlers == []
    assert vayesta.errhandler is None
    assert not any(hdl in vayesta.log.handlers for hdl in first)
    # The NullHandler is kept
    assert any(isinstance(hdl, logging.NullHandler) for hdl in vayesta.log.handlers)


def test_setup_level(restore_logging):
    vayesta.setup(log_level=logging.DEBUG, logfile=False, errfile=False, quiet=True, banner=False)
    assert vayesta.log.level == logging.DEBUG


def test_setup_disabled(capsys, caplog, restore_logging):
    vayesta.setup(logfile=False, errfile=False, quiet=True, banner=False)
    assert vayesta._handlers == []
    with caplog.at_level(logging.WARNING):
        vayesta.log.warning("a warning")
    # Nothing is logged to the terminal, but the message reaches the handlers of the program
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert [r.getMessage() for r in caplog.records] == ["a warning"]


def test_setup_banner(tmp_path, restore_logging):
    vayesta.setup(log_level=logging.DEBUG, logfile="log.txt", errfile=False, quiet=True, output_dir=str(tmp_path))
    for hdl in vayesta._handlers:
        hdl.flush()
    log = (tmp_path / "log.txt").read_text()
    assert "Required packages:" in log
    assert "PySCF" in log
    assert "Git hashes:" in log


def test_setup_print_options(restore_logging):
    np.set_printoptions(linewidth=75)
    vayesta.setup(logfile=False, errfile=False, quiet=True, banner=False)
    assert np.get_printoptions()["linewidth"] == 120


def test_new_log_without_errfile(tmp_path, restore_logging):
    vayesta.setup(logfile=False, errfile=False, quiet=True, banner=False)
    vayesta.new_log(str(tmp_path / "new.txt"))
    vayesta.log.info("in new log")
    for hdl in vayesta._handlers:
        hdl.flush()
    assert "in new log" in (tmp_path / "new.txt").read_text()
    # setup replaces the handler added by new_log
    vayesta.setup(logfile=False, errfile=False, quiet=True, banner=False)
    assert vayesta._handlers == []
