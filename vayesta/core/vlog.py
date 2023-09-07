"""Vayesta logging module"""

import functools
import contextlib
import logging

from vayesta.mpi import mpi

"""
Log levels (* are custom levels):

Name            Level           Usage
----            -----           -----
FATAL   (*)     100             For errors which will raise a non-recoverable Exception
CRITICAL        50              For errors which will are non-recoverable
ERROR           40              For errors which are likely non-recoverable
WARNING         30              For possible errors and important information
DEPRECATED      30              For deprecated code
OUTPUT  (*)     25              Main result level - the only level which by default gets streamed to stdout
INFO            20              Information, readable to users
INFOV   (*)     15  (-v)        Verbose information, readable to users
TIMING  (*)     12  (-vv)       Timing information for primary routines
DEBUG           10  (-vv)       Debugging information, indented for developers
DEBUGV  (*)      5  (-vvv)      Verbose debugging information
TIMINGV (*)      2  (-vvv)      Verbose timings information for secondary subroutines
TRACE   (*)      1  (-vvv)      To trace function flow
"""

LVL_PREFIX = {
    "FATAL": "FATAL",
    "CRITICAL": "CRITICAL",
    "ERROR": "ERROR",
    "WARNING": "WARNING",
    "DEPRECATED": "WARNING",
    "OUT": "OUTPUT",
    "DEBUG": "DEBUG",
    "DEBUGV": "DEBUG",
    "TRACE": "TRACE",
}


class NoLogger:
    def __getattr__(self, key):
        """Return function which does nothing."""
        return lambda *args, **kwargs: None


class LevelRangeFilter(logging.Filter):
    """Only log events with level in interval [low, high)."""

    def __init__(self, *args, low=None, high=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._low = low
        self._high = high

    def filter(self, record):
        if self._low is None:
            return record.levelno < self._high
        if self._high is None:
            return self._low <= record.levelno
        return self._low <= record.levelno < self._high


class LevelIncludeFilter(logging.Filter):
    """Only log events with level in include."""

    def __init__(self, *args, include, **kwargs):
        super().__init__(*args, **kwargs)
        self._include = include

    def filter(self, record):
        return record.levelno in self._include


class LevelExcludeFilter(logging.Filter):
    """Only log events with level not in exlude."""

    def __init__(self, *args, exclude, **kwargs):
        super().__init__(*args, **kwargs)
        self._exclude = exclude

    def filter(self, record):
        return record.levelno not in self._exclude


class VFormatter(logging.Formatter):
    """Formatter which adds a prefix column and indentation."""

    def __init__(
        self,
        *args,
        show_level=True,
        show_mpi_rank=False,
        prefix_sep="|",
        indent=False,
        indent_char=" ",
        indent_width=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.show_level = show_level
        self.show_mpi_rank = show_mpi_rank

        self.prefix_width = 0
        if show_level:
            self.prefix_width += len(max(LVL_PREFIX.values(), key=len)) + 2
        if show_mpi_rank:
            self.prefix_width += len(str(mpi.size - 1)) + 6

        self.prefix_sep = prefix_sep
        self.indent = indent
        self.indent_char = indent_char
        self.indent_width = indent_width

    def format(self, record):
        message = record.msg % record.args
        indent = prefix = ""
        if self.show_level:
            prefix = LVL_PREFIX.get(record.levelname, "")
            if prefix:
                prefix = "[%s]" % prefix
        if self.show_mpi_rank:
            prefix += "[MPI %d]" % mpi.rank
        prefix = "%-*s%s" % (self.prefix_width, prefix, self.prefix_sep)
        if self.indent:
            root = logging.getLogger()
            indent = root.indentLevel * self.indent_width * self.indent_char
        lines = [indent + x for x in message.split("\n")]
        if prefix:
            lines = [((prefix + "  " + line) if line else prefix) for line in lines]
        return "\n".join(lines)


class VStreamHandler(logging.StreamHandler):
    """Default stream handler with IndentedFormatter"""

    def __init__(self, stream, formatter=None, **kwargs):
        super().__init__(stream, **kwargs)
        if formatter is None:
            formatter = VFormatter()
        self.setFormatter(formatter)


class VFileHandler(logging.FileHandler):
    """Default file handler with IndentedFormatter"""

    def __init__(self, filename, mode="a", formatter=None, add_mpi_rank=True, delay=True, **kwargs):
        filename = get_logname(filename, add_mpi_rank=add_mpi_rank)
        super().__init__(filename, mode=mode, delay=delay, **kwargs)
        if formatter is None:
            formatter = VFormatter()
        self.setFormatter(formatter)


def get_logname(name, add_mpi_rank=True, ext="txt"):
    if mpi and add_mpi_rank:
        name = "%s.mpi%d" % (name, mpi.rank)
    if ext and not name.endswith(".%s" % ext):
        name = "%s.%s" % (name, ext)
    return name


def init_logging():
    """Call this to initialize and configure logging, when importing Vayesta.

    This will:
    1) Add four new logging levels:
        `output`, `infov`, `timing`, `debugv`, and `timingv`.
    2) Adds the attribute `indentLevel` to the root logger and two new Logger methods:
        `setIndentLevel`, `changeIndentLevel`.
    """

    # Add new log levels
    # ------------------
    def add_log_level(level, name, log_once=False):
        logging.addLevelName(level, name.upper())
        setattr(logging, name.upper(), level)

        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)

        if log_once:
            logForLevel = functools.lru_cache(None)(logForLevel)

        def logToRoot(message, *args, **kwargs):
            logging.log(level, message, *args, **kwargs)

        setattr(logging.getLoggerClass(), name, logForLevel)
        setattr(logging, name, logToRoot)

    add_log_level(100, "fatal")
    add_log_level(30, "deprecated", log_once=True)
    add_log_level(25, "output")
    add_log_level(15, "infov")
    add_log_level(12, "timing")
    add_log_level(5, "debugv")
    add_log_level(2, "timingv")
    add_log_level(1, "trace")

    # Add indentation support
    # -----------------------
    # Note that indents are only tracked by the root logger
    root = logging.getLogger()
    root.indentLevel = 0

    def setIndentLevel(self, level):
        root = logging.getLogger()
        root.indentLevel = max(level, 0)
        return root.indentLevel

    def changeIndentLevel(self, delta):
        root = logging.getLogger()
        root.indentLevel = max(root.indentLevel + delta, 0)
        return root.indentLevel

    class indent(contextlib.ContextDecorator):
        def __init__(self, delta=1):
            self.delta = delta
            self.level_init = None
            self.root = logging.getLogger()

        def __enter__(self):
            self.level_init = self.root.indentLevel
            self.root.indentLevel = max(self.level_init + self.delta, 0)

        def __exit__(self, *args):
            self.root.indentLevel = self.level_init

    logging.Logger.setIndentLevel = setIndentLevel
    logging.Logger.changeIndentLevel = changeIndentLevel
    logging.Logger.indent = indent
    # Deprecated:
    logging.Logger.withIndentLevel = indent
