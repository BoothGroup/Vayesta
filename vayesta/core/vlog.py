"""Vayesta logging module"""

import logging
import os
import contextlib

from vayesta.core.mpi import mpi

"""
Log levels (* are non-standard):

Name            Level           Usage
----            -----           -----
CRITICAL        50              For immediate, non-recoverable errors
ERROR           40              For errors which are likely non-recoverable
WARNING         30              For possible errors and important information
OUTPUT  (*)     25              Main result level - the only level which by default gets streamed to stdout
INFO            20              Information, readable to users
INFOV   (*)     15  (-v)        Verbose information, readable to users
TIMING  (*)     12  (-vv)       Timing information for primary routines
DEBUG           10  (-vv)       Debugging information, indented for developers
DEBUGV  (*)      5  (-vvv)      Verbose debugging information
TIMINGV (*)      2  (-vvv)      Verbose timings information for secondary subroutines
"""

LVL_PREFIX = {
   "CRITICAL" : "CRITICAL",
   "ERROR" : "ERROR",
   "WARNING" : "WARNING",
   "OUT" : "OUTPUT",
   "DEBUGV" : "DEBUG",
   }


class LevelRangeFilter(logging.Filter):
    """Only log events with level in interval [low, high)."""

    def __init__(self, *args, low=None, high=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._low = low
        self._high = high


    def filter(self, record):
        if self._low is None:
            return (record.levelno < self._high)
        if self._high is None:
            return (self._low <= record.levelno)
        return (self._low <= record.levelno < self._high)


class LevelIncludeFilter(logging.Filter):
    """Only log events with level in include."""

    def __init__(self, *args, include, **kwargs):
        super().__init__(*args, **kwargs)
        self._include = include


    def filter(self, record):
        return (record.levelno in self._include)


class LevelExcludeFilter(logging.Filter):
    """Only log events with level not in exlude."""

    def __init__(self, *args, exclude, **kwargs):
        super().__init__(*args, **kwargs)
        self._exclude = exclude


    def filter(self, record):
        return (record.levelno not in self._exclude)


class VFormatter(logging.Formatter):
    """Formatter which adds a prefix column and indentation."""

    def __init__(self, *args, prefix=True, prefix_width=10, prefix_sep='|',
            indent=False, indent_char=' ', indent_width=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix
        self.prefix_width = prefix_width
        self.prefix_sep = prefix_sep
        self.indent = indent
        self.indent_char = indent_char
        self.indent_width = indent_width

    def format(self, record):
        message = record.msg % record.args
        indent = prefix = ""
        if self.prefix:
            prefix = LVL_PREFIX.get(record.levelname, "")
            if prefix:
                prefix = "[%s]" % prefix
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

    def __init__(self, filename, mode='a', formatter=None, **kwargs):
        filename = get_logname(filename)
        super().__init__(filename, mode=mode, **kwargs)
        if formatter is None:
            formatter = VFormatter()
        self.setFormatter(formatter)

def get_logname(basename, ext='log'):
    if ext and '.' not in basename:
        ext = '.' + ext
    else:
        ext = ''
    name = '%s%s%s' % (basename, (('.mpi%d' % mpi.rank) if mpi.rank > 0 else ''), ext)
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
    def add_log_level(level, name):
        logging.addLevelName(level, name.upper())
        setattr(logging, name.upper(), level)
        def logForLevel(self, message, *args, **kwargs):
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)
        def logToRoot(message, *args, **kwargs):
            logging.log(level, message, *args, **kwargs)
        setattr(logging.getLoggerClass(), name, logForLevel)
        setattr(logging, name, logToRoot)
    add_log_level(25, "output")
    add_log_level(15, "infov")
    add_log_level(12, "timing")
    add_log_level(5, "debugv")
    add_log_level(2, "timingv")

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

    class withIndentLevel(contextlib.ContextDecorator):
        def __init__(self, delta):
            self.delta = delta
            self.root = logging.getLogger()

        def __enter__(self):
            self.root.indentLevel = max(self.root.indentLevel + self.delta, 0)

        def __exit__(self, *args):
            self.root.indentLevel = max(self.root.indentLevel - self.delta, 0)

    logging.Logger.setIndentLevel = setIndentLevel
    logging.Logger.changeIndentLevel = changeIndentLevel
    logging.Logger.withIndentLevel = withIndentLevel
