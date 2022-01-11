import numpy as np

from vayesta.core import UEmbedding
from vayesta.core.util import *

from vayesta.dmet import UDMET
from vayesta.edmet import REDMET
from vayesta.edmet.ufragment import UEDMETFragment as Fragment


class UEDMET(REDMET, UDMET):
    Fragment = Fragment