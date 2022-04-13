import os
import logging
from ._version import __version__

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(format="%(levelname)s:%(module)s:%(funcName)s:%(lineno)d - %(message)s", level=LOGLEVEL)
