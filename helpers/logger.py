"""Simple logging helper for the project."""

import logging
import os

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()

# set a console logging config
logging.basicConfig(
    level=LOGLEVEL,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# suppress these libraries anyways, they are often very verbose even at INFO level
logging.getLogger("matplotlib").setLevel(logging.WARNING)
