from importlib.metadata import version

from . import basic, simulate

__version__ = version("mulink")

__all__ = ["basic", "simulate"]
