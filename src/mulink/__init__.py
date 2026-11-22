from importlib.metadata import version

from . import basic, simulate, validate

__version__ = version("mulink")

__all__ = ["basic", "simulate", "validate"]
