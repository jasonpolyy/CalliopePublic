import importlib.metadata

__version__ = importlib.metadata.version("calliope")

from calliope import optimisation, prices, defaults, market, market_model
