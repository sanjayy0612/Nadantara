"""Dataset readers used by training and data inspection tools."""

from seamtranslation.datasets.indicvoices import IndicVoicesExample, iter_indicvoices
from seamtranslation.datasets.samanantar import ParallelTextExample, iter_samanantar
from seamtranslation.datasets.wordproject import WordProjectExample, iter_wordproject

__all__ = [
    "IndicVoicesExample",
    "ParallelTextExample",
    "WordProjectExample",
    "iter_indicvoices",
    "iter_samanantar",
    "iter_wordproject",
]
