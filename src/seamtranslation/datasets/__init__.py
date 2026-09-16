"""Dataset readers used by training and data inspection tools."""

from seamtranslation.datasets.indicvoices import IndicVoicesExample, iter_indicvoices
from seamtranslation.datasets.samanantar import ParallelTextExample, iter_samanantar

__all__ = [
    "IndicVoicesExample",
    "ParallelTextExample",
    "iter_indicvoices",
    "iter_samanantar",
]
