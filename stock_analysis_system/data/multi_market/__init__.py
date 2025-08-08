"""
Multi-Market Data Support Module

This module provides support for multiple stock markets including:
- Hong Kong Stock Exchange (HKEX)
- US Stock Markets (NYSE, NASDAQ)
- Cross-market data synchronization
- Currency conversion and timezone handling
"""

from .hk_adapter import HongKongStockAdapter
from .us_adapter import USStockAdapter
from .market_synchronizer import MultiMarketSynchronizer
from .currency_converter import CurrencyConverter

__all__ = [
    'HongKongStockAdapter',
    'USStockAdapter', 
    'MultiMarketSynchronizer',
    'CurrencyConverter'
]