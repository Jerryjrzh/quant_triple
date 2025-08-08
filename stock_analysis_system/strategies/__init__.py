"""
Quantitative Strategy Module

This module provides comprehensive quantitative trading strategies including:
- Technical indicator strategies
- Multi-factor strategies
- Strategy templates and builders
- Advanced backtesting capabilities
"""

from .technical_indicators import TechnicalIndicatorLibrary
from .strategy_templates import StrategyTemplateManager
from .multi_factor_strategy import MultiFactorStrategy
from .strategy_optimizer import StrategyOptimizer

__all__ = [
    'TechnicalIndicatorLibrary',
    'StrategyTemplateManager',
    'MultiFactorStrategy',
    'StrategyOptimizer'
]