"""
Screening Criteria Classes

This module defines various screening criteria types and a builder pattern
for creating complex screening conditions.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from datetime import datetime, date


class ComparisonOperator(str, Enum):
    """Comparison operators for screening criteria."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"


class LogicalOperator(str, Enum):
    """Logical operators for combining criteria."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class BaseCriteria:
    """Base class for all screening criteria."""
    name: str
    description: str
    enabled: bool = True
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'weight': self.weight,
            'type': self.__class__.__name__
        }


@dataclass
class TechnicalCriteria(BaseCriteria):
    """Technical analysis based screening criteria."""
    
    # Price-based criteria
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    price_change_pct_min: Optional[float] = None
    price_change_pct_max: Optional[float] = None
    
    # Volume criteria
    volume_min: Optional[int] = None
    volume_avg_ratio_min: Optional[float] = None  # Current volume / 20-day avg
    
    # Moving averages
    ma5_position: Optional[str] = None  # "above", "below", "cross_up", "cross_down"
    ma10_position: Optional[str] = None
    ma20_position: Optional[str] = None
    ma50_position: Optional[str] = None
    ma200_position: Optional[str] = None
    
    # Technical indicators
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    macd_signal: Optional[str] = None  # "bullish", "bearish", "neutral"
    bollinger_position: Optional[str] = None  # "upper", "lower", "middle"
    
    # Momentum indicators
    momentum_days: int = 20
    momentum_min: Optional[float] = None
    momentum_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert technical criteria to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'price_min': self.price_min,
            'price_max': self.price_max,
            'price_change_pct_min': self.price_change_pct_min,
            'price_change_pct_max': self.price_change_pct_max,
            'volume_min': self.volume_min,
            'volume_avg_ratio_min': self.volume_avg_ratio_min,
            'ma5_position': self.ma5_position,
            'ma10_position': self.ma10_position,
            'ma20_position': self.ma20_position,
            'ma50_position': self.ma50_position,
            'ma200_position': self.ma200_position,
            'rsi_min': self.rsi_min,
            'rsi_max': self.rsi_max,
            'macd_signal': self.macd_signal,
            'bollinger_position': self.bollinger_position,
            'momentum_days': self.momentum_days,
            'momentum_min': self.momentum_min,
            'momentum_max': self.momentum_max
        })
        return base_dict


@dataclass
class SeasonalCriteria(BaseCriteria):
    """Spring Festival and seasonal pattern based criteria."""
    
    # Spring Festival position
    spring_festival_days_range: Optional[tuple] = None  # (min_days, max_days) from SF
    spring_festival_pattern_strength: Optional[float] = None  # 0-1 scale
    spring_festival_historical_performance: Optional[str] = None  # "strong", "weak", "neutral"
    
    # Seasonal patterns
    current_season_strength: Optional[float] = None
    historical_seasonal_rank: Optional[int] = None  # 1-12 for months
    
    # Pattern matching
    pattern_similarity_threshold: Optional[float] = None
    pattern_confidence_min: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert seasonal criteria to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'spring_festival_days_range': self.spring_festival_days_range,
            'spring_festival_pattern_strength': self.spring_festival_pattern_strength,
            'spring_festival_historical_performance': self.spring_festival_historical_performance,
            'current_season_strength': self.current_season_strength,
            'historical_seasonal_rank': self.historical_seasonal_rank,
            'pattern_similarity_threshold': self.pattern_similarity_threshold,
            'pattern_confidence_min': self.pattern_confidence_min
        })
        return base_dict


@dataclass
class InstitutionalCriteria(BaseCriteria):
    """Institutional fund activity based criteria."""
    
    # Institutional attention
    attention_score_min: Optional[float] = None  # 0-100 scale
    attention_score_max: Optional[float] = None
    
    # Recent activity
    new_institutional_entry: bool = False
    dragon_tiger_appearances: Optional[int] = None  # Min appearances in last N days
    dragon_tiger_days: int = 30
    
    # Fund types
    mutual_fund_activity: bool = False
    social_security_activity: bool = False
    qfii_activity: bool = False
    hot_money_activity: bool = False
    
    # Shareholding changes
    top10_shareholder_changes: bool = False
    institutional_ownership_min: Optional[float] = None  # Percentage
    institutional_ownership_max: Optional[float] = None
    
    # Block trades
    block_trade_activity: bool = False
    block_trade_volume_min: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert institutional criteria to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'attention_score_min': self.attention_score_min,
            'attention_score_max': self.attention_score_max,
            'new_institutional_entry': self.new_institutional_entry,
            'dragon_tiger_appearances': self.dragon_tiger_appearances,
            'dragon_tiger_days': self.dragon_tiger_days,
            'mutual_fund_activity': self.mutual_fund_activity,
            'social_security_activity': self.social_security_activity,
            'qfii_activity': self.qfii_activity,
            'hot_money_activity': self.hot_money_activity,
            'top10_shareholder_changes': self.top10_shareholder_changes,
            'institutional_ownership_min': self.institutional_ownership_min,
            'institutional_ownership_max': self.institutional_ownership_max,
            'block_trade_activity': self.block_trade_activity,
            'block_trade_volume_min': self.block_trade_volume_min
        })
        return base_dict


@dataclass
class RiskCriteria(BaseCriteria):
    """Risk management based screening criteria."""
    
    # Volatility measures
    volatility_min: Optional[float] = None
    volatility_max: Optional[float] = None
    volatility_days: int = 20
    
    # Value at Risk
    var_max: Optional[float] = None  # Maximum acceptable VaR
    var_confidence: float = 0.95
    
    # Risk ratios
    sharpe_ratio_min: Optional[float] = None
    sortino_ratio_min: Optional[float] = None
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    
    # Drawdown measures
    max_drawdown_max: Optional[float] = None
    current_drawdown_max: Optional[float] = None
    
    # Liquidity risk
    avg_daily_volume_min: Optional[int] = None
    bid_ask_spread_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk criteria to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'volatility_min': self.volatility_min,
            'volatility_max': self.volatility_max,
            'volatility_days': self.volatility_days,
            'var_max': self.var_max,
            'var_confidence': self.var_confidence,
            'sharpe_ratio_min': self.sharpe_ratio_min,
            'sortino_ratio_min': self.sortino_ratio_min,
            'beta_min': self.beta_min,
            'beta_max': self.beta_max,
            'max_drawdown_max': self.max_drawdown_max,
            'current_drawdown_max': self.current_drawdown_max,
            'avg_daily_volume_min': self.avg_daily_volume_min,
            'bid_ask_spread_max': self.bid_ask_spread_max
        })
        return base_dict


@dataclass
class ScreeningTemplate:
    """Template for saving and loading screening criteria combinations."""
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    technical_criteria: Optional[TechnicalCriteria] = None
    seasonal_criteria: Optional[SeasonalCriteria] = None
    institutional_criteria: Optional[InstitutionalCriteria] = None
    risk_criteria: Optional[RiskCriteria] = None
    logical_operator: LogicalOperator = LogicalOperator.AND
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'technical_criteria': self.technical_criteria.to_dict() if self.technical_criteria else None,
            'seasonal_criteria': self.seasonal_criteria.to_dict() if self.seasonal_criteria else None,
            'institutional_criteria': self.institutional_criteria.to_dict() if self.institutional_criteria else None,
            'risk_criteria': self.risk_criteria.to_dict() if self.risk_criteria else None,
            'logical_operator': self.logical_operator.value,
            'tags': self.tags
        }


class ScreeningCriteriaBuilder:
    """Builder pattern for creating complex screening criteria."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder to start fresh."""
        self._technical = None
        self._seasonal = None
        self._institutional = None
        self._risk = None
        self._logical_operator = LogicalOperator.AND
        return self
    
    def with_technical_criteria(self, **kwargs) -> 'ScreeningCriteriaBuilder':
        """Add technical analysis criteria."""
        self._technical = TechnicalCriteria(
            name="Technical Analysis",
            description="Technical indicators and price action criteria",
            **kwargs
        )
        return self
    
    def with_seasonal_criteria(self, **kwargs) -> 'ScreeningCriteriaBuilder':
        """Add seasonal pattern criteria."""
        self._seasonal = SeasonalCriteria(
            name="Seasonal Patterns",
            description="Spring Festival and seasonal pattern criteria",
            **kwargs
        )
        return self
    
    def with_institutional_criteria(self, **kwargs) -> 'ScreeningCriteriaBuilder':
        """Add institutional activity criteria."""
        self._institutional = InstitutionalCriteria(
            name="Institutional Activity",
            description="Fund activity and institutional behavior criteria",
            **kwargs
        )
        return self
    
    def with_risk_criteria(self, **kwargs) -> 'ScreeningCriteriaBuilder':
        """Add risk management criteria."""
        self._risk = RiskCriteria(
            name="Risk Management",
            description="Risk metrics and volatility criteria",
            **kwargs
        )
        return self
    
    def with_logical_operator(self, operator: LogicalOperator) -> 'ScreeningCriteriaBuilder':
        """Set the logical operator for combining criteria."""
        self._logical_operator = operator
        return self
    
    def build_template(self, name: str, description: str, tags: List[str] = None) -> ScreeningTemplate:
        """Build a screening template with current criteria."""
        now = datetime.now()
        return ScreeningTemplate(
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            technical_criteria=self._technical,
            seasonal_criteria=self._seasonal,
            institutional_criteria=self._institutional,
            risk_criteria=self._risk,
            logical_operator=self._logical_operator,
            tags=tags or []
        )
    
    def build_criteria_dict(self) -> Dict[str, Any]:
        """Build a dictionary representation of all criteria."""
        return {
            'technical': self._technical.to_dict() if self._technical else None,
            'seasonal': self._seasonal.to_dict() if self._seasonal else None,
            'institutional': self._institutional.to_dict() if self._institutional else None,
            'risk': self._risk.to_dict() if self._risk else None,
            'logical_operator': self._logical_operator.value
        }


# Predefined screening templates
class PredefinedTemplates:
    """Collection of predefined screening templates."""
    
    @staticmethod
    def growth_momentum_template() -> ScreeningTemplate:
        """Template for growth momentum stocks."""
        builder = ScreeningCriteriaBuilder()
        return builder.with_technical_criteria(
            price_change_pct_min=5.0,
            volume_avg_ratio_min=1.5,
            rsi_min=50.0,
            rsi_max=80.0,
            ma20_position="above"
        ).with_risk_criteria(
            volatility_max=0.4,
            sharpe_ratio_min=0.5
        ).build_template(
            name="Growth Momentum",
            description="Stocks with strong price momentum and reasonable risk",
            tags=["growth", "momentum", "technical"]
        )
    
    @staticmethod
    def spring_festival_opportunity_template() -> ScreeningTemplate:
        """Template for Spring Festival seasonal opportunities."""
        builder = ScreeningCriteriaBuilder()
        return builder.with_seasonal_criteria(
            spring_festival_days_range=(-30, 30),
            spring_festival_pattern_strength=0.7,
            pattern_confidence_min=0.6
        ).with_institutional_criteria(
            attention_score_min=60.0,
            new_institutional_entry=True
        ).build_template(
            name="Spring Festival Opportunity",
            description="Stocks with strong Spring Festival patterns and institutional interest",
            tags=["seasonal", "spring_festival", "institutional"]
        )
    
    @staticmethod
    def low_risk_value_template() -> ScreeningTemplate:
        """Template for low-risk value stocks."""
        builder = ScreeningCriteriaBuilder()
        return builder.with_technical_criteria(
            rsi_max=50.0,
            ma200_position="above"
        ).with_risk_criteria(
            volatility_max=0.25,
            beta_max=1.2,
            sharpe_ratio_min=0.3,
            max_drawdown_max=0.15
        ).build_template(
            name="Low Risk Value",
            description="Conservative value stocks with low volatility",
            tags=["value", "low_risk", "conservative"]
        )
    
    @staticmethod
    def institutional_following_template() -> ScreeningTemplate:
        """Template for following institutional activity."""
        builder = ScreeningCriteriaBuilder()
        return builder.with_institutional_criteria(
            attention_score_min=70.0,
            dragon_tiger_appearances=2,
            mutual_fund_activity=True,
            institutional_ownership_min=10.0
        ).with_technical_criteria(
            volume_avg_ratio_min=1.2,
            price_change_pct_min=0.0
        ).build_template(
            name="Institutional Following",
            description="Stocks with strong institutional activity and support",
            tags=["institutional", "smart_money", "activity"]
        )