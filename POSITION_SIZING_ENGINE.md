# Dynamic Position Sizing Engine

## Overview

The Dynamic Position Sizing Engine implements comprehensive position sizing strategies including Kelly Criterion-based sizing, risk-adjusted position sizing, portfolio concentration monitoring, and risk budget management. This implementation addresses task 5.3 of the stock analysis system specification.

## Key Features

### Position Sizing Methods

1. **Kelly Criterion**
   - Optimal position sizing based on expected returns and volatility
   - Safety multiplier to reduce risk of ruin
   - Handles negative expected returns gracefully

2. **Volatility-Adjusted Sizing**
   - Inverse relationship to asset volatility
   - Target volatility approach
   - Consistent risk exposure across assets

3. **VaR-Based Sizing**
   - Position sizing based on Value at Risk
   - Target portfolio risk approach
   - Tail risk consideration

4. **Fixed Fractional**
   - Simple percentage-based allocation
   - Risk-adjusted based on volatility levels
   - Conservative approach for uncertain markets

### Portfolio Risk Budget Management

1. **Risk Parity**
   - Equal risk contribution from all assets
   - Optimization-based approach
   - Diversification maximization

2. **Equal Risk Contribution**
   - Iterative approach to risk parity
   - Faster computation than optimization
   - Suitable for frequent rebalancing

3. **Inverse Volatility Weighting**
   - Simple inverse volatility approach
   - Lower volatility assets get higher weights
   - Computationally efficient

### Concentration Risk Monitoring

1. **Comprehensive Concentration Metrics**
   - Herfindahl-Hirschman Index
   - Effective number of assets
   - Top-N concentration measures
   - Gini coefficient for inequality

2. **Multi-Level Risk Assessment**
   - 5-level concentration risk scale
   - Sector and industry concentration
   - Risk contribution concentration
   - Automated warnings and recommendations

### Advanced Features

1. **Market Impact Analysis**
   - Volume participation calculation
   - Non-linear impact modeling
   - Multi-day execution planning
   - Transaction cost estimation

2. **Rebalancing Optimization**
   - Optimal frequency calculation
   - Transaction cost consideration
   - Priority-based recommendations
   - Net benefit analysis

3. **Strategy Backtesting**
   - Historical performance evaluation
   - Multiple rebalancing frequencies
   - Risk-adjusted metrics
   - Drawdown analysis

## Usage

### Basic Position Sizing

```python
import asyncio
from stock_analysis_system.analysis.position_sizing_engine import (
    DynamicPositionSizingEngine, PositionSizingMethod
)

# Initialize engine
sizing_engine = DynamicPositionSizingEngine(
    default_risk_budget=0.02,    # 2% portfolio risk
    max_position_weight=0.20,    # 20% max position
    min_position_weight=0.01,    # 1% min position
    kelly_multiplier=0.25        # Conservative Kelly
)

# Calculate position size
async def calculate_position():
    recommendation = await sizing_engine.calculate_position_size(
        symbol='AAPL',
        price_data=price_data,
        portfolio_value=1000000,
        method=PositionSizingMethod.KELLY_CRITERION
    )
    
    print(f"Recommended weight: {recommendation.recommended_weight:.2%}")
    print(f"Dollar amount: ${recommendation.recommended_dollar_amount:,.0f}")
    print(f"Shares: {recommendation.recommended_shares:,}")
    print(f"Kelly fraction: {recommendation.kelly_fraction:.3f}")
    print(f"Confidence: {recommendation.confidence_level:.1%}")

asyncio.run(calculate_position())
```

### Portfolio Risk Budget Optimization

```python
from stock_analysis_system.analysis.position_sizing_engine import RiskBudgetMethod

async def optimize_portfolio():
    # Portfolio assets
    assets = {
        'AAPL': apple_price_data,
        'GOOGL': google_price_data,
        'MSFT': microsoft_price_data
    }
    
    # Optimize risk budget allocation
    risk_budget = await sizing_engine.optimize_portfolio_risk_budget(
        assets=assets,
        portfolio_value=2000000,
        risk_budget=0.025,  # 2.5% portfolio risk
        method=RiskBudgetMethod.RISK_PARITY
    )
    
    print("Optimal Allocation:")
    for asset, weight in risk_budget.asset_weights.items():
        risk_contrib = risk_budget.risk_contributions[asset]
        print(f"{asset}: {weight:.1%} weight, {risk_contrib:.1%} risk")
    
    print(f"Diversification ratio: {risk_budget.diversification_ratio:.2f}")

asyncio.run(optimize_portfolio())
```

### Concentration Risk Analysis

```python
async def analyze_concentration():
    portfolio_weights = {
        'AAPL': 0.25, 'GOOGL': 0.20, 'MSFT': 0.15,
        'TSLA': 0.15, 'NVDA': 0.10, 'META': 0.10, 'AMZN': 0.05
    }
    
    # Sector mapping
    asset_sectors = {
        'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
        'TSLA': 'Automotive', 'NVDA': 'Technology', 'META': 'Technology',
        'AMZN': 'E-commerce'
    }
    
    analysis = await sizing_engine.analyze_concentration_risk(
        portfolio_weights=portfolio_weights,
        asset_sectors=asset_sectors
    )
    
    print(f"Concentration level: {analysis.concentration_level.value}")
    print(f"Concentration score: {analysis.concentration_score:.1f}/100")
    print(f"Effective # assets: {analysis.effective_number_of_assets:.1f}")
    print(f"Max weight: {analysis.max_weight:.1%}")
    
    if analysis.sector_concentration:
        print("Sector concentration:")
        for sector, weight in analysis.sector_concentration.items():
            print(f"  {sector}: {weight:.1%}")
    
    for warning in analysis.concentration_warnings:
        print(f"Warning: {warning}")

asyncio.run(analyze_concentration())
```

### Rebalancing Recommendations

```python
async def generate_rebalancing():
    current_weights = {'AAPL': 0.30, 'GOOGL': 0.25, 'MSFT': 0.20, 'TSLA': 0.25}
    target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
    
    recommendations = await sizing_engine.generate_portfolio_recommendations(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=1500000,
        transaction_costs={'AAPL': 0.001, 'GOOGL': 0.001, 'MSFT': 0.001, 'TSLA': 0.002}
    )
    
    print("Trading Recommendations:")
    for asset, rec in recommendations.items():
        print(f"{asset}: {rec['action']} ${rec['dollar_amount']:,.0f} "
              f"(Cost: ${rec['transaction_cost']:.0f}, Priority: {rec['priority']}/10)")

asyncio.run(generate_rebalancing())
```

## Configuration Options

### Engine Parameters

```python
sizing_engine = DynamicPositionSizingEngine(
    default_risk_budget=0.02,        # Default portfolio risk budget (2%)
    max_position_weight=0.20,        # Maximum single position (20%)
    min_position_weight=0.01,        # Minimum position size (1%)
    kelly_multiplier=0.25,           # Kelly safety multiplier (25%)
    concentration_threshold=0.60     # Concentration warning threshold (60%)
)
```

### Position Sizing Methods

```python
from stock_analysis_system.analysis.position_sizing_engine import PositionSizingMethod

# Available methods
PositionSizingMethod.KELLY_CRITERION      # Optimal growth
PositionSizingMethod.VOLATILITY_ADJUSTED  # Inverse volatility
PositionSizingMethod.VAR_BASED           # VaR targeting
PositionSizingMethod.FIXED_FRACTIONAL    # Simple percentage
PositionSizingMethod.RISK_PARITY         # Equal risk contribution
PositionSizingMethod.MAXIMUM_DIVERSIFICATION  # Max diversification
```

### Risk Budget Methods

```python
from stock_analysis_system.analysis.position_sizing_engine import RiskBudgetMethod

# Available methods
RiskBudgetMethod.RISK_PARITY              # Equal risk contribution
RiskBudgetMethod.EQUAL_RISK              # Iterative equal risk
RiskBudgetMethod.INVERSE_VOLATILITY      # Inverse volatility weighting
RiskBudgetMethod.HIERARCHICAL_RISK_PARITY  # Hierarchical approach
```

## Data Structures

### PositionSizeRecommendation

```python
@dataclass
class PositionSizeRecommendation:
    symbol: str
    recommended_weight: float              # Portfolio weight (0-1)
    recommended_shares: Optional[int]      # Number of shares
    recommended_dollar_amount: Optional[float]  # Dollar amount
    
    # Sizing rationale
    method_used: PositionSizingMethod
    kelly_fraction: Optional[float]
    risk_contribution: Optional[float]
    
    # Risk metrics
    expected_return: Optional[float]
    volatility: Optional[float]
    var_95: Optional[float]
    max_loss_estimate: Optional[float]
    
    # Constraints and confidence
    min_weight_constraint: Optional[float]
    max_weight_constraint: Optional[float]
    confidence_level: Optional[float]
    warnings: List[str]
```

### PortfolioRiskBudget

```python
@dataclass
class PortfolioRiskBudget:
    total_risk_budget: float               # Total portfolio risk
    asset_risk_budgets: Dict[str, float]   # Risk budget per asset
    asset_weights: Dict[str, float]        # Optimal weights
    risk_contributions: Dict[str, float]   # Actual risk contributions
    
    # Portfolio metrics
    budget_utilization: float              # Budget usage (0-1)
    diversification_ratio: float           # Diversification measure
    concentration_metrics: Dict[str, float]  # Concentration measures
    
    method_used: RiskBudgetMethod
    constraints_applied: List[str]
```

### ConcentrationRiskAnalysis

```python
@dataclass
class ConcentrationRiskAnalysis:
    concentration_level: ConcentrationRiskLevel  # LOW/MODERATE/HIGH/EXTREME
    concentration_score: float             # 0-100 scale
    
    # Concentration metrics
    herfindahl_index: float               # Sum of squared weights
    effective_number_of_assets: float     # 1/HHI
    max_weight: float                     # Largest position
    top_5_concentration: float            # Top 5 positions weight
    
    # Sector/industry analysis
    sector_concentration: Optional[Dict[str, float]]
    industry_concentration: Optional[Dict[str, float]]
    
    # Warnings and recommendations
    concentration_warnings: List[str]
    diversification_recommendations: List[str]
```

## Utility Functions

### Optimal Rebalancing Frequency

```python
from stock_analysis_system.analysis.position_sizing_engine import calculate_optimal_rebalancing_frequency

frequency = calculate_optimal_rebalancing_frequency(
    portfolio_weights={'AAPL': 0.4, 'GOOGL': 0.6},
    transaction_costs={'AAPL': 0.001, 'GOOGL': 0.001},
    volatilities={'AAPL': 0.25, 'GOOGL': 0.30}
)

print(f"Optimal rebalancing frequency: {frequency} days")
```

### Market Impact Analysis

```python
from stock_analysis_system.analysis.position_sizing_engine import calculate_position_size_impact

impact = calculate_position_size_impact(
    position_size=10000,      # 10,000 shares
    daily_volume=100000,      # 100,000 daily volume
    price=50.0,               # $50 per share
    participation_rate=0.1    # 10% max participation
)

print(f"Volume participation: {impact['volume_participation']:.1%}")
print(f"Market impact: {impact['market_impact_pct']:.2%}")
print(f"Impact cost: ${impact['impact_cost_dollars']:,.0f}")
print(f"Days to trade: {impact['days_to_trade']}")
```

### Strategy Backtesting

```python
from stock_analysis_system.analysis.position_sizing_engine import backtest_position_sizing_strategy

# Historical data for multiple assets
historical_data = {
    'AAPL': apple_historical_data,
    'GOOGL': google_historical_data,
    'MSFT': microsoft_historical_data
}

# Run backtest
results = await backtest_position_sizing_strategy(
    sizing_engine=sizing_engine,
    historical_data=historical_data,
    initial_capital=1000000,
    rebalance_frequency=30  # Monthly rebalancing
)

print(f"Total return: {results['total_return']:.1%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Max drawdown: {results['max_drawdown']:.1%}")
```

## Kelly Criterion Implementation

### Mathematical Foundation

The Kelly Criterion calculates the optimal position size as:

```
f* = (bp - q) / b
```

Where:
- f* = fraction of capital to wager
- b = odds received (return if win)
- p = probability of winning
- q = probability of losing (1-p)

For continuous returns, this becomes:

```
f* = (μ - r) / σ²
```

Where:
- μ = expected return
- r = risk-free rate
- σ² = variance of returns

### Safety Modifications

The engine implements several safety modifications:

1. **Kelly Multiplier**: Reduces position size to avoid over-leveraging
2. **Minimum Data Requirements**: Ensures statistical significance
3. **Negative Return Handling**: Avoids short positions
4. **Position Limits**: Caps maximum position size
5. **Confidence Intervals**: Provides uncertainty estimates

## Risk Parity Implementation

### Optimization Approach

Risk parity seeks to equalize risk contributions:

```
RC_i = w_i * (Σw)_i / σ_p
```

Where:
- RC_i = risk contribution of asset i
- w_i = weight of asset i
- (Σw)_i = marginal contribution to portfolio risk
- σ_p = portfolio volatility

The optimization minimizes:

```
Σ(RC_i - 1/n)²
```

Subject to:
- Σw_i = 1 (weights sum to 1)
- w_i ≥ 0 (long-only positions)
- w_i ≤ max_weight (position limits)

### Iterative Approach

For faster computation, the engine also implements an iterative approach:

1. Start with inverse volatility weights
2. Calculate risk contributions
3. Adjust weights to equalize contributions
4. Repeat until convergence

## Concentration Risk Metrics

### Herfindahl-Hirschman Index (HHI)

```
HHI = Σw_i²
```

- Range: 1/n to 1
- Lower values indicate better diversification
- 1/n = perfect diversification
- 1 = complete concentration

### Effective Number of Assets

```
N_eff = 1 / HHI
```

- Represents the number of equally-weighted assets
- Higher values indicate better diversification

### Concentration Score Calculation

The concentration score (0-100) combines multiple metrics:

1. **HHI Component (40%)**: Scaled HHI contribution
2. **Max Weight Component (30%)**: Largest position impact
3. **Top-N Component (30%)**: Concentration in top positions

## Performance Considerations

### Optimization Tips

1. **Data Requirements**
   - Minimum 30 data points for reliable estimates
   - More data improves Kelly Criterion accuracy
   - Consider data quality and recency

2. **Computational Efficiency**
   - Use inverse volatility for quick estimates
   - Cache covariance matrix calculations
   - Parallel processing for large portfolios

3. **Rebalancing Frequency**
   - Balance transaction costs vs. drift
   - Consider market volatility
   - Use optimal frequency calculation

### Memory Management

- Efficient matrix operations using NumPy
- Avoid storing unnecessary historical data
- Use generators for large backtests

## Integration with Other Components

### Risk Management Engine Integration

```python
# Use risk metrics from risk management engine
from stock_analysis_system.analysis.risk_management_engine import EnhancedRiskManagementEngine

risk_engine = EnhancedRiskManagementEngine()
risk_metrics = await risk_engine.calculate_comprehensive_risk_metrics(price_data)

# Use in position sizing
recommendation = await sizing_engine.calculate_position_size(
    symbol='AAPL',
    price_data=price_data,
    portfolio_value=1000000,
    risk_metrics={
        'expected_return': risk_metrics.sharpe_ratio * risk_metrics.volatility_results['historical'].annualized_volatility,
        'volatility': risk_metrics.volatility_results['historical'].annualized_volatility,
        'var_95': risk_metrics.var_results['historical'].var_95
    }
)
```

### Spring Festival Engine Integration

```python
# Adjust position sizes based on seasonal risk
if spring_festival_risk_adjustment > 1.2:  # High seasonal risk
    # Reduce position sizes
    adjusted_engine = DynamicPositionSizingEngine(
        max_position_weight=0.10,  # Reduce from 20% to 10%
        kelly_multiplier=0.15      # More conservative Kelly
    )
```

## Testing

### Run Tests
```bash
python -m pytest tests/test_position_sizing_engine.py -v
```

### Run Demo
```bash
python test_position_sizing_demo.py
```

## Requirements Addressed

This implementation addresses the following requirements:

- **4.1**: Kelly Criterion-based position sizing with risk management
- **4.2**: Risk-adjusted position sizing with multiple factor adjustments
- **4.3**: Portfolio concentration risk monitoring and analysis
- **4.4**: Position sizing recommendations with risk budget management
- **4.5**: Dynamic stop-loss levels and risk-based position adjustments

## Future Enhancements

1. **Advanced Optimization**
   - Black-Litterman model integration
   - Robust optimization techniques
   - Multi-objective optimization

2. **Machine Learning Integration**
   - ML-based return predictions
   - Dynamic risk model updates
   - Regime-aware position sizing

3. **Alternative Risk Measures**
   - Expected Shortfall optimization
   - Drawdown-based sizing
   - Tail risk parity

4. **Real-time Features**
   - Streaming position updates
   - Dynamic rebalancing triggers
   - Live risk monitoring

## Dependencies

- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0 (for optimization)
- asyncio (built-in)
- dataclasses (built-in)