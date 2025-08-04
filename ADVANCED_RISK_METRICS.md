# Advanced Risk Metrics Calculator

## Overview

The Advanced Risk Metrics Calculator extends the basic risk management engine with enhanced risk metrics including confidence intervals, market risk assessment, liquidity risk scoring, and seasonal risk integration with Spring Festival analysis. This implementation addresses task 5.2 of the stock analysis system specification.

## Key Features

### Enhanced Risk-Adjusted Return Metrics

1. **Sharpe Ratio with Confidence Intervals**
   - Bootstrap-based confidence intervals
   - Statistical significance assessment
   - Uncertainty quantification

2. **Sortino Ratio with Confidence Intervals**
   - Downside risk focus
   - Bootstrap confidence intervals
   - Better for asymmetric return distributions

3. **Calmar Ratio with Confidence Intervals**
   - Return to maximum drawdown ratio
   - Rolling window bootstrap analysis
   - Drawdown-focused risk assessment

### Market Risk Metrics

1. **Beta Calculation**
   - Systematic risk measurement
   - Confidence intervals via bootstrap
   - Market sensitivity assessment

2. **Jensen's Alpha**
   - Risk-adjusted excess return
   - Outperformance measurement
   - CAPM-based analysis

3. **Tracking Error**
   - Active risk measurement
   - Benchmark deviation quantification

4. **Information Ratio**
   - Risk-adjusted active return
   - Manager skill assessment

### Liquidity Risk Assessment

1. **Comprehensive Liquidity Scoring (0-100 scale)**
   - Volume-based metrics
   - Amihud illiquidity measure
   - Bid-ask spread proxies
   - Market impact assessment

2. **Multi-Factor Liquidity Analysis**
   - Volume volatility patterns
   - Zero-volume day analysis
   - Price impact measurement
   - Trading cost estimation

### Seasonal Risk Integration

1. **Spring Festival Risk Analysis**
   - Proximity-based risk adjustment
   - Historical volatility patterns
   - Seasonal risk scoring

2. **Monthly Volatility Patterns**
   - Historical seasonal analysis
   - Risk level categorization
   - Temporal risk factors

### Additional Advanced Metrics

1. **Omega Ratio**
   - Probability-weighted performance
   - Gain/loss ratio analysis

2. **Kappa 3**
   - Skewness-adjusted risk measure
   - Third moment incorporation

3. **Tail Ratio**
   - Extreme return analysis
   - Fat tail assessment

## Usage

### Basic Advanced Metrics Calculation

```python
import asyncio
from stock_analysis_system.analysis.advanced_risk_metrics import AdvancedRiskMetricsCalculator

# Initialize calculator
calculator = AdvancedRiskMetricsCalculator(
    risk_free_rate=0.03,
    confidence_level=0.95,
    bootstrap_iterations=1000,
    seasonal_window_years=5
)

# Calculate advanced metrics
async def calculate_metrics():
    advanced_metrics = await calculator.calculate_advanced_metrics(
        price_data=price_data,
        benchmark_data=benchmark_data,
        volume_data=volume_data,
        spring_festival_engine=sf_engine
    )
    
    # Access enhanced metrics
    print(f"Sharpe Ratio: {advanced_metrics.sharpe_ratio:.4f}")
    if advanced_metrics.sharpe_confidence_interval:
        ci_lower, ci_upper = advanced_metrics.sharpe_confidence_interval
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"Beta: {advanced_metrics.beta:.4f}")
    print(f"Liquidity Risk Score: {advanced_metrics.liquidity_risk_score:.1f}/100")
    print(f"Seasonal Risk Score: {advanced_metrics.seasonal_risk_score:.1f}/100")

asyncio.run(calculate_metrics())
```

### Comprehensive Risk Profile

```python
from stock_analysis_system.analysis.advanced_risk_metrics import calculate_comprehensive_risk_profile

async def get_risk_profile():
    risk_profile = await calculate_comprehensive_risk_profile(
        price_data=price_data,
        benchmark_data=benchmark_data,
        volume_data=volume_data,
        spring_festival_engine=sf_engine,
        risk_engine=None  # Will create default if None
    )
    
    # Access comprehensive analysis
    summary = risk_profile['summary']
    print(f"Overall Risk Score: {summary['overall_risk_score']:.1f}/100")
    print(f"Risk Level: {summary['risk_level']}")
    
    print("Key Risk Factors:")
    for factor in summary['key_risk_factors']:
        print(f"  • {factor}")
    
    print("Recommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")

asyncio.run(get_risk_profile())
```

## Configuration Options

### Calculator Parameters

```python
calculator = AdvancedRiskMetricsCalculator(
    risk_free_rate=0.03,           # Annual risk-free rate
    confidence_level=0.95,         # Confidence level for intervals
    bootstrap_iterations=1000,     # Bootstrap samples
    seasonal_window_years=5        # Years for seasonal analysis
)
```

### Risk Adjustment Methods

```python
from stock_analysis_system.analysis.advanced_risk_metrics import RiskAdjustmentMethod

# Available methods
RiskAdjustmentMethod.STANDARD    # Mean and standard deviation
RiskAdjustmentMethod.ROBUST      # Median and MAD
RiskAdjustmentMethod.BOOTSTRAP   # Bootstrap confidence intervals
```

## Data Requirements

### Price Data
```python
price_data = pd.DataFrame({
    'date': pd.DatetimeIndex,
    'close': float,           # Required
    'high': float,            # Required for liquidity analysis
    'low': float,             # Required for liquidity analysis
    'open': float             # Optional
})
```

### Benchmark Data (Optional)
Same format as price data, used for beta, alpha, and tracking error calculations.

### Volume Data (Optional)
```python
volume_data = pd.DataFrame({
    'date': pd.DatetimeIndex,
    'volume': float           # Trading volume
})
```

## Output Structures

### AdvancedRiskMetrics

```python
@dataclass
class AdvancedRiskMetrics:
    # Enhanced ratio metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Confidence intervals
    sharpe_confidence_interval: Optional[Tuple[float, float]]
    sortino_confidence_interval: Optional[Tuple[float, float]]
    calmar_confidence_interval: Optional[Tuple[float, float]]
    
    # Market risk metrics
    beta: Optional[float]
    beta_confidence_interval: Optional[Tuple[float, float]]
    alpha: Optional[float]                    # Jensen's alpha
    tracking_error: Optional[float]
    information_ratio: Optional[float]
    
    # Liquidity risk metrics
    liquidity_risk_score: float               # 0-100 scale
    liquidity_risk_level: str                 # Categorical
    bid_ask_spread_proxy: Optional[float]
    market_impact_score: Optional[float]
    
    # Seasonal risk metrics
    seasonal_risk_score: float                # 0-100 scale
    seasonal_risk_level: SeasonalRiskLevel
    spring_festival_risk_adjustment: float   # Risk multiplier
    historical_seasonal_volatility: Dict[str, float]
    
    # Additional metrics
    omega_ratio: Optional[float]
    kappa_3: Optional[float]
    tail_ratio: Optional[float]
```

## Seasonal Risk Levels

```python
class SeasonalRiskLevel(str, Enum):
    VERY_LOW = "very_low"      # Score: 0-30
    LOW = "low"                # Score: 30-45
    MODERATE = "moderate"      # Score: 45-65
    HIGH = "high"              # Score: 65-80
    VERY_HIGH = "very_high"    # Score: 80-100
```

## Liquidity Risk Assessment

### Scoring Components

1. **Volume Volatility (25%)**
   - Consistency of trading volume
   - Higher volatility = higher risk

2. **Amihud Illiquidity Measure (25%)**
   - Price impact per dollar volume
   - Higher impact = higher risk

3. **Bid-Ask Spread Proxy (20%)**
   - High-low spread relative to price
   - Wider spreads = higher risk

4. **Zero Volume Days (15%)**
   - Frequency of no-trading days
   - More zero days = higher risk

5. **Market Impact Score (15%)**
   - Price volatility relative to volume
   - Higher impact = higher risk

### Liquidity Risk Levels

- **0-20**: Very High Liquidity
- **20-40**: High Liquidity  
- **40-60**: Moderate Liquidity
- **60-80**: Low Liquidity
- **80-100**: Very Low Liquidity

## Spring Festival Risk Integration

### Risk Adjustment Periods

1. **Peak Risk Period (-5 to +10 days)**
   - Risk multiplier: 1.5x
   - Seasonal score: 85/100

2. **Elevated Risk Period (-15 to +30 days)**
   - Risk multiplier: 1.2x
   - Seasonal score: 70/100

3. **Moderate Risk Period (-45 to +60 days)**
   - Risk multiplier: 1.1x
   - Seasonal score: 60/100

4. **Normal Period (other times)**
   - Risk multiplier: 1.0x
   - Seasonal score: 45/100

### Historical Analysis

The system analyzes historical volatility patterns around Spring Festival dates to:
- Identify recurring seasonal patterns
- Adjust current risk assessments
- Provide context for seasonal risk scores

## Confidence Intervals

### Bootstrap Methodology

1. **Sample Generation**
   - Random sampling with replacement
   - Multiple bootstrap iterations (default: 1000)
   - Preserves original data characteristics

2. **Metric Calculation**
   - Calculate metric for each bootstrap sample
   - Build distribution of metric values
   - Extract percentiles for confidence intervals

3. **Interpretation**
   - Wider intervals indicate higher uncertainty
   - Intervals not containing zero suggest significance
   - Useful for comparing strategies

### Example Interpretation

```python
sharpe_ratio = 1.2
confidence_interval = (0.8, 1.6)

# Interpretation:
# - Point estimate: 1.2
# - 95% confident true value is between 0.8 and 1.6
# - Since interval doesn't contain 0, likely positive performance
# - Width of 0.8 indicates moderate uncertainty
```

## Integration with Other Components

### Spring Festival Engine Integration

```python
# Automatic integration when engine is provided
advanced_metrics = await calculator.calculate_advanced_metrics(
    price_data=price_data,
    spring_festival_engine=sf_engine  # Enables seasonal analysis
)

# Access seasonal insights
print(f"Current SF risk adjustment: {advanced_metrics.spring_festival_risk_adjustment:.2f}x")
print(f"Seasonal risk level: {advanced_metrics.seasonal_risk_level.value}")
```

### Risk Management Engine Integration

```python
# Comprehensive analysis combining basic and advanced metrics
risk_profile = await calculate_comprehensive_risk_profile(
    price_data=price_data,
    benchmark_data=benchmark_data,
    volume_data=volume_data,
    spring_festival_engine=sf_engine,
    risk_engine=basic_risk_engine
)

# Compare basic vs advanced metrics
basic = risk_profile['basic_metrics']
advanced = risk_profile['advanced_metrics']
```

## Performance Considerations

### Optimization Tips

1. **Bootstrap Iterations**
   - Reduce iterations for faster calculation
   - Increase for more precise confidence intervals
   - Default 1000 provides good balance

2. **Data Window Sizes**
   - Larger windows provide more stable estimates
   - Smaller windows capture recent changes
   - Adjust based on data availability

3. **Caching**
   - Cache Spring Festival dates
   - Store intermediate calculations
   - Reuse bootstrap samples when possible

## Testing

### Run Tests
```bash
python -m pytest tests/test_advanced_risk_metrics.py -v
```

### Run Demo
```bash
python test_advanced_risk_metrics_demo.py
```

## Requirements Addressed

This implementation addresses the following requirements:

- **4.1**: Enhanced Sharpe, Sortino, and Calmar ratio calculations
- **4.2**: Beta calculation and market risk assessment  
- **4.3**: Liquidity risk scoring based on volume patterns
- **4.4**: Seasonal risk scoring integration with Spring Festival analysis
- **4.5**: Confidence intervals and statistical significance testing

## Future Enhancements

1. **Additional Risk Metrics**
   - Maximum Drawdown Duration
   - Ulcer Index
   - Pain Index

2. **Advanced Statistical Methods**
   - Robust statistics (Huber M-estimators)
   - Non-parametric confidence intervals
   - Bayesian risk estimation

3. **Multi-Asset Extensions**
   - Portfolio-level advanced metrics
   - Cross-asset correlations
   - Regime-dependent analysis

4. **Real-time Integration**
   - Streaming risk calculations
   - Dynamic confidence intervals
   - Live seasonal adjustments