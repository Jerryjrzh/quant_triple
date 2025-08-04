# Enhanced Risk Management Engine

## Overview

The Enhanced Risk Management Engine is a comprehensive risk assessment system that implements multiple Value at Risk (VaR) calculation methods, volatility measures, and risk metrics for stock analysis. This implementation addresses task 5.1 of the stock analysis system specification.

## Features

### VaR Calculation Methods

1. **Historical VaR**
   - Uses historical return distribution
   - Non-parametric approach
   - Captures actual market behavior including fat tails

2. **Parametric VaR**
   - Assumes normal distribution of returns
   - Fast calculation
   - Provides analytical solution

3. **Monte Carlo VaR**
   - Simulation-based approach
   - Includes confidence intervals
   - Flexible for complex scenarios

### Volatility Measures

1. **Historical Volatility**
   - Simple standard deviation of returns
   - Annualized using √252 factor

2. **EWMA (Exponentially Weighted Moving Average)**
   - Gives more weight to recent observations
   - Adapts quickly to changing market conditions

3. **GARCH(1,1)**
   - Models volatility clustering
   - Captures time-varying volatility

### Risk Metrics

- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Beta**: Systematic risk relative to benchmark
- **Liquidity Risk Score**: 0-100 scale liquidity assessment

## Usage

### Basic Usage

```python
import asyncio
from stock_analysis_system.analysis.risk_management_engine import EnhancedRiskManagementEngine
import pandas as pd

# Initialize the engine
risk_engine = EnhancedRiskManagementEngine(
    confidence_levels=[0.95, 0.99],
    var_window=252,
    volatility_window=30,
    monte_carlo_simulations=10000
)

# Prepare price data
price_data = pd.DataFrame({
    'date': pd.date_range('2022-01-01', periods=500),
    'close': [100 + i * 0.1 + np.random.normal(0, 2) for i in range(500)],
    'high': [...],  # High prices
    'low': [...],   # Low prices
    'open': [...]   # Open prices
})

# Calculate comprehensive risk metrics
async def calculate_risk():
    risk_metrics = await risk_engine.calculate_comprehensive_risk_metrics(
        price_data=price_data,
        benchmark_data=benchmark_data,  # Optional
        volume_data=volume_data         # Optional
    )
    
    # Access VaR results
    historical_var = risk_metrics.var_results['historical']
    print(f"95% VaR: {historical_var.var_95:.4f}")
    print(f"99% VaR: {historical_var.var_99:.4f}")
    
    # Access volatility results
    historical_vol = risk_metrics.volatility_results['historical']
    print(f"Annualized Volatility: {historical_vol.annualized_volatility:.4f}")
    
    # Access other risk metrics
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {risk_metrics.max_drawdown:.4f}")

# Run the calculation
asyncio.run(calculate_risk())
```

### Portfolio Risk Analysis

```python
from stock_analysis_system.analysis.risk_management_engine import (
    calculate_portfolio_var,
    calculate_component_var
)
import numpy as np

# Individual asset VaRs
individual_vars = [0.025, 0.032, 0.028]

# Correlation matrix
correlations = np.array([
    [1.0, 0.6, 0.4],
    [0.6, 1.0, 0.5],
    [0.4, 0.5, 1.0]
])

# Portfolio weights
weights = np.array([0.5, 0.3, 0.2])

# Calculate portfolio VaR
portfolio_var = calculate_portfolio_var(individual_vars, correlations, weights)
print(f"Portfolio VaR: {portfolio_var:.4f}")

# Calculate component VaRs
component_vars = calculate_component_var(individual_vars, correlations, weights)
print(f"Component VaRs: {component_vars}")
```

### Stress Testing

```python
from stock_analysis_system.analysis.risk_management_engine import stress_test_portfolio

# Define stress scenarios
stress_scenarios = {
    'market_crash': {'AAPL': -0.3, 'GOOGL': -0.25},
    'tech_selloff': {'AAPL': -0.2, 'GOOGL': -0.35}
}

# Portfolio data
portfolio_data = {
    'AAPL': apple_price_data,
    'GOOGL': google_price_data
}

# Run stress test
stress_results = await stress_test_portfolio(
    risk_engine, portfolio_data, stress_scenarios
)
```

## Configuration

### Engine Parameters

- `confidence_levels`: List of confidence levels for VaR (default: [0.95, 0.99])
- `var_window`: Window size for VaR calculations in trading days (default: 252)
- `volatility_window`: Window size for volatility calculations (default: 30)
- `monte_carlo_simulations`: Number of simulations for Monte Carlo VaR (default: 10000)

### Risk-Free Rate

The engine uses a configurable risk-free rate for Sharpe and Sortino ratio calculations:

```python
risk_engine.risk_free_rate = 0.03  # 3% annual risk-free rate
```

## Data Requirements

### Price Data Format

```python
price_data = pd.DataFrame({
    'date': pd.DatetimeIndex,  # Trading dates
    'close': float,            # Closing prices (required)
    'high': float,             # High prices (optional)
    'low': float,              # Low prices (optional)
    'open': float              # Opening prices (optional)
})
```

### Volume Data Format (Optional)

```python
volume_data = pd.DataFrame({
    'date': pd.DatetimeIndex,  # Trading dates
    'volume': float            # Trading volume
})
```

### Benchmark Data Format (Optional)

Same format as price data, used for beta calculation.

## Output Data Structures

### VaRResult

```python
@dataclass
class VaRResult:
    var_95: float                           # 95% VaR
    var_99: float                           # 99% VaR
    cvar_95: float                          # 95% Conditional VaR
    cvar_99: float                          # 99% Conditional VaR
    method: VaRMethod                       # Calculation method
    confidence_interval: Optional[Tuple]    # Confidence interval (Monte Carlo only)
    calculation_date: datetime              # When calculated
```

### VolatilityResult

```python
@dataclass
class VolatilityResult:
    daily_volatility: float        # Daily volatility
    annualized_volatility: float   # Annualized volatility
    method: VolatilityMethod       # Calculation method
    window_size: int               # Data window size used
    realized_volatility: Optional[float]  # Realized volatility
```

### RiskMetrics

```python
@dataclass
class RiskMetrics:
    var_results: Dict[str, VaRResult]           # VaR by method
    volatility_results: Dict[str, VolatilityResult]  # Volatility by method
    max_drawdown: float                         # Maximum drawdown
    sharpe_ratio: float                         # Sharpe ratio
    sortino_ratio: float                        # Sortino ratio
    calmar_ratio: float                         # Calmar ratio
    beta: Optional[float]                       # Beta vs benchmark
    liquidity_risk_score: Optional[float]      # Liquidity risk (0-100)
    seasonal_risk_score: Optional[float]       # Seasonal risk (future)
```

## Error Handling

The engine includes comprehensive error handling:

- **Data Validation**: Checks for required columns, sufficient data, and data quality
- **Calculation Errors**: Graceful fallbacks when specific methods fail
- **Insufficient Data**: Clear error messages for data requirements
- **Numerical Issues**: Handles edge cases like zero volatility

## Performance Considerations

- **Async Operations**: All main calculations are async for better performance
- **Caching**: Consider implementing caching for repeated calculations
- **Memory Usage**: Large datasets are processed efficiently
- **Parallel Processing**: Monte Carlo simulations can be parallelized

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_risk_management_engine.py -v
```

Run the demo script:

```bash
python test_risk_management_demo.py
```

## Integration with Other Components

The Risk Management Engine integrates with:

- **Data Source Manager**: For retrieving price and volume data
- **Spring Festival Engine**: For seasonal risk scoring
- **Portfolio Manager**: For portfolio-level risk analysis
- **Alert System**: For risk-based notifications

## Future Enhancements

- **Advanced GARCH Models**: EGARCH, GJR-GARCH implementations
- **Copula-based VaR**: For better dependency modeling
- **Backtesting Framework**: VaR model validation
- **Real-time Risk Monitoring**: Streaming risk calculations
- **Regulatory VaR**: Basel III compliant calculations

## Requirements Addressed

This implementation addresses the following requirements:

- **4.1**: Historical volatility and Value at Risk (VaR) for individual stocks
- **4.2**: Seasonal risk assessment integration capability
- **4.3**: System-wide alerts for extreme market conditions
- **4.4**: Dynamic stop-loss levels based on volatility
- **4.5**: Confidence intervals and maximum potential daily losses

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0 (for advanced features)