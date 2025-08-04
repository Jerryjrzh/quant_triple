"""
Demo script for Advanced Risk Metrics Calculator

This script demonstrates the enhanced risk metrics including Sharpe/Sortino/Calmar ratios
with confidence intervals, beta calculations, liquidity risk scoring, and seasonal risk
integration implemented in task 5.2.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.advanced_risk_metrics import (
    AdvancedRiskMetricsCalculator,
    calculate_comprehensive_risk_profile,
    SeasonalRiskLevel
)


def generate_realistic_stock_data():
    """Generate realistic stock data with various market conditions."""
    
    np.random.seed(42)
    
    # Generate 3 years of daily data
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    # Create more realistic stock price simulation
    returns = []
    volatility = 0.02  # Base volatility
    
    for i, date in enumerate(dates):
        # Add seasonal volatility patterns
        month = date.month
        if month in [1, 2, 12]:  # Higher volatility around Chinese New Year
            seasonal_multiplier = 1.3
        elif month in [6, 7, 8]:  # Summer volatility
            seasonal_multiplier = 1.1
        else:
            seasonal_multiplier = 1.0
        
        # Add volatility clustering
        if i > 0 and abs(returns[i-1]) > 0.03:
            volatility = min(0.06, volatility * 1.3)
        else:
            volatility = max(0.015, volatility * 0.98)
        
        # Generate return with fat tails and seasonal effects
        current_vol = volatility * seasonal_multiplier
        
        if np.random.random() < 0.03:  # 3% chance of extreme move
            ret = np.random.normal(0, current_vol * 4)
        else:
            ret = np.random.normal(0.0008, current_vol)  # Slight positive drift
        
        returns.append(ret)
    
    # Convert returns to prices
    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data with realistic spreads
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.012))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.012))) for p in prices]
    })
    
    return df


def generate_benchmark_data():
    """Generate benchmark (market index) data."""
    
    np.random.seed(123)
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    # Market index with lower volatility and steady growth
    returns = []
    for i, date in enumerate(dates):
        # Market tends to have lower volatility
        base_vol = 0.015
        
        # Add some market-wide events
        if np.random.random() < 0.01:  # 1% chance of market event
            ret = np.random.normal(0, base_vol * 3)
        else:
            ret = np.random.normal(0.0005, base_vol)  # Steady growth
        
        returns.append(ret)
    
    # Convert to prices
    prices = [3000]  # Start at 3000 (like a major index)
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'date': dates,
        'close': prices
    })


def generate_volume_data_with_patterns():
    """Generate volume data with realistic trading patterns."""
    
    np.random.seed(456)
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    volumes = []
    base_volume = 1000000
    
    for i, date in enumerate(dates):
        # Day of week effects
        weekday = date.weekday()
        if weekday == 0:  # Monday - higher volume
            day_multiplier = 1.2
        elif weekday == 4:  # Friday - higher volume
            day_multiplier = 1.15
        elif weekday in [5, 6]:  # Weekend - no trading
            volumes.append(0)
            continue
        else:
            day_multiplier = 1.0
        
        # Month effects (higher volume at month end)
        if date.day > 25:
            month_multiplier = 1.1
        else:
            month_multiplier = 1.0
        
        # Random volume with log-normal distribution
        volume = base_volume * day_multiplier * month_multiplier * np.random.lognormal(0, 0.6)
        volumes.append(max(0, volume))
    
    return pd.DataFrame({
        'date': dates,
        'volume': volumes
    })


class MockSpringFestivalEngine:
    """Mock Spring Festival engine for demonstration."""
    
    async def analyze_seasonal_patterns(self, price_data):
        """Mock seasonal pattern analysis."""
        return {
            'seasonal_volatility': 0.28,
            'current_risk_level': 'moderate',
            'spring_festival_proximity': 45  # days
        }


async def demonstrate_advanced_risk_metrics():
    """Demonstrate advanced risk metrics calculation."""
    
    print("=== Advanced Risk Metrics Calculator Demo ===\n")
    
    # Initialize calculator
    calculator = AdvancedRiskMetricsCalculator(
        risk_free_rate=0.03,
        confidence_level=0.95,
        bootstrap_iterations=1000,
        seasonal_window_years=3
    )
    
    # Generate sample data
    print("Generating realistic market data...")
    stock_data = generate_realistic_stock_data()
    benchmark_data = generate_benchmark_data()
    volume_data = generate_volume_data_with_patterns()
    
    print(f"Generated {len(stock_data)} days of stock data")
    print(f"Stock price range: ${stock_data['close'].min():.2f} - ${stock_data['close'].max():.2f}")
    print(f"Final stock price: ${stock_data['close'].iloc[-1]:.2f}")
    print(f"Benchmark final value: {benchmark_data['close'].iloc[-1]:.2f}")
    print(f"Average daily volume: {volume_data['volume'].mean():,.0f}\n")
    
    # Calculate advanced risk metrics
    print("Calculating advanced risk metrics...")
    
    try:
        mock_sf_engine = MockSpringFestivalEngine()
        
        advanced_metrics = await calculator.calculate_advanced_metrics(
            price_data=stock_data,
            benchmark_data=benchmark_data,
            volume_data=volume_data,
            spring_festival_engine=mock_sf_engine
        )
        
        print("✓ Advanced risk metrics calculation completed!\n")
        
        # Display enhanced ratio metrics
        print("=== Enhanced Risk-Adjusted Return Metrics ===")
        print(f"Sharpe Ratio: {advanced_metrics.sharpe_ratio:.4f}")
        if advanced_metrics.sharpe_confidence_interval:
            ci_lower, ci_upper = advanced_metrics.sharpe_confidence_interval
            print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        print(f"\nSortino Ratio: {advanced_metrics.sortino_ratio:.4f}")
        if advanced_metrics.sortino_confidence_interval:
            ci_lower, ci_upper = advanced_metrics.sortino_confidence_interval
            print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        print(f"\nCalmar Ratio: {advanced_metrics.calmar_ratio:.4f}")
        if advanced_metrics.calmar_confidence_interval:
            ci_lower, ci_upper = advanced_metrics.calmar_confidence_interval
            print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Display market risk metrics
        print("\n=== Market Risk Metrics ===")
        if advanced_metrics.beta is not None:
            print(f"Beta: {advanced_metrics.beta:.4f}")
            if advanced_metrics.beta_confidence_interval:
                ci_lower, ci_upper = advanced_metrics.beta_confidence_interval
                print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Interpret beta
            if advanced_metrics.beta < 0.8:
                beta_desc = "Defensive (less volatile than market)"
            elif advanced_metrics.beta > 1.2:
                beta_desc = "Aggressive (more volatile than market)"
            else:
                beta_desc = "Market-like volatility"
            print(f"  Interpretation: {beta_desc}")
        
        if advanced_metrics.alpha is not None:
            print(f"\nJensen's Alpha: {advanced_metrics.alpha:.4f} ({advanced_metrics.alpha*100:.2f}% annually)")
            if advanced_metrics.alpha > 0.02:
                alpha_desc = "Positive alpha - outperforming risk-adjusted expectations"
            elif advanced_metrics.alpha < -0.02:
                alpha_desc = "Negative alpha - underperforming risk-adjusted expectations"
            else:
                alpha_desc = "Neutral alpha - performing as expected"
            print(f"  Interpretation: {alpha_desc}")
        
        if advanced_metrics.tracking_error is not None:
            print(f"\nTracking Error: {advanced_metrics.tracking_error:.4f} ({advanced_metrics.tracking_error*100:.2f}% annually)")
        
        if advanced_metrics.information_ratio is not None:
            print(f"Information Ratio: {advanced_metrics.information_ratio:.4f}")
        
        # Display liquidity risk metrics
        print("\n=== Liquidity Risk Assessment ===")
        print(f"Liquidity Risk Score: {advanced_metrics.liquidity_risk_score:.2f}/100")
        print(f"Liquidity Level: {advanced_metrics.liquidity_risk_level}")
        
        if advanced_metrics.bid_ask_spread_proxy is not None:
            print(f"Bid-Ask Spread Proxy: {advanced_metrics.bid_ask_spread_proxy:.4f} ({advanced_metrics.bid_ask_spread_proxy*100:.2f}%)")
        
        if advanced_metrics.market_impact_score is not None:
            print(f"Market Impact Score: {advanced_metrics.market_impact_score:.6f}")
        
        # Display seasonal risk metrics
        print("\n=== Seasonal Risk Analysis ===")
        print(f"Seasonal Risk Score: {advanced_metrics.seasonal_risk_score:.2f}/100")
        print(f"Seasonal Risk Level: {advanced_metrics.seasonal_risk_level.value.replace('_', ' ').title()}")
        print(f"Spring Festival Risk Adjustment: {advanced_metrics.spring_festival_risk_adjustment:.2f}x")
        
        if advanced_metrics.spring_festival_risk_adjustment > 1.1:
            sf_desc = "Currently in elevated risk period due to Spring Festival proximity"
        elif advanced_metrics.spring_festival_risk_adjustment < 0.9:
            sf_desc = "Currently in reduced risk period"
        else:
            sf_desc = "Normal seasonal risk level"
        print(f"  Interpretation: {sf_desc}")
        
        # Display monthly volatility patterns
        if advanced_metrics.historical_seasonal_volatility:
            print("\nHistorical Monthly Volatility Patterns:")
            for month_key, volatility in advanced_metrics.historical_seasonal_volatility.items():
                month_num = int(month_key.split('_')[1])
                month_name = pd.Timestamp(2023, month_num, 1).strftime('%B')
                print(f"  {month_name}: {volatility:.2f}% (annualized)")
        
        # Display additional advanced metrics
        print("\n=== Additional Advanced Metrics ===")
        
        if advanced_metrics.omega_ratio is not None:
            print(f"Omega Ratio: {advanced_metrics.omega_ratio:.4f}")
            if advanced_metrics.omega_ratio > 1.5:
                omega_desc = "Strong probability-weighted performance"
            elif advanced_metrics.omega_ratio > 1.0:
                omega_desc = "Positive probability-weighted performance"
            else:
                omega_desc = "Weak probability-weighted performance"
            print(f"  Interpretation: {omega_desc}")
        
        if advanced_metrics.kappa_3 is not None:
            print(f"\nKappa 3 (Skewness-adjusted): {advanced_metrics.kappa_3:.4f}")
        
        if advanced_metrics.tail_ratio is not None:
            print(f"Tail Ratio (95th/5th percentile): {advanced_metrics.tail_ratio:.4f}")
            if advanced_metrics.tail_ratio > 3.0:
                tail_desc = "High tail risk - large potential gains and losses"
            elif advanced_metrics.tail_ratio > 2.0:
                tail_desc = "Moderate tail risk"
            else:
                tail_desc = "Low tail risk - more symmetric return distribution"
            print(f"  Interpretation: {tail_desc}")
        
    except Exception as e:
        print(f"❌ Error calculating advanced risk metrics: {e}")
        return


async def demonstrate_comprehensive_risk_profile():
    """Demonstrate comprehensive risk profile calculation."""
    
    print("\n\n=== Comprehensive Risk Profile Analysis ===")
    
    # Generate data
    stock_data = generate_realistic_stock_data()
    benchmark_data = generate_benchmark_data()
    volume_data = generate_volume_data_with_patterns()
    
    try:
        # Calculate comprehensive risk profile
        risk_profile = await calculate_comprehensive_risk_profile(
            price_data=stock_data,
            benchmark_data=benchmark_data,
            volume_data=volume_data,
            spring_festival_engine=MockSpringFestivalEngine(),
            risk_engine=None
        )
        
        print("✓ Comprehensive risk profile calculation completed!\n")
        
        # Display summary
        summary = risk_profile['summary']
        
        print("=== Risk Profile Summary ===")
        print(f"Overall Risk Score: {summary['overall_risk_score']:.1f}/100")
        print(f"Risk Level: {summary['risk_level']}")
        
        print(f"\nKey Risk Factors:")
        for factor in summary['key_risk_factors']:
            print(f"  • {factor}")
        
        print(f"\nRisk Management Recommendations:")
        for recommendation in summary['recommendations']:
            print(f"  • {recommendation}")
        
        # Display metric comparison
        print("\n=== Basic vs Advanced Metrics Comparison ===")
        basic = risk_profile['basic_metrics']
        advanced = risk_profile['advanced_metrics']
        
        print("Sharpe Ratio:")
        print(f"  Basic: {basic.sharpe_ratio:.4f}")
        print(f"  Advanced: {advanced.sharpe_ratio:.4f}")
        if advanced.sharpe_confidence_interval:
            ci_lower, ci_upper = advanced.sharpe_confidence_interval
            print(f"  Advanced 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        print(f"\nLiquidity Assessment:")
        if basic.liquidity_risk_score:
            print(f"  Basic Score: {basic.liquidity_risk_score:.1f}/100")
        print(f"  Advanced Score: {advanced.liquidity_risk_score:.1f}/100")
        print(f"  Advanced Level: {advanced.liquidity_risk_level}")
        
        print(f"\nSeasonal Risk Integration:")
        print(f"  Seasonal Risk Score: {advanced.seasonal_risk_score:.1f}/100")
        print(f"  Spring Festival Adjustment: {advanced.spring_festival_risk_adjustment:.2f}x")
        
    except Exception as e:
        print(f"❌ Error calculating comprehensive risk profile: {e}")


async def demonstrate_confidence_intervals():
    """Demonstrate the importance of confidence intervals in risk metrics."""
    
    print("\n\n=== Confidence Intervals Demonstration ===")
    
    print("Generating multiple scenarios to show confidence interval importance...")
    
    # Generate multiple datasets with different characteristics
    scenarios = {
        'Low Volatility': {'vol': 0.15, 'drift': 0.08},
        'High Volatility': {'vol': 0.35, 'drift': 0.12},
        'Negative Skew': {'vol': 0.25, 'drift': 0.10},
        'High Sharpe': {'vol': 0.20, 'drift': 0.15}
    }
    
    calculator = AdvancedRiskMetricsCalculator(bootstrap_iterations=500)
    
    for scenario_name, params in scenarios.items():
        print(f"\n--- {scenario_name} Scenario ---")
        
        # Generate scenario-specific data
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        
        if scenario_name == 'Negative Skew':
            # Create negative skew with occasional large negative returns
            returns = []
            for _ in range(len(dates)):
                if np.random.random() < 0.05:  # 5% chance of large negative return
                    ret = np.random.normal(-0.05, 0.02)
                else:
                    ret = np.random.normal(params['drift']/252, params['vol']/np.sqrt(252))
                returns.append(ret)
        else:
            returns = np.random.normal(params['drift']/252, params['vol']/np.sqrt(252), len(dates))
        
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        scenario_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'open': prices
        })
        
        # Calculate metrics
        try:
            returns_series = calculator._calculate_returns(scenario_data)
            sharpe, sharpe_ci = await calculator._calculate_enhanced_sharpe_ratio(returns_series)
            sortino, sortino_ci = await calculator._calculate_enhanced_sortino_ratio(returns_series)
            
            print(f"Sharpe Ratio: {sharpe:.4f}")
            if sharpe_ci:
                print(f"  95% CI: [{sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f}]")
                ci_width = sharpe_ci[1] - sharpe_ci[0]
                print(f"  CI Width: {ci_width:.4f} (uncertainty measure)")
            
            print(f"Sortino Ratio: {sortino:.4f}")
            if sortino_ci and not np.isinf(sortino):
                print(f"  95% CI: [{sortino_ci[0]:.4f}, {sortino_ci[1]:.4f}]")
            
        except Exception as e:
            print(f"  Error in scenario calculation: {e}")


if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_risk_metrics())
    asyncio.run(demonstrate_comprehensive_risk_profile())
    asyncio.run(demonstrate_confidence_intervals())