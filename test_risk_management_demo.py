"""
Demo script for Enhanced Risk Management Engine

This script demonstrates the comprehensive VaR calculations and risk metrics
implemented in task 5.1.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.risk_management_engine import (
    EnhancedRiskManagementEngine,
    VaRMethod,
    VolatilityMethod,
    calculate_portfolio_var,
    calculate_component_var,
    stress_test_portfolio
)


def generate_sample_data():
    """Generate sample stock price data for demonstration."""
    
    np.random.seed(42)
    
    # Generate 2 years of daily data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    
    # Simulate stock price with realistic characteristics
    # - Daily returns with some volatility clustering
    # - Occasional large moves (fat tails)
    
    returns = []
    volatility = 0.02  # Base volatility
    
    for i in range(len(dates)):
        # Add volatility clustering
        if i > 0 and abs(returns[i-1]) > 0.03:
            volatility = min(0.05, volatility * 1.2)  # Increase volatility after large moves
        else:
            volatility = max(0.015, volatility * 0.99)  # Decay volatility
        
        # Generate return with fat tails (mix of normal and extreme moves)
        if np.random.random() < 0.05:  # 5% chance of extreme move
            ret = np.random.normal(0, volatility * 3)
        else:
            ret = np.random.normal(0.0005, volatility)  # Slight positive drift
        
        returns.append(ret)
    
    # Convert returns to prices
    prices = [100]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices]
    })
    
    return df


def generate_volume_data(price_data):
    """Generate volume data correlated with price movements."""
    
    np.random.seed(123)
    
    # Calculate price changes
    price_changes = price_data['close'].pct_change().fillna(0)
    
    # Generate volume with inverse correlation to price (high volume on down days)
    base_volume = 1000000
    volumes = []
    
    for change in price_changes:
        # Higher volume on larger price moves
        volume_multiplier = 1 + abs(change) * 5
        
        # Add some randomness
        volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.5)
        volumes.append(max(0, volume))
    
    return pd.DataFrame({
        'date': price_data['date'],
        'volume': volumes
    })


async def demonstrate_var_calculations():
    """Demonstrate VaR calculations using different methods."""
    
    print("=== Enhanced Risk Management Engine Demo ===\n")
    
    # Initialize the risk engine
    risk_engine = EnhancedRiskManagementEngine(
        confidence_levels=[0.95, 0.99],
        var_window=252,
        volatility_window=30,
        monte_carlo_simulations=5000
    )
    
    # Generate sample data
    print("Generating sample stock price data...")
    price_data = generate_sample_data()
    volume_data = generate_volume_data(price_data)
    
    # Create benchmark data (market index)
    benchmark_data = price_data.copy()
    benchmark_data['close'] = benchmark_data['close'] * 0.8 + np.random.normal(0, 5, len(benchmark_data))
    
    print(f"Generated {len(price_data)} days of price data")
    print(f"Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
    print(f"Final price: ${price_data['close'].iloc[-1]:.2f}\n")
    
    # Calculate comprehensive risk metrics
    print("Calculating comprehensive risk metrics...")
    
    try:
        risk_metrics = await risk_engine.calculate_comprehensive_risk_metrics(
            price_data=price_data,
            benchmark_data=benchmark_data,
            volume_data=volume_data
        )
        
        print("✓ Risk metrics calculation completed successfully!\n")
        
        # Display VaR results
        print("=== Value at Risk (VaR) Results ===")
        for method_name, var_result in risk_metrics.var_results.items():
            print(f"\n{method_name.upper()} VaR:")
            print(f"  95% VaR: {var_result.var_95:.4f} ({var_result.var_95*100:.2f}%)")
            print(f"  99% VaR: {var_result.var_99:.4f} ({var_result.var_99*100:.2f}%)")
            print(f"  95% CVaR: {var_result.cvar_95:.4f} ({var_result.cvar_95*100:.2f}%)")
            print(f"  99% CVaR: {var_result.cvar_99:.4f} ({var_result.cvar_99*100:.2f}%)")
            
            if var_result.confidence_interval:
                ci_lower, ci_upper = var_result.confidence_interval
                print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Display volatility results
        print("\n=== Volatility Measures ===")
        for method_name, vol_result in risk_metrics.volatility_results.items():
            print(f"\n{method_name.upper()} Volatility:")
            print(f"  Daily: {vol_result.daily_volatility:.4f} ({vol_result.daily_volatility*100:.2f}%)")
            print(f"  Annualized: {vol_result.annualized_volatility:.4f} ({vol_result.annualized_volatility*100:.2f}%)")
            print(f"  Window size: {vol_result.window_size} days")
        
        # Display additional risk metrics
        print("\n=== Additional Risk Metrics ===")
        print(f"Maximum Drawdown: {risk_metrics.max_drawdown:.4f} ({risk_metrics.max_drawdown*100:.2f}%)")
        print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.4f}")
        print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.4f}")
        print(f"Calmar Ratio: {risk_metrics.calmar_ratio:.4f}")
        
        if risk_metrics.beta is not None:
            print(f"Beta (vs benchmark): {risk_metrics.beta:.4f}")
        
        if risk_metrics.liquidity_risk_score is not None:
            print(f"Liquidity Risk Score: {risk_metrics.liquidity_risk_score:.2f}/100")
            
            # Interpret liquidity risk score
            if risk_metrics.liquidity_risk_score < 30:
                liquidity_desc = "High liquidity (low risk)"
            elif risk_metrics.liquidity_risk_score < 60:
                liquidity_desc = "Moderate liquidity"
            else:
                liquidity_desc = "Low liquidity (high risk)"
            print(f"Liquidity Assessment: {liquidity_desc}")
        
    except Exception as e:
        print(f"❌ Error calculating risk metrics: {e}")
        return


async def demonstrate_portfolio_risk():
    """Demonstrate portfolio-level risk calculations."""
    
    print("\n\n=== Portfolio Risk Analysis ===")
    
    # Simulate a 3-asset portfolio
    individual_vars = [0.025, 0.032, 0.028]  # Individual 95% VaRs
    asset_names = ['Stock A', 'Stock B', 'Stock C']
    
    # Correlation matrix (realistic correlations)
    correlations = np.array([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.5],
        [0.4, 0.5, 1.0]
    ])
    
    # Portfolio weights
    weights = np.array([0.5, 0.3, 0.2])
    
    print("Portfolio composition:")
    for i, (name, weight, var) in enumerate(zip(asset_names, weights, individual_vars)):
        print(f"  {name}: {weight*100:.1f}% weight, {var*100:.2f}% individual VaR")
    
    # Calculate portfolio VaR
    portfolio_var = calculate_portfolio_var(individual_vars, correlations, weights)
    print(f"\nPortfolio VaR (95%): {portfolio_var:.4f} ({portfolio_var*100:.2f}%)")
    
    # Calculate diversification benefit
    weighted_avg_var = np.dot(weights, individual_vars)
    diversification_benefit = weighted_avg_var - portfolio_var
    print(f"Weighted average VaR: {weighted_avg_var:.4f} ({weighted_avg_var*100:.2f}%)")
    print(f"Diversification benefit: {diversification_benefit:.4f} ({diversification_benefit*100:.2f}%)")
    print(f"Risk reduction: {(diversification_benefit/weighted_avg_var)*100:.1f}%")
    
    # Calculate component VaRs
    component_vars = calculate_component_var(individual_vars, correlations, weights)
    
    print("\nComponent VaR analysis:")
    for i, (name, comp_var, weight) in enumerate(zip(asset_names, component_vars, weights)):
        contribution_pct = (comp_var / portfolio_var) * 100
        print(f"  {name}: {comp_var:.4f} ({contribution_pct:.1f}% of portfolio risk)")


async def demonstrate_stress_testing():
    """Demonstrate stress testing functionality."""
    
    print("\n\n=== Stress Testing Demo ===")
    
    # This is a simplified demo - in practice, you'd use real portfolio data
    print("Note: This is a simplified demonstration of stress testing framework.")
    print("In production, this would use actual portfolio holdings and market data.")
    
    # Define stress scenarios
    stress_scenarios = {
        'Market Crash (-30%)': {'market_factor': -0.30},
        'Tech Selloff (-40%)': {'tech_factor': -0.40},
        'Interest Rate Shock': {'rate_sensitive': -0.25},
        'Liquidity Crisis': {'small_cap': -0.50}
    }
    
    print("\nDefined stress scenarios:")
    for scenario_name, shocks in stress_scenarios.items():
        print(f"  {scenario_name}: {shocks}")
    
    print("\nStress testing would analyze:")
    print("  • Portfolio VaR under each scenario")
    print("  • Maximum potential losses")
    print("  • Asset-level impacts")
    print("  • Liquidity requirements")
    print("  • Recovery time estimates")


if __name__ == "__main__":
    asyncio.run(demonstrate_var_calculations())
    asyncio.run(demonstrate_portfolio_risk())
    asyncio.run(demonstrate_stress_testing())