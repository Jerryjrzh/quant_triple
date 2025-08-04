"""
Demo script for Dynamic Position Sizing Engine

This script demonstrates Kelly Criterion-based position sizing, risk-adjusted position sizing,
portfolio concentration monitoring, and risk budget management implemented in task 5.3.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_analysis_system.analysis.position_sizing_engine import (
    DynamicPositionSizingEngine,
    PositionSizingMethod,
    RiskBudgetMethod,
    ConcentrationRiskLevel,
    calculate_optimal_rebalancing_frequency,
    calculate_position_size_impact,
    backtest_position_sizing_strategy
)


def generate_realistic_asset_data(symbol: str, start_date: str, periods: int, 
                                expected_return: float, volatility: float) -> pd.DataFrame:
    """Generate realistic asset price data with specified characteristics."""
    
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate returns with some autocorrelation (more realistic)
    returns = []
    prev_return = 0.0
    
    for i in range(periods):
        # Add some momentum/mean reversion
        momentum = prev_return * 0.1  # 10% momentum
        random_shock = np.random.normal(expected_return/252, volatility/np.sqrt(252))
        
        current_return = momentum + random_shock
        returns.append(current_return)
        prev_return = current_return
    
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


def create_sample_portfolio():
    """Create a sample portfolio with different asset characteristics."""
    
    # Define assets with different risk/return profiles
    asset_specs = {
        'GROWTH_STOCK': {'return': 0.15, 'volatility': 0.35, 'sector': 'Technology'},
        'VALUE_STOCK': {'return': 0.10, 'volatility': 0.20, 'sector': 'Financial'},
        'DIVIDEND_STOCK': {'return': 0.08, 'volatility': 0.15, 'sector': 'Utilities'},
        'MOMENTUM_STOCK': {'return': 0.18, 'volatility': 0.40, 'sector': 'Technology'},
        'DEFENSIVE_STOCK': {'return': 0.06, 'volatility': 0.12, 'sector': 'Consumer Staples'},
        'CYCLICAL_STOCK': {'return': 0.12, 'volatility': 0.30, 'sector': 'Industrial'},
        'COMMODITY_STOCK': {'return': 0.09, 'volatility': 0.45, 'sector': 'Energy'}
    }
    
    portfolio_data = {}
    asset_sectors = {}
    
    for symbol, specs in asset_specs.items():
        portfolio_data[symbol] = generate_realistic_asset_data(
            symbol=symbol,
            start_date='2021-01-01',
            periods=1000,  # ~3 years of data
            expected_return=specs['return'],
            volatility=specs['volatility']
        )
        asset_sectors[symbol] = specs['sector']
    
    return portfolio_data, asset_sectors


async def demonstrate_individual_position_sizing():
    """Demonstrate individual position sizing methods."""
    
    print("=== Individual Position Sizing Demo ===\n")
    
    # Initialize position sizing engine
    sizing_engine = DynamicPositionSizingEngine(
        default_risk_budget=0.02,    # 2% portfolio risk budget
        max_position_weight=0.15,    # 15% max position
        min_position_weight=0.02,    # 2% min position
        kelly_multiplier=0.25,       # Conservative Kelly multiplier
        concentration_threshold=0.60  # 60% concentration warning
    )
    
    # Create sample asset data
    print("Generating sample asset data...")
    
    # High-quality growth stock
    growth_stock = generate_realistic_asset_data(
        'GROWTH_STOCK', '2022-01-01', 500, 0.15, 0.25
    )
    
    # Volatile momentum stock
    momentum_stock = generate_realistic_asset_data(
        'MOMENTUM_STOCK', '2022-01-01', 500, 0.20, 0.45
    )
    
    # Defensive dividend stock
    defensive_stock = generate_realistic_asset_data(
        'DEFENSIVE_STOCK', '2022-01-01', 500, 0.08, 0.15
    )
    
    portfolio_value = 1000000  # $1M portfolio
    
    print(f"Portfolio Value: ${portfolio_value:,}")
    print(f"Max Position Weight: {sizing_engine.max_position_weight:.1%}")
    print(f"Min Position Weight: {sizing_engine.min_position_weight:.1%}\n")
    
    # Test different sizing methods on different assets
    assets_to_test = [
        ('GROWTH_STOCK', growth_stock, "High-quality growth stock"),
        ('MOMENTUM_STOCK', momentum_stock, "Volatile momentum stock"),
        ('DEFENSIVE_STOCK', defensive_stock, "Defensive dividend stock")
    ]
    
    methods_to_test = [
        (PositionSizingMethod.KELLY_CRITERION, "Kelly Criterion"),
        (PositionSizingMethod.VOLATILITY_ADJUSTED, "Volatility Adjusted"),
        (PositionSizingMethod.VAR_BASED, "VaR Based"),
        (PositionSizingMethod.FIXED_FRACTIONAL, "Fixed Fractional")
    ]
    
    for symbol, price_data, description in assets_to_test:
        print(f"--- {symbol} ({description}) ---")
        
        # Show asset characteristics
        returns = price_data['close'].pct_change().dropna()
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        current_price = price_data['close'].iloc[-1]
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Expected Annual Return: {annual_return:.1%}")
        print(f"Annual Volatility: {annual_vol:.1%}")
        
        print("\nPosition Sizing Recommendations:")
        
        for method, method_name in methods_to_test:
            try:
                recommendation = await sizing_engine.calculate_position_size(
                    symbol=symbol,
                    price_data=price_data,
                    portfolio_value=portfolio_value,
                    method=method
                )
                
                print(f"\n  {method_name}:")
                print(f"    Recommended Weight: {recommendation.recommended_weight:.2%}")
                print(f"    Dollar Amount: ${recommendation.recommended_dollar_amount:,.0f}")
                print(f"    Shares: {recommendation.recommended_shares:,}")
                print(f"    Max Loss Estimate: ${recommendation.max_loss_estimate:,.0f}")
                print(f"    Confidence Level: {recommendation.confidence_level:.1%}")
                
                if recommendation.kelly_fraction is not None:
                    print(f"    Kelly Fraction: {recommendation.kelly_fraction:.3f}")
                
                if recommendation.warnings:
                    print(f"    Warnings: {', '.join(recommendation.warnings)}")
                
            except Exception as e:
                print(f"  {method_name}: Error - {e}")
        
        print("\n" + "="*60 + "\n")


async def demonstrate_portfolio_risk_budgeting():
    """Demonstrate portfolio risk budget optimization."""
    
    print("=== Portfolio Risk Budget Optimization Demo ===\n")
    
    # Create portfolio data
    portfolio_data, asset_sectors = create_sample_portfolio()
    
    print("Portfolio Assets:")
    for symbol, sector in asset_sectors.items():
        price_data = portfolio_data[symbol]
        returns = price_data['close'].pct_change().dropna()
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        print(f"  {symbol}: {annual_return:.1%} return, {annual_vol:.1%} volatility ({sector})")
    
    print(f"\nTotal Assets: {len(portfolio_data)}")
    
    # Initialize sizing engine
    sizing_engine = DynamicPositionSizingEngine()
    portfolio_value = 2000000  # $2M portfolio
    
    print(f"Portfolio Value: ${portfolio_value:,}")
    
    # Test different risk budgeting methods
    methods_to_test = [
        (RiskBudgetMethod.RISK_PARITY, "Risk Parity"),
        (RiskBudgetMethod.EQUAL_RISK, "Equal Risk Contribution"),
        (RiskBudgetMethod.INVERSE_VOLATILITY, "Inverse Volatility")
    ]
    
    for method, method_name in methods_to_test:
        print(f"\n--- {method_name} Optimization ---")
        
        try:
            risk_budget = await sizing_engine.optimize_portfolio_risk_budget(
                assets=portfolio_data,
                portfolio_value=portfolio_value,
                risk_budget=0.025,  # 2.5% portfolio risk budget
                method=method
            )
            
            print(f"Total Risk Budget: {risk_budget.total_risk_budget:.1%}")
            print(f"Budget Utilization: {risk_budget.budget_utilization:.1%}")
            print(f"Diversification Ratio: {risk_budget.diversification_ratio:.2f}")
            
            print("\nAsset Allocation:")
            sorted_weights = sorted(risk_budget.asset_weights.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for asset, weight in sorted_weights:
                risk_contrib = risk_budget.risk_contributions[asset]
                risk_budget_allocated = risk_budget.asset_risk_budgets[asset]
                dollar_amount = weight * portfolio_value
                
                print(f"  {asset:15s}: {weight:6.1%} weight, "
                      f"{risk_contrib:6.1%} risk contrib, "
                      f"${dollar_amount:8,.0f}")
            
            # Show concentration metrics
            conc_metrics = risk_budget.concentration_metrics
            print(f"\nConcentration Metrics:")
            print(f"  Herfindahl Index: {conc_metrics['herfindahl_index']:.3f}")
            print(f"  Effective # Assets: {conc_metrics['effective_number_assets']:.1f}")
            print(f"  Max Weight: {conc_metrics['max_weight']:.1%}")
            print(f"  Top 3 Concentration: {conc_metrics['top_3_concentration']:.1%}")
            
        except Exception as e:
            print(f"Error in {method_name}: {e}")
        
        print("\n" + "="*60)


async def demonstrate_concentration_risk_analysis():
    """Demonstrate portfolio concentration risk analysis."""
    
    print("\n=== Portfolio Concentration Risk Analysis Demo ===\n")
    
    sizing_engine = DynamicPositionSizingEngine()
    
    # Test different concentration scenarios
    scenarios = {
        "Well Diversified": {
            'STOCK_A': 0.12, 'STOCK_B': 0.12, 'STOCK_C': 0.11, 'STOCK_D': 0.11,
            'STOCK_E': 0.11, 'STOCK_F': 0.10, 'STOCK_G': 0.10, 'STOCK_H': 0.10,
            'STOCK_I': 0.08, 'STOCK_J': 0.05
        },
        "Moderate Concentration": {
            'LARGE_POSITION': 0.25, 'MEDIUM_1': 0.20, 'MEDIUM_2': 0.15,
            'SMALL_1': 0.10, 'SMALL_2': 0.10, 'SMALL_3': 0.08,
            'SMALL_4': 0.07, 'SMALL_5': 0.05
        },
        "High Concentration": {
            'DOMINANT_STOCK': 0.45, 'SECOND_LARGEST': 0.20, 'THIRD': 0.15,
            'FOURTH': 0.10, 'FIFTH': 0.05, 'SIXTH': 0.03, 'SEVENTH': 0.02
        },
        "Extreme Concentration": {
            'MEGA_POSITION': 0.70, 'SECOND': 0.15, 'THIRD': 0.08,
            'FOURTH': 0.04, 'FIFTH': 0.03
        }
    }
    
    # Sector mapping for some scenarios
    sector_mappings = {
        "High Concentration": {
            'DOMINANT_STOCK': 'Technology', 'SECOND_LARGEST': 'Technology',
            'THIRD': 'Technology', 'FOURTH': 'Financial', 'FIFTH': 'Healthcare',
            'SIXTH': 'Energy', 'SEVENTH': 'Utilities'
        }
    }
    
    for scenario_name, weights in scenarios.items():
        print(f"--- {scenario_name} Portfolio ---")
        
        # Get sector mapping if available
        sectors = sector_mappings.get(scenario_name)
        
        analysis = await sizing_engine.analyze_concentration_risk(
            portfolio_weights=weights,
            asset_sectors=sectors
        )
        
        print(f"Concentration Level: {analysis.concentration_level.value.replace('_', ' ').title()}")
        print(f"Concentration Score: {analysis.concentration_score:.1f}/100")
        print(f"Herfindahl Index: {analysis.herfindahl_index:.3f}")
        print(f"Effective # of Assets: {analysis.effective_number_of_assets:.1f}")
        print(f"Maximum Weight: {analysis.max_weight:.1%}")
        print(f"Top 5 Concentration: {analysis.top_5_concentration:.1%}")
        
        if analysis.sector_concentration:
            print("\nSector Concentration:")
            for sector, concentration in analysis.sector_concentration.items():
                print(f"  {sector}: {concentration:.1%}")
        
        if analysis.concentration_warnings:
            print(f"\nWarnings:")
            for warning in analysis.concentration_warnings:
                print(f"  • {warning}")
        
        if analysis.diversification_recommendations:
            print(f"\nRecommendations:")
            for rec in analysis.diversification_recommendations:
                print(f"  • {rec}")
        
        print("\n" + "="*50 + "\n")


async def demonstrate_rebalancing_recommendations():
    """Demonstrate portfolio rebalancing recommendations."""
    
    print("=== Portfolio Rebalancing Recommendations Demo ===\n")
    
    sizing_engine = DynamicPositionSizingEngine()
    
    # Current portfolio (drifted from targets)
    current_weights = {
        'AAPL': 0.28,   # Grown from target
        'GOOGL': 0.18,  # Slightly below target
        'MSFT': 0.22,   # Above target
        'TSLA': 0.12,   # Below target
        'NVDA': 0.08,   # Below target
        'META': 0.06,   # Below target
        'AMZN': 0.06    # New position needed
    }
    
    # Target weights (from optimization)
    target_weights = {
        'AAPL': 0.20,   # Reduce
        'GOOGL': 0.20,  # Increase
        'MSFT': 0.18,   # Reduce
        'TSLA': 0.15,   # Increase
        'NVDA': 0.12,   # Increase
        'META': 0.10,   # Increase
        'AMZN': 0.05    # Reduce
    }
    
    portfolio_value = 1500000  # $1.5M portfolio
    
    # Transaction costs (different for each asset)
    transaction_costs = {
        'AAPL': 0.0005,   # 0.05% - liquid large cap
        'GOOGL': 0.0008,  # 0.08% - slightly higher
        'MSFT': 0.0005,   # 0.05% - liquid large cap
        'TSLA': 0.0015,   # 0.15% - more volatile
        'NVDA': 0.0012,   # 0.12% - tech stock
        'META': 0.0010,   # 0.10% - large cap
        'AMZN': 0.0008    # 0.08% - large cap
    }
    
    print(f"Portfolio Value: ${portfolio_value:,}")
    print(f"Number of Assets: {len(current_weights)}")
    
    print("\nCurrent vs Target Allocation:")
    print(f"{'Asset':<8} {'Current':<8} {'Target':<8} {'Difference':<10}")
    print("-" * 40)
    
    for asset in sorted(current_weights.keys()):
        current = current_weights[asset]
        target = target_weights[asset]
        diff = target - current
        
        print(f"{asset:<8} {current:>7.1%} {target:>7.1%} {diff:>+9.1%}")
    
    # Generate recommendations
    recommendations = await sizing_engine.generate_portfolio_recommendations(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=portfolio_value,
        transaction_costs=transaction_costs
    )
    
    print(f"\n=== Trading Recommendations ===")
    print(f"{'Asset':<8} {'Action':<6} {'Amount':<12} {'Cost':<10} {'Priority':<8}")
    print("-" * 55)
    
    # Sort by priority (highest first)
    sorted_recs = sorted(recommendations.items(), 
                        key=lambda x: x[1]['priority'], reverse=True)
    
    total_transaction_costs = 0
    
    for asset, rec in sorted_recs:
        action = rec['action']
        amount = rec['dollar_amount']
        cost = rec['transaction_cost']
        priority = rec['priority']
        
        total_transaction_costs += cost
        
        print(f"{asset:<8} {action:<6} ${amount:>10,.0f} ${cost:>8,.0f} {priority:>7}/10")
    
    print(f"\nTotal Transaction Costs: ${total_transaction_costs:,.0f}")
    print(f"Transaction Cost as % of Portfolio: {total_transaction_costs/portfolio_value:.3%}")
    
    # Calculate net benefit of rebalancing
    print(f"\nRebalancing Analysis:")
    print(f"• Total trades required: {len(recommendations)}")
    print(f"• Largest position change: {max(abs(rec['weight_change']) for rec in recommendations.values()):.1%}")
    print(f"• Average transaction cost: {np.mean([rec['transaction_cost'] for rec in recommendations.values()]):,.0f}")


async def demonstrate_utility_functions():
    """Demonstrate utility functions."""
    
    print("\n=== Utility Functions Demo ===\n")
    
    # Optimal rebalancing frequency
    print("--- Optimal Rebalancing Frequency ---")
    
    portfolio_weights = {'STOCK_A': 0.3, 'STOCK_B': 0.4, 'STOCK_C': 0.3}
    transaction_costs = {'STOCK_A': 0.001, 'STOCK_B': 0.002, 'STOCK_C': 0.0015}
    volatilities = {'STOCK_A': 0.20, 'STOCK_B': 0.35, 'STOCK_C': 0.25}
    
    optimal_frequency = calculate_optimal_rebalancing_frequency(
        portfolio_weights, transaction_costs, volatilities
    )
    
    print(f"Portfolio Characteristics:")
    for asset in portfolio_weights:
        print(f"  {asset}: {portfolio_weights[asset]:.1%} weight, "
              f"{volatilities[asset]:.1%} volatility, "
              f"{transaction_costs[asset]:.2%} transaction cost")
    
    print(f"\nOptimal Rebalancing Frequency: {optimal_frequency} days")
    
    if optimal_frequency <= 14:
        frequency_desc = "Weekly or bi-weekly"
    elif optimal_frequency <= 45:
        frequency_desc = "Monthly"
    elif optimal_frequency <= 120:
        frequency_desc = "Quarterly"
    else:
        frequency_desc = "Semi-annually or annually"
    
    print(f"Recommendation: {frequency_desc} rebalancing")
    
    # Position size market impact
    print(f"\n--- Position Size Market Impact Analysis ---")
    
    scenarios = [
        {"name": "Small Position", "shares": 5000, "volume": 100000, "price": 50},
        {"name": "Medium Position", "shares": 15000, "volume": 100000, "price": 50},
        {"name": "Large Position", "shares": 30000, "volume": 100000, "price": 50},
        {"name": "Very Large Position", "shares": 50000, "volume": 100000, "price": 50}
    ]
    
    print(f"{'Scenario':<18} {'Participation':<12} {'Impact':<8} {'Cost':<10} {'Days':<6}")
    print("-" * 60)
    
    for scenario in scenarios:
        impact = calculate_position_size_impact(
            position_size=scenario["shares"],
            daily_volume=scenario["volume"],
            price=scenario["price"],
            participation_rate=0.1  # 10% max participation
        )
        
        print(f"{scenario['name']:<18} "
              f"{impact['volume_participation']:>11.1%} "
              f"{impact['market_impact_pct']:>7.2%} "
              f"${impact['impact_cost_dollars']:>8,.0f} "
              f"{impact['days_to_trade']:>5}")
    
    print(f"\nKey Insights:")
    print(f"• Keep position sizes under 10% of daily volume for minimal impact")
    print(f"• Large positions require multiple days to execute efficiently")
    print(f"• Market impact increases non-linearly with position size")


async def demonstrate_backtesting():
    """Demonstrate position sizing strategy backtesting."""
    
    print(f"\n=== Position Sizing Strategy Backtesting Demo ===\n")
    
    print("Generating historical data for backtesting...")
    
    # Create longer historical data for backtesting
    historical_data = {}
    asset_specs = [
        ('TECH_STOCK', 0.12, 0.28),
        ('VALUE_STOCK', 0.09, 0.18),
        ('GROWTH_STOCK', 0.14, 0.32),
        ('DIVIDEND_STOCK', 0.07, 0.15)
    ]
    
    for symbol, expected_return, volatility in asset_specs:
        historical_data[symbol] = generate_realistic_asset_data(
            symbol=symbol,
            start_date='2020-01-01',
            periods=1000,  # ~3 years
            expected_return=expected_return,
            volatility=volatility
        )
    
    print(f"Assets in backtest: {list(historical_data.keys())}")
    print(f"Backtest period: {historical_data['TECH_STOCK']['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{historical_data['TECH_STOCK']['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Create sizing engine
    sizing_engine = DynamicPositionSizingEngine(
        default_risk_budget=0.015,  # 1.5% risk budget
        max_position_weight=0.30,   # 30% max position
        min_position_weight=0.05    # 5% min position
    )
    
    # Run backtest
    print(f"\nRunning backtest...")
    
    try:
        results = await backtest_position_sizing_strategy(
            sizing_engine=sizing_engine,
            historical_data=historical_data,
            initial_capital=1000000,
            rebalance_frequency=30  # Monthly rebalancing
        )
        
        print(f"✓ Backtest completed successfully!")
        
        # Display results
        print(f"\n=== Backtest Results ===")
        print(f"Initial Capital: ${1000000:,}")
        print(f"Final Capital: ${results['final_capital']:,.0f}")
        print(f"Total Return: {results['total_return']:+.1%}")
        print(f"Annualized Return: {results['annual_return']:+.1%}")
        print(f"Volatility: {results['volatility']:.1%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.1%}")
        
        # Performance assessment
        print(f"\n=== Performance Assessment ===")
        
        if results['sharpe_ratio'] > 1.0:
            sharpe_assessment = "Excellent risk-adjusted returns"
        elif results['sharpe_ratio'] > 0.5:
            sharpe_assessment = "Good risk-adjusted returns"
        elif results['sharpe_ratio'] > 0.0:
            sharpe_assessment = "Positive but modest risk-adjusted returns"
        else:
            sharpe_assessment = "Poor risk-adjusted returns"
        
        print(f"• Sharpe Ratio: {sharpe_assessment}")
        
        if results['max_drawdown'] < 0.10:
            drawdown_assessment = "Low drawdown - good risk control"
        elif results['max_drawdown'] < 0.20:
            drawdown_assessment = "Moderate drawdown - acceptable risk"
        else:
            drawdown_assessment = "High drawdown - consider risk reduction"
        
        print(f"• Drawdown: {drawdown_assessment}")
        
        print(f"• Rebalancing Events: {len(results['rebalance_dates'])}")
        print(f"• Average Time Between Rebalancing: {1000 // len(results['rebalance_dates'])} days")
        
        # Show final allocation
        final_weights = results['portfolio_weights'][-1]
        print(f"\nFinal Portfolio Allocation:")
        for asset, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {asset}: {weight:.1%}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")


if __name__ == "__main__":
    print("Dynamic Position Sizing Engine - Comprehensive Demo")
    print("=" * 60)
    
    asyncio.run(demonstrate_individual_position_sizing())
    asyncio.run(demonstrate_portfolio_risk_budgeting())
    asyncio.run(demonstrate_concentration_risk_analysis())
    asyncio.run(demonstrate_rebalancing_recommendations())
    asyncio.run(demonstrate_utility_functions())
    asyncio.run(demonstrate_backtesting())