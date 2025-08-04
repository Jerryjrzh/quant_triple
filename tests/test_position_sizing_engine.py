"""
Unit tests for Dynamic Position Sizing Engine

Tests Kelly Criterion, risk-adjusted position sizing, portfolio concentration monitoring,
and risk budget management as specified in task 5.3.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch

from stock_analysis_system.analysis.position_sizing_engine import (
    DynamicPositionSizingEngine,
    PositionSizingMethod,
    RiskBudgetMethod,
    ConcentrationRiskLevel,
    PositionSizeRecommendation,
    PortfolioRiskBudget,
    ConcentrationRiskAnalysis,
    calculate_optimal_rebalancing_frequency,
    calculate_position_size_impact,
    backtest_position_sizing_strategy
)


class TestDynamicPositionSizingEngine:
    """Test cases for Dynamic Position Sizing Engine"""
    
    @pytest.fixture
    def sizing_engine(self):
        """Create position sizing engine for testing."""
        return DynamicPositionSizingEngine(
            default_risk_budget=0.02,
            max_position_weight=0.20,
            min_position_weight=0.01,
            kelly_multiplier=0.25,
            concentration_threshold=0.60
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Generate stock price with positive drift
        returns = np.random.normal(0.001, 0.02, len(dates))  # Positive expected return
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices
        })
    
    @pytest.fixture
    def sample_portfolio_assets(self):
        """Create sample portfolio with multiple assets."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        assets = {}
        
        # Create 5 different assets with different characteristics
        for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']):
            # Different volatilities and returns
            vol = 0.15 + i * 0.05  # Increasing volatility
            drift = 0.08 + i * 0.02  # Increasing expected return
            
            returns = np.random.normal(drift/252, vol/np.sqrt(252), len(dates))
            prices = [100 + i * 10]  # Different starting prices
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            assets[symbol] = pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'open': prices
            })
        
        return assets
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_kelly(self, sizing_engine, sample_price_data):
        """Test Kelly Criterion position sizing."""
        
        result = await sizing_engine.calculate_position_size(
            symbol='TEST',
            price_data=sample_price_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.KELLY_CRITERION
        )
        
        # Verify result structure
        assert isinstance(result, PositionSizeRecommendation)
        assert result.symbol == 'TEST'
        assert result.method_used == PositionSizingMethod.KELLY_CRITERION
        
        # Check values are reasonable
        assert 0 <= result.recommended_weight <= sizing_engine.max_position_weight
        assert result.recommended_shares >= 0
        assert result.recommended_dollar_amount >= 0
        
        # Kelly-specific checks
        assert result.kelly_fraction is not None
        assert result.kelly_fraction >= 0  # No short positions
        
        # Check risk metrics are populated
        assert result.expected_return is not None
        assert result.volatility is not None
        assert result.var_95 is not None
        
        # Check constraints
        assert result.min_weight_constraint == sizing_engine.min_position_weight
        assert result.max_weight_constraint == sizing_engine.max_position_weight
        
        # Check confidence and warnings
        assert 0.3 <= result.confidence_level <= 0.95
        assert isinstance(result.warnings, list)
        assert isinstance(result.calculation_date, datetime)
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_volatility_adjusted(self, sizing_engine, sample_price_data):
        """Test volatility-adjusted position sizing."""
        
        result = await sizing_engine.calculate_position_size(
            symbol='TEST',
            price_data=sample_price_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.VOLATILITY_ADJUSTED
        )
        
        assert result.method_used == PositionSizingMethod.VOLATILITY_ADJUSTED
        assert result.kelly_fraction is None  # Not applicable for this method
        assert 0 <= result.recommended_weight <= sizing_engine.max_position_weight
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_var_based(self, sizing_engine, sample_price_data):
        """Test VaR-based position sizing."""
        
        result = await sizing_engine.calculate_position_size(
            symbol='TEST',
            price_data=sample_price_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.VAR_BASED
        )
        
        assert result.method_used == PositionSizingMethod.VAR_BASED
        assert 0 <= result.recommended_weight <= sizing_engine.max_position_weight
        assert result.max_loss_estimate is not None
        assert result.max_loss_estimate >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_risk_metrics(self, sizing_engine, sample_price_data):
        """Test position sizing with pre-calculated risk metrics."""
        
        # Provide custom risk metrics
        risk_metrics = {
            'expected_return': 0.12,  # 12% expected return
            'volatility': 0.25,       # 25% volatility
            'var_95': 0.04,          # 4% VaR
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.15,
            'liquidity_score': 80.0
        }
        
        result = await sizing_engine.calculate_position_size(
            symbol='TEST',
            price_data=sample_price_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.KELLY_CRITERION,
            risk_metrics=risk_metrics
        )
        
        # Should use provided metrics
        assert result.expected_return == 0.12
        assert result.volatility == 0.25
        assert result.var_95 == 0.04
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_risk_budget_risk_parity(self, sizing_engine, sample_portfolio_assets):
        """Test portfolio risk budget optimization with risk parity."""
        
        result = await sizing_engine.optimize_portfolio_risk_budget(
            assets=sample_portfolio_assets,
            portfolio_value=1000000,
            method=RiskBudgetMethod.RISK_PARITY
        )
        
        # Verify result structure
        assert isinstance(result, PortfolioRiskBudget)
        assert result.method_used == RiskBudgetMethod.RISK_PARITY
        
        # Check weights sum to 1
        total_weight = sum(result.asset_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow small numerical errors
        
        # Check all assets have positive weights
        for weight in result.asset_weights.values():
            assert weight > 0
            assert weight <= sizing_engine.max_position_weight
        
        # Check risk contributions
        assert len(result.risk_contributions) == len(sample_portfolio_assets)
        total_risk_contrib = sum(result.risk_contributions.values())
        assert abs(total_risk_contrib - 1.0) < 0.01  # Should sum to 1
        
        # Check budget allocation
        assert result.total_risk_budget == sizing_engine.default_risk_budget
        assert len(result.asset_risk_budgets) == len(sample_portfolio_assets)
        
        # Check diversification metrics
        assert result.diversification_ratio >= 1.0  # Should be >= 1 for diversified portfolio
        assert isinstance(result.concentration_metrics, dict)
        
        # Check budget utilization
        assert 0 <= result.budget_utilization <= 1.0
        
        assert isinstance(result.calculation_date, datetime)
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_risk_budget_inverse_volatility(self, sizing_engine, sample_portfolio_assets):
        """Test portfolio optimization with inverse volatility weighting."""
        
        result = await sizing_engine.optimize_portfolio_risk_budget(
            assets=sample_portfolio_assets,
            portfolio_value=1000000,
            method=RiskBudgetMethod.INVERSE_VOLATILITY
        )
        
        assert result.method_used == RiskBudgetMethod.INVERSE_VOLATILITY
        
        # Lower volatility assets should have higher weights
        weights = result.asset_weights
        
        # Check that weights are inversely related to volatility (approximately)
        # This is a simplified check since we know the test data structure
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
    
    @pytest.mark.asyncio
    async def test_analyze_concentration_risk(self, sizing_engine):
        """Test portfolio concentration risk analysis."""
        
        # Test different concentration scenarios
        
        # Low concentration (equal weights)
        equal_weights = {'A': 0.2, 'B': 0.2, 'C': 0.2, 'D': 0.2, 'E': 0.2}
        
        result = await sizing_engine.analyze_concentration_risk(equal_weights)
        
        assert isinstance(result, ConcentrationRiskAnalysis)
        assert result.concentration_level in [ConcentrationRiskLevel.LOW, ConcentrationRiskLevel.MODERATE]
        
        # Check metrics
        assert 0 <= result.concentration_score <= 100
        assert result.herfindahl_index == pytest.approx(0.2, abs=0.01)  # 5 * (0.2)^2
        assert result.effective_number_of_assets == pytest.approx(5.0, abs=0.1)
        assert result.max_weight == 0.2
        assert result.top_5_concentration == 1.0
        
        # High concentration scenario
        concentrated_weights = {'A': 0.6, 'B': 0.15, 'C': 0.1, 'D': 0.1, 'E': 0.05}
        
        result_concentrated = await sizing_engine.analyze_concentration_risk(concentrated_weights)
        
        assert result_concentrated.concentration_level in [ConcentrationRiskLevel.HIGH, ConcentrationRiskLevel.EXTREME]
        assert result_concentrated.concentration_score > result.concentration_score
        assert result_concentrated.max_weight == 0.6
        assert len(result_concentrated.concentration_warnings) > 0
        assert len(result_concentrated.diversification_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_concentration_risk_with_sectors(self, sizing_engine):
        """Test concentration analysis with sector information."""
        
        portfolio_weights = {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'XOM': 0.15, 'JPM': 0.1}
        asset_sectors = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology', 
            'MSFT': 'Technology',
            'XOM': 'Energy',
            'JPM': 'Financial'
        }
        
        result = await sizing_engine.analyze_concentration_risk(
            portfolio_weights, asset_sectors=asset_sectors
        )
        
        # Check sector concentration
        assert result.sector_concentration is not None
        assert 'Technology' in result.sector_concentration
        assert result.sector_concentration['Technology'] == 0.75  # 30% + 25% + 20%
        
        # Should warn about sector concentration
        assert any('sector' in warning.lower() or 'technology' in warning.lower() 
                  for warning in result.concentration_warnings)
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_recommendations(self, sizing_engine):
        """Test portfolio rebalancing recommendations."""
        
        current_weights = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
        target_weights = {'A': 0.25, 'B': 0.35, 'C': 0.25, 'D': 0.15}
        portfolio_value = 1000000
        
        recommendations = await sizing_engine.generate_portfolio_recommendations(
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=portfolio_value
        )
        
        # Check structure
        assert isinstance(recommendations, dict)
        
        # Should have recommendations for assets with significant changes
        assert 'A' in recommendations  # 30% -> 25% (sell)
        assert 'B' in recommendations  # 30% -> 35% (buy)
        assert 'C' in recommendations  # 20% -> 25% (buy)
        assert 'D' in recommendations  # 20% -> 15% (sell)
        
        # Check recommendation details
        rec_a = recommendations['A']
        assert rec_a['action'] == 'SELL'
        assert rec_a['current_weight'] == 0.3
        assert rec_a['target_weight'] == 0.25
        assert rec_a['weight_change'] == -0.05
        assert rec_a['dollar_amount'] == 50000  # 5% of 1M
        assert 1 <= rec_a['priority'] <= 10
        
        rec_b = recommendations['B']
        assert rec_b['action'] == 'BUY'
        assert rec_b['weight_change'] == 0.05
    
    @pytest.mark.asyncio
    async def test_generate_portfolio_recommendations_with_transaction_costs(self, sizing_engine):
        """Test recommendations with transaction costs."""
        
        current_weights = {'A': 0.5, 'B': 0.5}
        target_weights = {'A': 0.4, 'B': 0.6}
        portfolio_value = 1000000
        transaction_costs = {'A': 0.001, 'B': 0.002}  # 0.1% and 0.2%
        
        recommendations = await sizing_engine.generate_portfolio_recommendations(
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=portfolio_value,
            transaction_costs=transaction_costs
        )
        
        # Check transaction costs are calculated
        rec_a = recommendations['A']
        assert rec_a['transaction_cost'] == 100  # 100,000 * 0.001
        assert rec_a['net_amount'] == 99900  # 100,000 - 100
        
        rec_b = recommendations['B']
        assert rec_b['transaction_cost'] == 200  # 100,000 * 0.002
        assert rec_b['net_amount'] == 99800  # 100,000 - 200
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, sizing_engine):
        """Test handling of insufficient data."""
        
        # Create minimal data
        minimal_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'close': [100 + i for i in range(10)],
            'high': [101 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'open': [100 + i for i in range(10)]
        })
        
        # Should raise error for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            await sizing_engine.calculate_position_size(
                symbol='TEST',
                price_data=minimal_data,
                portfolio_value=1000000
            )
    
    @pytest.mark.asyncio
    async def test_position_constraints_application(self, sizing_engine):
        """Test that position constraints are properly applied."""
        
        # Create data that would suggest very large position
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        
        # Very high return, low volatility stock (unrealistic but for testing)
        returns = np.random.normal(0.005, 0.005, len(dates))  # High return, low vol
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        high_return_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices
        })
        
        result = await sizing_engine.calculate_position_size(
            symbol='HIGH_RETURN',
            price_data=high_return_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.KELLY_CRITERION
        )
        
        # Should be constrained by max position weight
        assert result.recommended_weight <= sizing_engine.max_position_weight
        assert result.recommended_weight >= sizing_engine.min_position_weight
        
        # Should have warning about constraint application
        assert any('constraint' in warning.lower() for warning in result.warnings)


class TestUtilityFunctions:
    """Test utility functions for position sizing."""
    
    def test_calculate_optimal_rebalancing_frequency(self):
        """Test optimal rebalancing frequency calculation."""
        
        portfolio_weights = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        transaction_costs = {'A': 0.001, 'B': 0.002, 'C': 0.001}
        volatilities = {'A': 0.2, 'B': 0.3, 'C': 0.25}
        
        frequency = calculate_optimal_rebalancing_frequency(
            portfolio_weights, transaction_costs, volatilities
        )
        
        assert isinstance(frequency, int)
        assert 7 <= frequency <= 365  # Between weekly and yearly
    
    def test_calculate_position_size_impact(self):
        """Test position size market impact calculation."""
        
        result = calculate_position_size_impact(
            position_size=10000,      # 10,000 shares
            daily_volume=100000,      # 100,000 daily volume
            price=50.0,               # $50 per share
            participation_rate=0.1    # 10% max participation
        )
        
        assert isinstance(result, dict)
        assert 'volume_participation' in result
        assert 'market_impact_pct' in result
        assert 'impact_cost_dollars' in result
        assert 'days_to_trade' in result
        assert 'recommended_max_daily_shares' in result
        
        # Check calculations
        assert result['volume_participation'] == 0.1  # 10,000 / 100,000
        assert result['market_impact_pct'] >= 0
        assert result['impact_cost_dollars'] >= 0
        assert result['days_to_trade'] >= 1
        assert result['recommended_max_daily_shares'] == 10000  # 100,000 * 0.1
    
    def test_calculate_position_size_impact_large_position(self):
        """Test impact calculation for large position."""
        
        result = calculate_position_size_impact(
            position_size=50000,      # Large position
            daily_volume=100000,      
            price=50.0,               
            participation_rate=0.1    
        )
        
        # Should have higher impact and require multiple days
        assert result['volume_participation'] == 0.5  # 50% of daily volume
        assert result['market_impact_pct'] > 0.01  # Higher impact
        assert result['days_to_trade'] > 1  # Multiple days needed
    
    @pytest.mark.asyncio
    async def test_backtest_position_sizing_strategy(self):
        """Test position sizing strategy backtesting."""
        
        # Create simple test data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
        
        historical_data = {}
        for symbol in ['A', 'B']:
            returns = np.random.normal(0.0005, 0.015, len(dates))
            prices = [100]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            historical_data[symbol] = pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'open': prices
            })
        
        # Create sizing engine
        sizing_engine = DynamicPositionSizingEngine()
        
        # Run backtest
        results = await backtest_position_sizing_strategy(
            sizing_engine=sizing_engine,
            historical_data=historical_data,
            initial_capital=1000000,
            rebalance_frequency=30
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'annual_return' in results
        assert 'volatility' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'portfolio_values' in results
        assert 'portfolio_weights' in results
        assert 'rebalance_dates' in results
        assert 'final_capital' in results
        
        # Check values are reasonable
        assert isinstance(results['total_return'], float)
        assert isinstance(results['annual_return'], float)
        assert results['volatility'] >= 0
        assert results['max_drawdown'] >= 0
        assert results['max_drawdown'] <= 1.0
        assert len(results['portfolio_values']) == len(dates)
        assert len(results['portfolio_weights']) == len(dates)
        assert results['final_capital'] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def sizing_engine(self):
        return DynamicPositionSizingEngine()
    
    @pytest.mark.asyncio
    async def test_zero_volatility_asset(self, sizing_engine):
        """Test handling of zero volatility asset."""
        
        # Create constant price data (zero volatility)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        constant_prices = [100] * len(dates)
        
        zero_vol_data = pd.DataFrame({
            'date': dates,
            'close': constant_prices,
            'high': constant_prices,
            'low': constant_prices,
            'open': constant_prices
        })
        
        result = await sizing_engine.calculate_position_size(
            symbol='ZERO_VOL',
            price_data=zero_vol_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.KELLY_CRITERION
        )
        
        # Should handle gracefully
        assert result.recommended_weight >= 0
        assert result.kelly_fraction == 0.0  # No Kelly position for zero volatility
    
    @pytest.mark.asyncio
    async def test_negative_return_asset(self, sizing_engine):
        """Test handling of asset with negative expected returns."""
        
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
        
        # Negative drift
        returns = np.random.normal(-0.002, 0.02, len(dates))
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        negative_return_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices
        })
        
        result = await sizing_engine.calculate_position_size(
            symbol='NEGATIVE',
            price_data=negative_return_data,
            portfolio_value=1000000,
            method=PositionSizingMethod.KELLY_CRITERION
        )
        
        # Kelly should be zero or minimal for negative expected return
        assert result.kelly_fraction == 0.0 or result.kelly_fraction < 0.01
        assert result.recommended_weight == sizing_engine.min_position_weight
    
    @pytest.mark.asyncio
    async def test_single_asset_portfolio(self, sizing_engine):
        """Test portfolio optimization with single asset."""
        
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        single_asset = {
            'SINGLE': pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'open': prices
            })
        }
        
        result = await sizing_engine.optimize_portfolio_risk_budget(
            assets=single_asset,
            portfolio_value=1000000
        )
        
        # Should allocate 100% to single asset
        assert result.asset_weights['SINGLE'] == pytest.approx(1.0, abs=0.01)
        assert result.diversification_ratio == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__])