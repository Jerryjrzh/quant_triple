"""
Unit tests for Enhanced Risk Management Engine

Tests comprehensive VaR calculations, volatility measures, and risk metrics
as specified in task 5.1.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch

from stock_analysis_system.analysis.risk_management_engine import (
    EnhancedRiskManagementEngine,
    VaRMethod,
    VolatilityMethod,
    VaRResult,
    VolatilityResult,
    RiskMetrics,
    calculate_portfolio_var,
    calculate_component_var,
    stress_test_portfolio
)


class TestEnhancedRiskManagementEngine:
    """Test cases for Enhanced Risk Management Engine"""
    
    @pytest.fixture
    def risk_engine(self):
        """Create a risk management engine instance for testing."""
        return EnhancedRiskManagementEngine(
            confidence_levels=[0.95, 0.99],
            var_window=252,
            volatility_window=30,
            monte_carlo_simulations=1000  # Reduced for faster testing
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate realistic stock price data with some volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [100]  # Starting price
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Add some realistic OHLC data
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
        })
        
        return df
    
    @pytest.fixture
    def sample_volume_data(self):
        """Create sample volume data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate realistic volume data
        volumes = np.random.lognormal(mean=10, sigma=1, size=len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'volume': volumes
        })
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data for testing."""
        np.random.seed(123)  # Different seed for benchmark
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate benchmark returns with lower volatility
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = [1000]  # Starting price
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'date': dates,
            'close': prices
        })
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_metrics_calculation(self, risk_engine, sample_price_data):
        """Test comprehensive risk metrics calculation."""
        
        result = await risk_engine.calculate_comprehensive_risk_metrics(sample_price_data)
        
        # Verify result structure
        assert isinstance(result, RiskMetrics)
        assert isinstance(result.var_results, dict)
        assert isinstance(result.volatility_results, dict)
        
        # Check that all VaR methods were calculated
        expected_var_methods = [method.value for method in VaRMethod]
        for method in expected_var_methods:
            assert method in result.var_results
            var_result = result.var_results[method]
            assert isinstance(var_result, VaRResult)
            assert var_result.var_95 > 0
            assert var_result.var_99 > var_result.var_95  # 99% VaR should be higher
            assert var_result.cvar_95 >= var_result.var_95  # CVaR should be >= VaR
            assert var_result.cvar_99 >= var_result.var_99
        
        # Check that all volatility methods were calculated
        expected_vol_methods = [method.value for method in VolatilityMethod]
        for method in expected_vol_methods:
            if method in result.volatility_results:  # Some methods might fail
                vol_result = result.volatility_results[method]
                assert isinstance(vol_result, VolatilityResult)
                assert vol_result.daily_volatility > 0
                assert vol_result.annualized_volatility > vol_result.daily_volatility
        
        # Check additional risk metrics
        assert 0 <= result.max_drawdown <= 1  # Max drawdown should be between 0 and 1
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.calmar_ratio, float)
    
    @pytest.mark.asyncio
    async def test_historical_var_calculation(self, risk_engine, sample_price_data):
        """Test historical VaR calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        result = await risk_engine._calculate_historical_var(returns)
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99
        assert isinstance(result.calculation_date, datetime)
    
    @pytest.mark.asyncio
    async def test_parametric_var_calculation(self, risk_engine, sample_price_data):
        """Test parametric VaR calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        result = await risk_engine._calculate_parametric_var(returns)
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99
    
    @pytest.mark.asyncio
    async def test_monte_carlo_var_calculation(self, risk_engine, sample_price_data):
        """Test Monte Carlo VaR calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        result = await risk_engine._calculate_monte_carlo_var(returns)
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
    
    @pytest.mark.asyncio
    async def test_historical_volatility_calculation(self, risk_engine, sample_price_data):
        """Test historical volatility calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        result = await risk_engine._calculate_historical_volatility(returns)
        
        assert isinstance(result, VolatilityResult)
        assert result.method == VolatilityMethod.HISTORICAL
        assert result.daily_volatility > 0
        assert result.annualized_volatility > result.daily_volatility
        assert result.window_size > 0
        assert result.realized_volatility is not None
    
    @pytest.mark.asyncio
    async def test_ewma_volatility_calculation(self, risk_engine, sample_price_data):
        """Test EWMA volatility calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        result = await risk_engine._calculate_ewma_volatility(returns)
        
        assert isinstance(result, VolatilityResult)
        assert result.method == VolatilityMethod.EWMA
        assert result.daily_volatility > 0
        assert result.annualized_volatility > result.daily_volatility
        assert result.window_size > 0
    
    def test_max_drawdown_calculation(self, risk_engine, sample_price_data):
        """Test maximum drawdown calculation."""
        
        max_dd = risk_engine._calculate_max_drawdown(sample_price_data)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1  # Should be between 0 and 1
    
    def test_sharpe_ratio_calculation(self, risk_engine, sample_price_data):
        """Test Sharpe ratio calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        sharpe = risk_engine._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        # Sharpe ratio can be negative, so no bounds check
    
    def test_sortino_ratio_calculation(self, risk_engine, sample_price_data):
        """Test Sortino ratio calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        sortino = risk_engine._calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        # Sortino ratio can be negative or infinite, so no bounds check
    
    def test_calmar_ratio_calculation(self, risk_engine, sample_price_data):
        """Test Calmar ratio calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        max_dd = risk_engine._calculate_max_drawdown(sample_price_data)
        calmar = risk_engine._calculate_calmar_ratio(returns, max_dd)
        
        assert isinstance(calmar, float)
    
    @pytest.mark.asyncio
    async def test_beta_calculation(self, risk_engine, sample_price_data, sample_benchmark_data):
        """Test beta calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        beta = await risk_engine._calculate_beta(returns, sample_benchmark_data)
        
        assert isinstance(beta, float)
        # Beta can be any real number, so no bounds check
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_score_calculation(self, risk_engine, sample_price_data, sample_volume_data):
        """Test liquidity risk score calculation."""
        
        score = await risk_engine._calculate_liquidity_risk_score(sample_price_data, sample_volume_data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100  # Score should be between 0 and 100
    
    def test_returns_calculation(self, risk_engine, sample_price_data):
        """Test returns calculation."""
        
        returns = risk_engine._calculate_returns(sample_price_data)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_price_data) - 1  # One less due to pct_change
        assert not returns.isna().all()  # Should not be all NaN
    
    def test_data_validation(self, risk_engine):
        """Test input data validation."""
        
        # Test missing required columns
        invalid_data = pd.DataFrame({'price': [100, 101, 102]})
        with pytest.raises(ValueError, match="Missing required columns"):
            risk_engine._validate_price_data(invalid_data)
        
        # Test insufficient data
        insufficient_data = pd.DataFrame({'close': [100, 101]})
        with pytest.raises(ValueError, match="Insufficient data"):
            risk_engine._validate_price_data(insufficient_data)
        
        # Test negative prices
        negative_data = pd.DataFrame({'close': [100, -101, 102] + [100] * 30})
        with pytest.raises(ValueError, match="Close prices must be positive"):
            risk_engine._validate_price_data(negative_data)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, risk_engine):
        """Test handling of insufficient data scenarios."""
        
        # Create minimal data
        minimal_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'close': [100 + i for i in range(10)]
        })
        
        returns = risk_engine._calculate_returns(minimal_data)
        
        # Should raise errors for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            await risk_engine._calculate_historical_var(returns)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            await risk_engine._calculate_parametric_var(returns)


class TestPortfolioRiskFunctions:
    """Test portfolio-level risk functions."""
    
    def test_portfolio_var_calculation(self):
        """Test portfolio VaR calculation."""
        
        individual_vars = [0.02, 0.03, 0.025]  # Individual VaRs
        correlations = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        weights = np.array([0.4, 0.4, 0.2])
        
        portfolio_var = calculate_portfolio_var(individual_vars, correlations, weights)
        
        assert isinstance(portfolio_var, float)
        assert portfolio_var > 0
        # Portfolio VaR should be less than weighted average due to diversification
        weighted_avg_var = np.dot(weights, individual_vars)
        assert portfolio_var <= weighted_avg_var
    
    def test_component_var_calculation(self):
        """Test component VaR calculation."""
        
        individual_vars = [0.02, 0.03, 0.025]
        correlations = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        weights = np.array([0.4, 0.4, 0.2])
        
        component_vars = calculate_component_var(individual_vars, correlations, weights)
        
        assert isinstance(component_vars, np.ndarray)
        assert len(component_vars) == len(weights)
        assert all(isinstance(cv, (int, float)) for cv in component_vars)
        
        # Component VaRs should sum to portfolio VaR
        portfolio_var = calculate_portfolio_var(individual_vars, correlations, weights)
        assert abs(component_vars.sum() - portfolio_var) < 1e-10
    
    def test_dimension_mismatch_errors(self):
        """Test error handling for dimension mismatches."""
        
        individual_vars = [0.02, 0.03]  # 2 assets
        correlations = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2x2
        weights = np.array([0.4, 0.4, 0.2])  # 3 assets - mismatch!
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            calculate_portfolio_var(individual_vars, correlations, weights)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            calculate_component_var(individual_vars, correlations, weights)


class TestStressTesting:
    """Test stress testing functionality."""
    
    @pytest.mark.asyncio
    async def test_stress_test_portfolio(self):
        """Test portfolio stress testing."""
        
        # Create mock risk engine
        risk_engine = Mock()
        
        # Create mock risk metrics
        mock_var_result = VaRResult(0.05, 0.08, 0.06, 0.09, VaRMethod.HISTORICAL)
        mock_vol_result = VolatilityResult(0.02, 0.32, VolatilityMethod.HISTORICAL, 30)
        mock_risk_metrics = RiskMetrics(
            var_results={'historical': mock_var_result},
            volatility_results={'historical': mock_vol_result},
            max_drawdown=0.15,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8
        )
        
        risk_engine.calculate_comprehensive_risk_metrics.return_value = mock_risk_metrics
        
        # Create sample portfolio data
        portfolio_data = {
            'AAPL': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'close': np.random.uniform(150, 200, 100),
                'high': np.random.uniform(150, 200, 100),
                'low': np.random.uniform(150, 200, 100),
                'open': np.random.uniform(150, 200, 100)
            }),
            'GOOGL': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'close': np.random.uniform(2000, 3000, 100),
                'high': np.random.uniform(2000, 3000, 100),
                'low': np.random.uniform(2000, 3000, 100),
                'open': np.random.uniform(2000, 3000, 100)
            })
        }
        
        # Define stress scenarios
        stress_scenarios = {
            'market_crash': {'AAPL': -0.3, 'GOOGL': -0.25},  # 30% and 25% declines
            'tech_selloff': {'AAPL': -0.2, 'GOOGL': -0.35}   # 20% and 35% declines
        }
        
        # Run stress test
        results = await stress_test_portfolio(risk_engine, portfolio_data, stress_scenarios)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'market_crash' in results
        assert 'tech_selloff' in results
        
        for scenario_name, scenario_results in results.items():
            assert 'AAPL' in scenario_results
            assert 'GOOGL' in scenario_results
            
            for symbol, metrics in scenario_results.items():
                assert 'var_95' in metrics
                assert 'max_drawdown' in metrics
                assert 'volatility' in metrics


if __name__ == "__main__":
    pytest.main([__file__])