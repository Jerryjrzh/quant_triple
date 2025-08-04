"""
Unit tests for Advanced Risk Metrics Calculator

Tests enhanced risk metrics including Sharpe/Sortino/Calmar ratios with confidence intervals,
beta calculations, liquidity risk scoring, and seasonal risk integration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from stock_analysis_system.analysis.advanced_risk_metrics import (
    AdvancedRiskMetricsCalculator,
    AdvancedRiskMetrics,
    RiskAdjustmentMethod,
    SeasonalRiskLevel,
    calculate_comprehensive_risk_profile,
    _calculate_overall_risk_score,
    _determine_overall_risk_level,
    _identify_key_risk_factors,
    _generate_risk_recommendations
)


class TestAdvancedRiskMetricsCalculator:
    """Test cases for Advanced Risk Metrics Calculator"""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return AdvancedRiskMetricsCalculator(
            risk_free_rate=0.03,
            confidence_level=0.95,
            bootstrap_iterations=100,  # Reduced for faster testing
            seasonal_window_years=3
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        # Generate realistic stock price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [100]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
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
        volumes = np.random.lognormal(mean=10, sigma=1, size=len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'volume': volumes
        })
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data for testing."""
        np.random.seed(123)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = [1000]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'date': dates,
            'close': prices
        })
    
    @pytest.fixture
    def mock_spring_festival_engine(self):
        """Create mock Spring Festival engine."""
        mock_engine = Mock()
        mock_engine.analyze_seasonal_patterns = AsyncMock(return_value={
            'seasonal_volatility': 0.25,
            'current_risk_level': 'moderate'
        })
        return mock_engine
    
    @pytest.mark.asyncio
    async def test_calculate_advanced_metrics(self, calculator, sample_price_data, 
                                            sample_benchmark_data, sample_volume_data,
                                            mock_spring_festival_engine):
        """Test comprehensive advanced metrics calculation."""
        
        result = await calculator.calculate_advanced_metrics(
            price_data=sample_price_data,
            benchmark_data=sample_benchmark_data,
            volume_data=sample_volume_data,
            spring_festival_engine=mock_spring_festival_engine
        )
        
        # Verify result structure
        assert isinstance(result, AdvancedRiskMetrics)
        
        # Check ratio metrics
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.calmar_ratio, float)
        
        # Check confidence intervals
        if result.sharpe_confidence_interval:
            assert len(result.sharpe_confidence_interval) == 2
            assert result.sharpe_confidence_interval[0] <= result.sharpe_confidence_interval[1]
        
        # Check market risk metrics
        assert result.beta is not None
        assert isinstance(result.beta, float)
        assert result.alpha is not None
        assert result.tracking_error is not None
        assert result.information_ratio is not None
        
        # Check liquidity metrics
        assert 0 <= result.liquidity_risk_score <= 100
        assert isinstance(result.liquidity_risk_level, str)
        
        # Check seasonal metrics
        assert 0 <= result.seasonal_risk_score <= 100
        assert isinstance(result.seasonal_risk_level, SeasonalRiskLevel)
        assert result.spring_festival_risk_adjustment > 0
        
        # Check additional metrics
        assert result.omega_ratio is None or isinstance(result.omega_ratio, float)
        assert result.kappa_3 is None or isinstance(result.kappa_3, float)
        assert result.tail_ratio is None or isinstance(result.tail_ratio, float)
        
        assert isinstance(result.calculation_date, datetime)
    
    @pytest.mark.asyncio
    async def test_enhanced_sharpe_ratio_calculation(self, calculator, sample_price_data):
        """Test enhanced Sharpe ratio with confidence intervals."""
        
        returns = calculator._calculate_returns(sample_price_data)
        sharpe, confidence_interval = await calculator._calculate_enhanced_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        
        if confidence_interval:
            assert len(confidence_interval) == 2
            assert confidence_interval[0] <= confidence_interval[1]
            # Sharpe ratio should be within confidence interval (approximately)
            assert confidence_interval[0] <= sharpe * 2  # Allow some tolerance
            assert sharpe * 0.5 <= confidence_interval[1]
    
    @pytest.mark.asyncio
    async def test_enhanced_sortino_ratio_calculation(self, calculator, sample_price_data):
        """Test enhanced Sortino ratio with confidence intervals."""
        
        returns = calculator._calculate_returns(sample_price_data)
        sortino, confidence_interval = await calculator._calculate_enhanced_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        
        if confidence_interval and not np.isinf(sortino):
            assert len(confidence_interval) == 2
            assert confidence_interval[0] <= confidence_interval[1]
    
    @pytest.mark.asyncio
    async def test_enhanced_calmar_ratio_calculation(self, calculator, sample_price_data):
        """Test enhanced Calmar ratio with confidence intervals."""
        
        returns = calculator._calculate_returns(sample_price_data)
        calmar, confidence_interval = await calculator._calculate_enhanced_calmar_ratio(
            sample_price_data, returns
        )
        
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)
        
        if confidence_interval and not np.isinf(calmar):
            assert len(confidence_interval) == 2
            assert confidence_interval[0] <= confidence_interval[1]
    
    @pytest.mark.asyncio
    async def test_market_risk_metrics_calculation(self, calculator, sample_price_data, 
                                                 sample_benchmark_data):
        """Test comprehensive market risk metrics."""
        
        returns = calculator._calculate_returns(sample_price_data)
        beta, beta_ci, alpha, tracking_error, info_ratio = await calculator._calculate_market_risk_metrics(
            returns, sample_benchmark_data
        )
        
        # Check beta
        assert beta is not None
        assert isinstance(beta, float)
        assert not np.isnan(beta)
        
        # Check beta confidence interval
        if beta_ci:
            assert len(beta_ci) == 2
            assert beta_ci[0] <= beta_ci[1]
        
        # Check alpha (Jensen's alpha)
        assert alpha is not None
        assert isinstance(alpha, float)
        assert not np.isnan(alpha)
        
        # Check tracking error
        assert tracking_error is not None
        assert isinstance(tracking_error, float)
        assert tracking_error >= 0
        
        # Check information ratio
        assert info_ratio is not None
        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)
    
    @pytest.mark.asyncio
    async def test_liquidity_risk_metrics_calculation(self, calculator, sample_price_data, 
                                                    sample_volume_data):
        """Test liquidity risk metrics calculation."""
        
        result = await calculator._calculate_liquidity_risk_metrics(
            sample_price_data, sample_volume_data
        )
        
        # Check structure
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'level' in result
        assert 'spread_proxy' in result
        assert 'impact_score' in result
        
        # Check values
        assert 0 <= result['score'] <= 100
        assert isinstance(result['level'], str)
        assert result['spread_proxy'] is None or isinstance(result['spread_proxy'], float)
        assert result['impact_score'] is None or isinstance(result['impact_score'], float)
    
    @pytest.mark.asyncio
    async def test_price_based_liquidity_metrics(self, calculator, sample_price_data):
        """Test liquidity metrics using only price data."""
        
        result = await calculator._calculate_price_based_liquidity_metrics(sample_price_data)
        
        assert isinstance(result, dict)
        assert 0 <= result['score'] <= 100
        assert 'Estimated' in result['level']  # Should indicate estimation
        assert result['spread_proxy'] is not None
        assert result['impact_score'] is None  # Not available without volume
    
    @pytest.mark.asyncio
    async def test_seasonal_risk_metrics_calculation(self, calculator, sample_price_data,
                                                   mock_spring_festival_engine):
        """Test seasonal risk metrics calculation."""
        
        returns = calculator._calculate_returns(sample_price_data)
        result = await calculator._calculate_seasonal_risk_metrics(
            sample_price_data, returns, mock_spring_festival_engine
        )
        
        # Check structure
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'level' in result
        assert 'sf_adjustment' in result
        assert 'seasonal_volatility' in result
        
        # Check values
        assert 0 <= result['score'] <= 100
        assert isinstance(result['level'], SeasonalRiskLevel)
        assert result['sf_adjustment'] > 0
        assert isinstance(result['seasonal_volatility'], dict)
    
    @pytest.mark.asyncio
    async def test_seasonal_risk_without_engine(self, calculator, sample_price_data):
        """Test seasonal risk calculation without Spring Festival engine."""
        
        returns = calculator._calculate_returns(sample_price_data)
        result = await calculator._calculate_seasonal_risk_metrics(
            sample_price_data, returns, None
        )
        
        # Should still return valid results with defaults
        assert isinstance(result, dict)
        assert result['score'] == 50.0  # Default moderate risk
        assert result['level'] == SeasonalRiskLevel.MODERATE
        assert result['sf_adjustment'] == 1.0  # No adjustment
    
    @pytest.mark.asyncio
    async def test_omega_ratio_calculation(self, calculator, sample_price_data):
        """Test Omega ratio calculation."""
        
        returns = calculator._calculate_returns(sample_price_data)
        omega = await calculator._calculate_omega_ratio(returns)
        
        if omega is not None:
            assert isinstance(omega, float)
            assert omega >= 0
            assert not np.isnan(omega)
    
    @pytest.mark.asyncio
    async def test_kappa_3_calculation(self, calculator, sample_price_data):
        """Test Kappa 3 calculation."""
        
        returns = calculator._calculate_returns(sample_price_data)
        kappa_3 = await calculator._calculate_kappa_3(returns)
        
        if kappa_3 is not None:
            assert isinstance(kappa_3, float)
            assert not np.isnan(kappa_3)
    
    @pytest.mark.asyncio
    async def test_tail_ratio_calculation(self, calculator, sample_price_data):
        """Test tail ratio calculation."""
        
        returns = calculator._calculate_returns(sample_price_data)
        tail_ratio = await calculator._calculate_tail_ratio(returns)
        
        if tail_ratio is not None:
            assert isinstance(tail_ratio, float)
            assert tail_ratio > 0
            assert not np.isnan(tail_ratio)
    
    @pytest.mark.asyncio
    async def test_spring_festival_dates_calculation(self, calculator):
        """Test Spring Festival dates calculation."""
        
        sf_dates = await calculator._get_spring_festival_dates(2020, 2025)
        
        assert isinstance(sf_dates, list)
        assert len(sf_dates) == 6  # 2020-2025 inclusive
        assert all(isinstance(date, datetime) for date in sf_dates)
        
        # Check dates are in reasonable range (January-February)
        for date in sf_dates:
            assert date.month in [1, 2]
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, calculator):
        """Test handling of insufficient data."""
        
        # Create minimal data
        minimal_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'close': [100 + i for i in range(10)],
            'high': [101 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'open': [100 + i for i in range(10)]
        })
        
        # Should handle gracefully without errors
        result = await calculator.calculate_advanced_metrics(
            price_data=minimal_data,
            benchmark_data=None,
            volume_data=None,
            spring_festival_engine=None
        )
        
        assert isinstance(result, AdvancedRiskMetrics)
        # Many metrics should be None or default values due to insufficient data
        assert result.beta is None
        assert result.liquidity_risk_score >= 0


class TestComprehensiveRiskProfile:
    """Test comprehensive risk profile calculation."""
    
    @pytest.fixture
    def sample_data_set(self):
        """Create complete sample data set."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Stock data
        stock_returns = np.random.normal(0.001, 0.025, len(dates))
        stock_prices = [100]
        for ret in stock_returns[1:]:
            stock_prices.append(stock_prices[-1] * (1 + ret))
        
        stock_data = pd.DataFrame({
            'date': dates,
            'close': stock_prices,
            'high': [p * 1.01 for p in stock_prices],
            'low': [p * 0.99 for p in stock_prices],
            'open': stock_prices
        })
        
        # Benchmark data
        bench_returns = np.random.normal(0.0005, 0.015, len(dates))
        bench_prices = [1000]
        for ret in bench_returns[1:]:
            bench_prices.append(bench_prices[-1] * (1 + ret))
        
        benchmark_data = pd.DataFrame({
            'date': dates,
            'close': bench_prices
        })
        
        # Volume data
        volume_data = pd.DataFrame({
            'date': dates,
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        
        return stock_data, benchmark_data, volume_data
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_profile_calculation(self, sample_data_set):
        """Test comprehensive risk profile calculation."""
        
        stock_data, benchmark_data, volume_data = sample_data_set
        
        result = await calculate_comprehensive_risk_profile(
            price_data=stock_data,
            benchmark_data=benchmark_data,
            volume_data=volume_data,
            spring_festival_engine=None,
            risk_engine=None
        )
        
        # Check structure
        assert isinstance(result, dict)
        assert 'basic_metrics' in result
        assert 'advanced_metrics' in result
        assert 'summary' in result
        
        # Check summary
        summary = result['summary']
        assert 'overall_risk_score' in summary
        assert 'risk_level' in summary
        assert 'key_risk_factors' in summary
        assert 'recommendations' in summary
        
        # Check values
        assert 0 <= summary['overall_risk_score'] <= 100
        assert isinstance(summary['risk_level'], str)
        assert isinstance(summary['key_risk_factors'], list)
        assert isinstance(summary['recommendations'], list)
    
    def test_overall_risk_score_calculation(self):
        """Test overall risk score calculation."""
        
        # Create mock metrics
        basic_metrics = Mock()
        basic_metrics.var_results = {'historical': Mock(var_95=0.03)}
        basic_metrics.volatility_results = {'historical': Mock(annualized_volatility=0.25)}
        basic_metrics.max_drawdown = 0.15
        basic_metrics.sharpe_ratio = 1.2
        
        advanced_metrics = Mock()
        advanced_metrics.liquidity_risk_score = 40.0
        advanced_metrics.seasonal_risk_score = 55.0
        
        score = _calculate_overall_risk_score(basic_metrics, advanced_metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_risk_level_determination(self):
        """Test risk level determination."""
        
        # Test different score ranges
        basic_metrics = Mock()
        advanced_metrics = Mock()
        
        # Mock the score calculation to return specific values
        with patch('stock_analysis_system.analysis.advanced_risk_metrics._calculate_overall_risk_score') as mock_score:
            mock_score.return_value = 15.0
            level = _determine_overall_risk_level(basic_metrics, advanced_metrics)
            assert level == "Very Low Risk"
            
            mock_score.return_value = 35.0
            level = _determine_overall_risk_level(basic_metrics, advanced_metrics)
            assert level == "Low Risk"
            
            mock_score.return_value = 55.0
            level = _determine_overall_risk_level(basic_metrics, advanced_metrics)
            assert level == "Moderate Risk"
            
            mock_score.return_value = 75.0
            level = _determine_overall_risk_level(basic_metrics, advanced_metrics)
            assert level == "High Risk"
            
            mock_score.return_value = 95.0
            level = _determine_overall_risk_level(basic_metrics, advanced_metrics)
            assert level == "Very High Risk"
    
    def test_key_risk_factors_identification(self):
        """Test key risk factors identification."""
        
        # Create mock metrics with high risk values
        basic_metrics = Mock()
        basic_metrics.var_results = {'historical': Mock(var_95=0.08)}  # High VaR
        basic_metrics.volatility_results = {'historical': Mock(annualized_volatility=0.5)}  # High volatility
        basic_metrics.max_drawdown = 0.4  # Large drawdown
        basic_metrics.sharpe_ratio = 0.3  # Poor Sharpe ratio
        
        advanced_metrics = Mock()
        advanced_metrics.liquidity_risk_score = 80.0  # Low liquidity
        advanced_metrics.seasonal_risk_score = 85.0  # High seasonal risk
        
        factors = _identify_key_risk_factors(basic_metrics, advanced_metrics)
        
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert "High Value at Risk" in factors
        assert "High Volatility" in factors
        assert "Large Historical Drawdowns" in factors
        assert "Low Liquidity" in factors
        assert "High Seasonal Risk" in factors
        assert "Poor Risk-Adjusted Returns" in factors
    
    def test_risk_recommendations_generation(self):
        """Test risk recommendations generation."""
        
        # Create mock metrics
        basic_metrics = Mock()
        basic_metrics.var_results = {'historical': Mock(var_95=0.05)}
        basic_metrics.max_drawdown = 0.3
        basic_metrics.sharpe_ratio = 0.8
        
        advanced_metrics = Mock()
        advanced_metrics.liquidity_risk_score = 70.0
        advanced_metrics.seasonal_risk_score = 75.0
        
        recommendations = _generate_risk_recommendations(basic_metrics, advanced_metrics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations are strings
        assert all(isinstance(rec, str) for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])