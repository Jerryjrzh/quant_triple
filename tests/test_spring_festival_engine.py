"""Tests for Spring Festival Analysis Engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, Mock

from stock_analysis_system.analysis.spring_festival_engine import (
    ChineseCalendar,
    SpringFestivalWindow,
    AlignedDataPoint,
    AlignedTimeSeries,
    SeasonalPattern,
    SpringFestivalAlignmentEngine
)


class TestChineseCalendar:
    """Test Chinese calendar utilities."""
    
    def test_get_spring_festival_known_years(self):
        """Test getting Spring Festival dates for known years."""
        # Test some known dates
        assert ChineseCalendar.get_spring_festival(2024) == date(2024, 2, 10)
        assert ChineseCalendar.get_spring_festival(2023) == date(2023, 1, 22)
        assert ChineseCalendar.get_spring_festival(2022) == date(2022, 2, 1)
    
    def test_get_spring_festival_unknown_year(self):
        """Test getting Spring Festival date for unknown year."""
        assert ChineseCalendar.get_spring_festival(1900) is None
        assert ChineseCalendar.get_spring_festival(2050) is None
    
    def test_get_available_years(self):
        """Test getting available years."""
        years = ChineseCalendar.get_available_years()
        assert isinstance(years, list)
        assert len(years) > 10
        assert 2024 in years
        assert 2023 in years
        assert years == sorted(years)  # Should be sorted
    
    def test_is_spring_festival_period(self):
        """Test checking if date is in Spring Festival period."""
        # Test date within period
        sf_2024 = date(2024, 2, 10)
        test_date = date(2024, 2, 5)  # 5 days before
        
        is_in_period, sf_date = ChineseCalendar.is_spring_festival_period(test_date, window_days=15)
        assert is_in_period is True
        assert sf_date == sf_2024
        
        # Test date outside period
        test_date = date(2024, 1, 1)  # Too far before
        is_in_period, sf_date = ChineseCalendar.is_spring_festival_period(test_date, window_days=15)
        assert is_in_period is False
        assert sf_date == sf_2024
    
    def test_get_trading_days_to_spring_festival(self):
        """Test calculating trading days to Spring Festival."""
        sf_2024 = date(2024, 2, 10)  # Saturday
        
        # Test from a weekday
        test_date = date(2024, 2, 5)  # Monday, 5 days before
        trading_days = ChineseCalendar.get_trading_days_to_spring_festival(test_date)
        assert trading_days == 5  # 5 calendar days, approximately 4-5 trading days
        
        # Test unknown year
        test_date = date(1900, 1, 1)
        trading_days = ChineseCalendar.get_trading_days_to_spring_festival(test_date)
        assert trading_days is None


class TestSpringFestivalWindow:
    """Test Spring Festival window."""
    
    def test_window_creation(self):
        """Test creating Spring Festival window."""
        sf_date = date(2024, 2, 10)
        window = SpringFestivalWindow(
            year=2024,
            spring_festival_date=sf_date,
            start_date=sf_date - timedelta(days=30),
            end_date=sf_date + timedelta(days=30),
            window_days=30
        )
        
        assert window.year == 2024
        assert window.spring_festival_date == sf_date
        assert window.window_days == 30
    
    def test_contains_date(self):
        """Test checking if date is in window."""
        sf_date = date(2024, 2, 10)
        window = SpringFestivalWindow(
            year=2024,
            spring_festival_date=sf_date,
            start_date=date(2024, 1, 15),
            end_date=date(2024, 3, 15),
            window_days=30
        )
        
        # Date in window
        assert window.contains_date(date(2024, 2, 1)) is True
        assert window.contains_date(sf_date) is True
        
        # Date outside window
        assert window.contains_date(date(2024, 1, 1)) is False
        assert window.contains_date(date(2024, 4, 1)) is False
    
    def test_days_from_spring_festival(self):
        """Test calculating days from Spring Festival."""
        sf_date = date(2024, 2, 10)
        window = SpringFestivalWindow(
            year=2024,
            spring_festival_date=sf_date,
            start_date=date(2024, 1, 15),
            end_date=date(2024, 3, 15),
            window_days=30
        )
        
        # Before Spring Festival
        assert window.days_from_spring_festival(date(2024, 2, 5)) == -5
        
        # After Spring Festival
        assert window.days_from_spring_festival(date(2024, 2, 15)) == 5
        
        # On Spring Festival
        assert window.days_from_spring_festival(sf_date) == 0


class TestAlignedDataPoint:
    """Test aligned data point."""
    
    def test_data_point_creation(self):
        """Test creating aligned data point."""
        point = AlignedDataPoint(
            original_date=date(2024, 2, 5),
            relative_day=-5,
            spring_festival_date=date(2024, 2, 10),
            year=2024,
            price=100.0,
            volume=1000,
            normalized_price=2.5
        )
        
        assert point.relative_day == -5
        assert point.is_before_spring_festival is True
        assert point.is_after_spring_festival is False
        assert point.price == 100.0
    
    def test_position_properties(self):
        """Test position properties."""
        # Before Spring Festival
        point_before = AlignedDataPoint(
            original_date=date(2024, 2, 5),
            relative_day=-5,
            spring_festival_date=date(2024, 2, 10),
            year=2024,
            price=100.0
        )
        assert point_before.is_before_spring_festival is True
        assert point_before.is_after_spring_festival is False
        
        # After Spring Festival
        point_after = AlignedDataPoint(
            original_date=date(2024, 2, 15),
            relative_day=5,
            spring_festival_date=date(2024, 2, 10),
            year=2024,
            price=105.0
        )
        assert point_after.is_before_spring_festival is False
        assert point_after.is_after_spring_festival is True


class TestAlignedTimeSeries:
    """Test aligned time series."""
    
    @pytest.fixture
    def sample_aligned_data(self):
        """Create sample aligned time series."""
        points = [
            AlignedDataPoint(date(2024, 2, 5), -5, date(2024, 2, 10), 2024, 100.0, normalized_price=-2.0),
            AlignedDataPoint(date(2024, 2, 10), 0, date(2024, 2, 10), 2024, 102.0, normalized_price=0.0),
            AlignedDataPoint(date(2024, 2, 15), 5, date(2024, 2, 10), 2024, 98.0, normalized_price=-4.0),
            AlignedDataPoint(date(2023, 1, 17), -5, date(2023, 1, 22), 2023, 95.0, normalized_price=-7.0),
            AlignedDataPoint(date(2023, 1, 22), 0, date(2023, 1, 22), 2023, 97.0, normalized_price=-5.0),
            AlignedDataPoint(date(2023, 1, 27), 5, date(2023, 1, 22), 2023, 103.0, normalized_price=1.0),
        ]
        
        return AlignedTimeSeries(
            symbol="000001.SZ",
            data_points=points,
            window_days=30,
            years=[2023, 2024],
            baseline_price=102.0
        )
    
    def test_time_series_creation(self, sample_aligned_data):
        """Test creating aligned time series."""
        assert sample_aligned_data.symbol == "000001.SZ"
        assert len(sample_aligned_data.data_points) == 6
        assert sample_aligned_data.years == [2023, 2024]
        assert sample_aligned_data.baseline_price == 102.0
    
    def test_get_year_data(self, sample_aligned_data):
        """Test getting data for specific year."""
        year_2024_data = sample_aligned_data.get_year_data(2024)
        assert len(year_2024_data) == 3
        assert all(point.year == 2024 for point in year_2024_data)
        
        year_2023_data = sample_aligned_data.get_year_data(2023)
        assert len(year_2023_data) == 3
        assert all(point.year == 2023 for point in year_2023_data)
    
    def test_get_relative_day_data(self, sample_aligned_data):
        """Test getting data for specific relative day."""
        day_minus_5_data = sample_aligned_data.get_relative_day_data(-5)
        assert len(day_minus_5_data) == 2
        assert all(point.relative_day == -5 for point in day_minus_5_data)
        
        day_0_data = sample_aligned_data.get_relative_day_data(0)
        assert len(day_0_data) == 2
        assert all(point.relative_day == 0 for point in day_0_data)
    
    def test_get_before_after_data(self, sample_aligned_data):
        """Test getting before/after Spring Festival data."""
        before_data = sample_aligned_data.get_before_spring_festival()
        assert len(before_data) == 2
        assert all(point.is_before_spring_festival for point in before_data)
        
        after_data = sample_aligned_data.get_after_spring_festival()
        assert len(after_data) == 2
        assert all(point.is_after_spring_festival for point in after_data)


class TestSeasonalPattern:
    """Test seasonal pattern."""
    
    def test_pattern_creation(self):
        """Test creating seasonal pattern."""
        pattern = SeasonalPattern(
            symbol="000001.SZ",
            pattern_strength=0.75,
            average_return_before=2.5,
            average_return_after=-1.2,
            volatility_before=1.5,
            volatility_after=2.8,
            consistency_score=0.8,
            confidence_level=0.85,
            years_analyzed=[2020, 2021, 2022, 2023, 2024],
            peak_day=-3,
            trough_day=7
        )
        
        assert pattern.symbol == "000001.SZ"
        assert pattern.pattern_strength == 0.75
        assert pattern.is_bullish_before is True
        assert pattern.is_bullish_after is False
        assert pattern.volatility_ratio == pytest.approx(2.8 / 1.5, rel=1e-2)
    
    def test_bullish_bearish_properties(self):
        """Test bullish/bearish properties."""
        # Bullish before, bearish after
        pattern1 = SeasonalPattern(
            symbol="TEST1",
            pattern_strength=0.5,
            average_return_before=3.0,
            average_return_after=-2.0,
            volatility_before=1.0,
            volatility_after=1.0,
            consistency_score=0.5,
            confidence_level=0.5,
            years_analyzed=[2023, 2024],
            peak_day=-5,
            trough_day=5
        )
        
        assert pattern1.is_bullish_before is True
        assert pattern1.is_bullish_after is False
        
        # Bearish before, bullish after
        pattern2 = SeasonalPattern(
            symbol="TEST2",
            pattern_strength=0.5,
            average_return_before=-1.5,
            average_return_after=2.5,
            volatility_before=1.0,
            volatility_after=1.0,
            consistency_score=0.5,
            confidence_level=0.5,
            years_analyzed=[2023, 2024],
            peak_day=3,
            trough_day=-3
        )
        
        assert pattern2.is_bullish_before is False
        assert pattern2.is_bullish_after is True


class TestSpringFestivalAlignmentEngine:
    """Test Spring Festival alignment engine."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data."""
        dates = pd.date_range('2023-01-01', '2024-03-31', freq='D')
        # Filter out weekends (assuming no trading)
        dates = dates[dates.weekday < 5]
        
        np.random.seed(42)  # For reproducible results
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        return pd.DataFrame({
            'stock_code': ['000001.SZ'] * len(dates),
            'trade_date': dates,
            'open_price': prices * 0.99,
            'high_price': prices * 1.02,
            'low_price': prices * 0.98,
            'close_price': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
    
    @pytest.fixture
    def engine(self):
        """Create Spring Festival alignment engine."""
        return SpringFestivalAlignmentEngine(window_days=30)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.window_days == 30
        assert engine.chinese_calendar is not None
        assert engine.min_years > 0
    
    def test_create_spring_festival_windows(self, engine):
        """Test creating Spring Festival windows."""
        years = [2023, 2024]
        windows = engine.create_spring_festival_windows(years)
        
        assert len(windows) == 2
        assert windows[0].year == 2023
        assert windows[1].year == 2024
        assert windows[0].spring_festival_date == date(2023, 1, 22)
        assert windows[1].spring_festival_date == date(2024, 2, 10)
    
    def test_align_to_spring_festival(self, engine, sample_stock_data):
        """Test aligning stock data to Spring Festival."""
        # Use more years to meet minimum requirement
        years = [2022, 2023, 2024]
        
        # Add 2022 data to sample
        dates_2022 = pd.date_range('2022-01-01', '2022-03-31', freq='D')
        dates_2022 = dates_2022[dates_2022.weekday < 5]
        np.random.seed(41)
        prices_2022 = 95 + np.cumsum(np.random.randn(len(dates_2022)) * 0.5)
        
        data_2022 = pd.DataFrame({
            'stock_code': ['000001.SZ'] * len(dates_2022),
            'trade_date': dates_2022,
            'open_price': prices_2022 * 0.99,
            'high_price': prices_2022 * 1.02,
            'low_price': prices_2022 * 0.98,
            'close_price': prices_2022,
            'volume': np.random.randint(1000, 10000, len(dates_2022))
        })
        
        extended_data = pd.concat([data_2022, sample_stock_data], ignore_index=True)
        aligned_data = engine.align_to_spring_festival(extended_data, years)
        
        assert aligned_data.symbol == "000001.SZ"
        assert len(aligned_data.data_points) > 0
        assert aligned_data.years == years
        assert aligned_data.baseline_price > 0
        
        # Check that we have data points from all years
        years_in_data = set(point.year for point in aligned_data.data_points)
        assert 2022 in years_in_data
        assert 2023 in years_in_data
        assert 2024 in years_in_data
    
    def test_align_empty_data(self, engine):
        """Test aligning empty stock data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Stock data is empty"):
            engine.align_to_spring_festival(empty_data)
    
    def test_align_insufficient_years(self, engine, sample_stock_data):
        """Test aligning with insufficient years."""
        # Filter data to only have one year
        single_year_data = sample_stock_data[sample_stock_data['trade_date'].dt.year == 2024]
        
        with patch.object(engine, 'min_years', 3):  # Require 3 years minimum
            with pytest.raises(ValueError, match="Insufficient years for analysis"):
                engine.align_to_spring_festival(single_year_data)
    
    def test_identify_seasonal_patterns(self, engine, sample_stock_data):
        """Test identifying seasonal patterns."""
        # Use more years and add 2022 data
        years = [2022, 2023, 2024]
        
        dates_2022 = pd.date_range('2022-01-01', '2022-03-31', freq='D')
        dates_2022 = dates_2022[dates_2022.weekday < 5]
        np.random.seed(41)
        prices_2022 = 95 + np.cumsum(np.random.randn(len(dates_2022)) * 0.5)
        
        data_2022 = pd.DataFrame({
            'stock_code': ['000001.SZ'] * len(dates_2022),
            'trade_date': dates_2022,
            'open_price': prices_2022 * 0.99,
            'high_price': prices_2022 * 1.02,
            'low_price': prices_2022 * 0.98,
            'close_price': prices_2022,
            'volume': np.random.randint(1000, 10000, len(dates_2022))
        })
        
        extended_data = pd.concat([data_2022, sample_stock_data], ignore_index=True)
        aligned_data = engine.align_to_spring_festival(extended_data, years)
        pattern = engine.identify_seasonal_patterns(aligned_data)
        
        assert pattern.symbol == "000001.SZ"
        assert 0.0 <= pattern.pattern_strength <= 1.0
        assert 0.0 <= pattern.consistency_score <= 1.0
        assert 0.0 <= pattern.confidence_level <= 1.0
        assert pattern.years_analyzed == years
        assert isinstance(pattern.peak_day, int)
        assert isinstance(pattern.trough_day, int)
    
    def test_get_current_position(self, engine):
        """Test getting current position relative to Spring Festival."""
        # Test with known Spring Festival date
        test_date = date(2024, 2, 5)  # 5 days before SF 2024
        position = engine.get_current_position("000001.SZ", test_date)
        
        assert position['symbol'] == "000001.SZ"
        assert position['current_date'] == test_date
        assert position['spring_festival_date'] == date(2024, 2, 10)
        assert position['days_to_spring_festival'] == 5
        assert position['in_analysis_window'] is True
        assert position['position'] == 'pre_festival'
    
    def test_get_current_position_unknown_year(self, engine):
        """Test getting current position for unknown year."""
        test_date = date(1900, 1, 1)  # Year not in calendar
        position = engine.get_current_position("000001.SZ", test_date)
        
        assert position['spring_festival_date'] is None
        assert position['days_to_spring_festival'] is None
        assert position['position'] == 'unknown'
        assert position['in_analysis_window'] is False
    
    def test_generate_trading_signals(self, engine):
        """Test generating trading signals."""
        # Create a bullish pattern
        bullish_pattern = SeasonalPattern(
            symbol="000001.SZ",
            pattern_strength=0.8,
            average_return_before=3.5,
            average_return_after=1.2,
            volatility_before=1.0,
            volatility_after=1.5,
            consistency_score=0.75,
            confidence_level=0.85,
            years_analyzed=[2020, 2021, 2022, 2023, 2024],
            peak_day=-2,
            trough_day=8
        )
        
        # Test pre-festival position
        pre_position = {
            'symbol': '000001.SZ',
            'current_date': date(2024, 2, 5),
            'spring_festival_date': date(2024, 2, 10),
            'days_to_spring_festival': 5,
            'position': 'pre_festival',
            'in_analysis_window': True,
            'relative_day': -5
        }
        
        signals = engine.generate_trading_signals(bullish_pattern, pre_position)
        
        assert signals['signal'] == 'bullish'
        assert signals['strength'] > 0.5
        assert 'buy' in signals['recommended_action'] or 'watch' in signals['recommended_action']
        assert signals['pattern_strength'] == 0.8
        assert signals['confidence_level'] == 0.85
    
    def test_generate_signals_outside_window(self, engine):
        """Test generating signals outside analysis window."""
        pattern = SeasonalPattern(
            symbol="000001.SZ",
            pattern_strength=0.8,
            average_return_before=3.5,
            average_return_after=1.2,
            volatility_before=1.0,
            volatility_after=1.5,
            consistency_score=0.75,
            confidence_level=0.85,
            years_analyzed=[2023, 2024],
            peak_day=-2,
            trough_day=8
        )
        
        outside_position = {
            'symbol': '000001.SZ',
            'current_date': date(2024, 6, 1),
            'spring_festival_date': date(2024, 2, 10),
            'days_to_spring_festival': -111,
            'position': 'normal_period',
            'in_analysis_window': False,
            'relative_day': 111
        }
        
        signals = engine.generate_trading_signals(pattern, outside_position)
        
        assert signals['signal'] == 'neutral'
        assert signals['strength'] == 0.0
        assert signals['recommended_action'] == 'hold'
        assert 'Outside Spring Festival analysis window' in signals['reason']


if __name__ == "__main__":
    pytest.main([__file__])