#!/usr/bin/env python3
"""Demo script for Spring Festival Analysis Engine."""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from stock_analysis_system.analysis.spring_festival_engine import (
    ChineseCalendar,
    SpringFestivalAlignmentEngine
)

def create_sample_stock_data():
    """Create sample stock data for demonstration."""
    print("📊 Creating sample stock data...")
    
    # Create 5 years of sample data (2020-2024)
    all_data = []
    
    for year in range(2020, 2025):
        # Create daily data for the year
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        # Filter out weekends (assuming no trading)
        trading_dates = dates[dates.weekday < 5]
        
        # Generate realistic price movements with some Spring Festival effects
        np.random.seed(year)  # Different seed for each year
        base_price = 100 + (year - 2020) * 5  # Slight upward trend over years
        
        # Create price series with some volatility
        returns = np.random.normal(0.001, 0.02, len(trading_dates))  # Daily returns
        
        # Add Spring Festival effect
        sf_date = ChineseCalendar.get_spring_festival(year)
        if sf_date:
            for i, trade_date in enumerate(trading_dates):
                days_to_sf = (sf_date - trade_date.date()).days
                
                # Add seasonal pattern: slight increase before SF, decrease after
                if -30 <= days_to_sf <= -1:  # Before Spring Festival
                    returns[i] += 0.002  # Small positive bias
                elif 1 <= days_to_sf <= 15:  # After Spring Festival
                    returns[i] -= 0.001  # Small negative bias
        
        # Calculate cumulative prices
        prices = base_price * np.cumprod(1 + returns)
        
        # Create DataFrame for this year
        year_data = pd.DataFrame({
            'stock_code': ['000001.SZ'] * len(trading_dates),
            'trade_date': trading_dates,
            'open_price': prices * 0.999,
            'high_price': prices * 1.015,
            'low_price': prices * 0.985,
            'close_price': prices,
            'volume': np.random.randint(1000000, 10000000, len(trading_dates)),
            'amount': prices * np.random.randint(1000000, 10000000, len(trading_dates))
        })
        
        all_data.append(year_data)
    
    # Combine all years
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"   ✓ Created {len(combined_data)} trading days of data")
    print(f"   ✓ Date range: {combined_data['trade_date'].min().date()} to {combined_data['trade_date'].max().date()}")
    print(f"   ✓ Price range: {combined_data['close_price'].min():.2f} to {combined_data['close_price'].max():.2f}")
    
    return combined_data

def demonstrate_chinese_calendar():
    """Demonstrate Chinese calendar functionality."""
    print("\n🏮 Chinese Calendar Demonstration")
    print("-" * 50)
    
    # Show Spring Festival dates
    print("Spring Festival dates:")
    available_years = ChineseCalendar.get_available_years()
    for year in available_years[-10:]:  # Show last 10 years
        sf_date = ChineseCalendar.get_spring_festival(year)
        weekday = sf_date.strftime('%A')
        print(f"   {year}: {sf_date} ({weekday})")
    
    # Test current position
    today = date.today()
    current_year = today.year
    sf_date = ChineseCalendar.get_spring_festival(current_year)
    
    if sf_date:
        days_to_sf = (sf_date - today).days
        is_in_period, _ = ChineseCalendar.is_spring_festival_period(today, window_days=30)
        
        print(f"\nCurrent date analysis:")
        print(f"   Today: {today}")
        print(f"   {current_year} Spring Festival: {sf_date}")
        print(f"   Days to Spring Festival: {days_to_sf}")
        print(f"   In Spring Festival period (±30 days): {is_in_period}")
    
    return available_years

def demonstrate_spring_festival_alignment(stock_data):
    """Demonstrate Spring Festival alignment functionality."""
    print("\n🎯 Spring Festival Alignment Demonstration")
    print("-" * 50)
    
    # Initialize engine
    engine = SpringFestivalAlignmentEngine(window_days=60)
    print(f"   Analysis window: ±{engine.window_days} days around Spring Festival")
    
    # Align data to Spring Festival
    print("\n   Aligning stock data to Spring Festival dates...")
    years = [2020, 2021, 2022, 2023, 2024]
    aligned_data = engine.align_to_spring_festival(stock_data, years)
    
    print(f"   ✓ Aligned {len(aligned_data.data_points)} data points")
    print(f"   ✓ Years analyzed: {aligned_data.years}")
    print(f"   ✓ Baseline price: {aligned_data.baseline_price:.2f}")
    
    # Show some statistics
    before_sf_data = aligned_data.get_before_spring_festival()
    after_sf_data = aligned_data.get_after_spring_festival()
    
    print(f"\n   Data distribution:")
    print(f"   Before Spring Festival: {len(before_sf_data)} points")
    print(f"   After Spring Festival: {len(after_sf_data)} points")
    
    # Show data for each year
    print(f"\n   Data points by year:")
    for year in aligned_data.years:
        year_data = aligned_data.get_year_data(year)
        if year_data:
            min_day = min(point.relative_day for point in year_data)
            max_day = max(point.relative_day for point in year_data)
            print(f"   {year}: {len(year_data)} points (days {min_day} to {max_day})")
    
    return aligned_data

def demonstrate_pattern_analysis(aligned_data):
    """Demonstrate seasonal pattern analysis."""
    print("\n📈 Seasonal Pattern Analysis Demonstration")
    print("-" * 50)
    
    # Initialize engine
    engine = SpringFestivalAlignmentEngine()
    
    # Identify patterns
    print("   Analyzing seasonal patterns...")
    pattern = engine.identify_seasonal_patterns(aligned_data)
    
    print(f"\n   📊 PATTERN ANALYSIS RESULTS")
    print(f"   Symbol: {pattern.symbol}")
    print(f"   Pattern Strength: {pattern.pattern_strength:.3f}")
    print(f"   Consistency Score: {pattern.consistency_score:.3f}")
    print(f"   Confidence Level: {pattern.confidence_level:.3f}")
    print(f"   Years Analyzed: {len(pattern.years_analyzed)}")
    
    print(f"\n   📈 RETURN ANALYSIS")
    print(f"   Average Return Before SF: {pattern.average_return_before:+.2f}%")
    print(f"   Average Return After SF: {pattern.average_return_after:+.2f}%")
    print(f"   Bullish Before SF: {'Yes' if pattern.is_bullish_before else 'No'}")
    print(f"   Bullish After SF: {'Yes' if pattern.is_bullish_after else 'No'}")
    
    print(f"\n   📊 VOLATILITY ANALYSIS")
    print(f"   Volatility Before SF: {pattern.volatility_before:.2f}%")
    print(f"   Volatility After SF: {pattern.volatility_after:.2f}%")
    print(f"   Volatility Ratio (After/Before): {pattern.volatility_ratio:.2f}x")
    
    print(f"\n   🎯 KEY DATES")
    print(f"   Peak Day (relative to SF): {pattern.peak_day:+d}")
    print(f"   Trough Day (relative to SF): {pattern.trough_day:+d}")
    
    # Interpret the pattern
    print(f"\n   💡 PATTERN INTERPRETATION")
    if pattern.pattern_strength > 0.6:
        strength_desc = "Strong"
    elif pattern.pattern_strength > 0.4:
        strength_desc = "Moderate"
    else:
        strength_desc = "Weak"
    
    print(f"   Pattern Strength: {strength_desc}")
    
    if pattern.confidence_level > 0.7:
        confidence_desc = "High confidence"
    elif pattern.confidence_level > 0.5:
        confidence_desc = "Medium confidence"
    else:
        confidence_desc = "Low confidence"
    
    print(f"   Statistical Confidence: {confidence_desc}")
    
    # Trading implications
    if pattern.is_bullish_before and pattern.pattern_strength > 0.5:
        print(f"   💰 Historically tends to rise before Spring Festival")
    elif not pattern.is_bullish_before and pattern.pattern_strength > 0.5:
        print(f"   📉 Historically tends to decline before Spring Festival")
    
    if pattern.volatility_ratio > 1.5:
        print(f"   ⚠️  Higher volatility after Spring Festival")
    
    return pattern

def demonstrate_current_position_and_signals(pattern):
    """Demonstrate current position analysis and trading signals."""
    print("\n🎯 Current Position & Trading Signals Demonstration")
    print("-" * 50)
    
    engine = SpringFestivalAlignmentEngine()
    
    # Test different dates throughout the year
    test_dates = [
        (date(2024, 1, 15), "Pre-Spring Festival"),
        (date(2024, 2, 10), "Spring Festival Day"),
        (date(2024, 2, 25), "Post-Spring Festival"),
        (date(2024, 6, 15), "Mid-year (Normal Period)"),
        (date.today(), "Today")
    ]
    
    print("   Testing different time periods:")
    
    for test_date, description in test_dates:
        print(f"\n   📅 {description} ({test_date})")
        
        # Get current position
        position = engine.get_current_position("000001.SZ", test_date)
        
        print(f"      Position: {position['position']}")
        print(f"      Days to SF: {position['days_to_spring_festival']}")
        print(f"      In Analysis Window: {position['in_analysis_window']}")
        
        if position['in_analysis_window']:
            # Generate trading signals
            signals = engine.generate_trading_signals(pattern, position)
            
            print(f"      🚦 TRADING SIGNAL")
            print(f"         Signal: {signals['signal'].upper()}")
            print(f"         Strength: {signals['strength']:.2f}")
            print(f"         Recommended Action: {signals['recommended_action'].upper()}")
            print(f"         Reason: {signals['reason']}")
            
            if signals.get('volatility_warning'):
                print(f"         ⚠️  High volatility warning")
        else:
            print(f"      🔄 Outside analysis window - Normal trading period")

def demonstrate_historical_performance():
    """Demonstrate historical performance analysis."""
    print("\n📚 Historical Performance Analysis")
    print("-" * 50)
    
    # Show Spring Festival dates and market context
    print("   Spring Festival dates and market context:")
    
    years = [2020, 2021, 2022, 2023, 2024]
    for year in years:
        sf_date = ChineseCalendar.get_spring_festival(year)
        if sf_date:
            weekday = sf_date.strftime('%A')
            
            # Add some market context (this would come from real data in practice)
            if year == 2020:
                context = "COVID-19 pandemic impact"
            elif year == 2021:
                context = "Economic recovery"
            elif year == 2022:
                context = "Market volatility"
            elif year == 2023:
                context = "Post-pandemic normalization"
            else:
                context = "Current year"
            
            print(f"   {year}: {sf_date} ({weekday}) - {context}")

def main():
    """Main demonstration function."""
    print("🚀 Spring Festival Analysis Engine Demonstration")
    print("=" * 70)
    print("This demo shows how the Spring Festival Analysis Engine works")
    print("to identify seasonal patterns in Chinese stock markets.")
    print()
    
    try:
        # Create sample data
        stock_data = create_sample_stock_data()
        
        # Demonstrate Chinese calendar
        available_years = demonstrate_chinese_calendar()
        
        # Demonstrate Spring Festival alignment
        aligned_data = demonstrate_spring_festival_alignment(stock_data)
        
        # Demonstrate pattern analysis
        pattern = demonstrate_pattern_analysis(aligned_data)
        
        # Demonstrate current position and signals
        demonstrate_current_position_and_signals(pattern)
        
        # Demonstrate historical performance
        demonstrate_historical_performance()
        
        print(f"\n🎉 Spring Festival Analysis Engine demonstration completed!")
        print("=" * 70)
        
        print(f"✅ Key Features Demonstrated:")
        print(f"   ✓ Chinese calendar integration with Spring Festival dates")
        print(f"   ✓ Temporal data alignment around Spring Festival")
        print(f"   ✓ Seasonal pattern identification and scoring")
        print(f"   ✓ Statistical confidence and consistency analysis")
        print(f"   ✓ Current position analysis relative to Spring Festival cycle")
        print(f"   ✓ Trading signal generation based on historical patterns")
        print(f"   ✓ Risk assessment with volatility analysis")
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Pattern Strength: {pattern.pattern_strength:.2f}")
        print(f"   Confidence Level: {pattern.confidence_level:.2f}")
        print(f"   Years Analyzed: {len(pattern.years_analyzed)}")
        print(f"   Data Points: {len(aligned_data.data_points)}")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()