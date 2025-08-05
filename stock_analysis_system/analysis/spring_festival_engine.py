"""Spring Festival Alignment Engine for temporal analysis."""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChineseCalendar:
    """Chinese calendar utilities for Spring Festival date calculation."""

    # Pre-calculated Spring Festival dates (Chinese New Year)
    # These are the actual dates when Spring Festival occurs
    SPRING_FESTIVAL_DATES = {
        2010: date(2010, 2, 14),
        2011: date(2011, 2, 3),
        2012: date(2012, 1, 23),
        2013: date(2013, 2, 10),
        2014: date(2014, 1, 31),
        2015: date(2015, 2, 19),
        2016: date(2016, 2, 8),
        2017: date(2017, 1, 28),
        2018: date(2018, 2, 16),
        2019: date(2019, 2, 5),
        2020: date(2020, 1, 25),
        2021: date(2021, 2, 12),
        2022: date(2022, 2, 1),
        2023: date(2023, 1, 22),
        2024: date(2024, 2, 10),
        2025: date(2025, 1, 29),
        2026: date(2026, 2, 17),
        2027: date(2027, 2, 6),
        2028: date(2028, 1, 26),
        2029: date(2029, 2, 13),
        2030: date(2030, 2, 3),
    }

    @classmethod
    def get_spring_festival(cls, year: int) -> Optional[date]:
        """Get Spring Festival date for a given year."""
        return cls.SPRING_FESTIVAL_DATES.get(year)

    @classmethod
    def get_available_years(cls) -> List[int]:
        """Get list of years with Spring Festival data."""
        return sorted(cls.SPRING_FESTIVAL_DATES.keys())

    @classmethod
    def is_spring_festival_period(
        cls, check_date: date, window_days: int = 15
    ) -> Tuple[bool, Optional[date]]:
        """Check if a date falls within Spring Festival period."""
        year = check_date.year
        sf_date = cls.get_spring_festival(year)

        if sf_date is None:
            return False, None

        # Check if within window
        start_date = sf_date - timedelta(days=window_days)
        end_date = sf_date + timedelta(days=window_days)

        is_in_period = start_date <= check_date <= end_date
        return is_in_period, sf_date

    @classmethod
    def get_trading_days_to_spring_festival(cls, check_date: date) -> Optional[int]:
        """Get number of trading days to Spring Festival (negative if after)."""
        year = check_date.year
        sf_date = cls.get_spring_festival(year)

        if sf_date is None:
            return None

        # Simple approximation: 5 trading days per week
        total_days = (sf_date - check_date).days
        weekends = total_days // 7 * 2
        trading_days = total_days - weekends

        return trading_days


@dataclass
class SpringFestivalWindow:
    """Represents a time window around Spring Festival."""

    year: int
    spring_festival_date: date
    start_date: date
    end_date: date
    window_days: int

    def contains_date(self, check_date: date) -> bool:
        """Check if date falls within this window."""
        return self.start_date <= check_date <= self.end_date

    def days_from_spring_festival(self, check_date: date) -> int:
        """Get number of days from Spring Festival (negative if before)."""
        return (check_date - self.spring_festival_date).days

    def relative_position(self, check_date: date) -> float:
        """Get relative position in window (-1.0 to 1.0)."""
        days_from_sf = self.days_from_spring_festival(check_date)
        return days_from_sf / self.window_days


@dataclass
class AlignedDataPoint:
    """Single data point aligned to Spring Festival."""

    original_date: date
    relative_day: int  # Days from Spring Festival
    spring_festival_date: date
    year: int
    price: float
    volume: Optional[int] = None
    normalized_price: Optional[float] = None

    @property
    def is_before_spring_festival(self) -> bool:
        """Check if this point is before Spring Festival."""
        return self.relative_day < 0

    @property
    def is_after_spring_festival(self) -> bool:
        """Check if this point is after Spring Festival."""
        return self.relative_day > 0


@dataclass
class AlignedTimeSeries:
    """Time series data aligned to Spring Festival."""

    symbol: str
    data_points: List[AlignedDataPoint]
    window_days: int
    years: List[int]
    baseline_price: float

    def __post_init__(self):
        # Sort data points by relative day
        self.data_points.sort(key=lambda x: (x.year, x.relative_day))

    def get_year_data(self, year: int) -> List[AlignedDataPoint]:
        """Get data points for a specific year."""
        return [dp for dp in self.data_points if dp.year == year]

    def get_relative_day_data(self, relative_day: int) -> List[AlignedDataPoint]:
        """Get data points for a specific relative day across all years."""
        return [dp for dp in self.data_points if dp.relative_day == relative_day]

    def get_before_spring_festival(self) -> List[AlignedDataPoint]:
        """Get all data points before Spring Festival."""
        return [dp for dp in self.data_points if dp.is_before_spring_festival]

    def get_after_spring_festival(self) -> List[AlignedDataPoint]:
        """Get all data points after Spring Festival."""
        return [dp for dp in self.data_points if dp.is_after_spring_festival]


@dataclass
class SeasonalPattern:
    """Represents a seasonal pattern around Spring Festival."""

    symbol: str
    pattern_strength: float  # 0.0 to 1.0
    average_return_before: float  # Average return in days before SF
    average_return_after: float  # Average return in days after SF
    volatility_before: float
    volatility_after: float
    consistency_score: float  # How consistent the pattern is across years
    confidence_level: float  # Statistical confidence in the pattern
    years_analyzed: List[int]
    peak_day: int  # Relative day with highest average return
    trough_day: int  # Relative day with lowest average return

    @property
    def is_bullish_before(self) -> bool:
        """Check if pattern is bullish before Spring Festival."""
        return self.average_return_before > 0

    @property
    def is_bullish_after(self) -> bool:
        """Check if pattern is bullish after Spring Festival."""
        return self.average_return_after > 0

    @property
    def volatility_ratio(self) -> float:
        """Ratio of volatility after vs before Spring Festival."""
        if self.volatility_before == 0:
            return float("inf") if self.volatility_after > 0 else 1.0
        return self.volatility_after / self.volatility_before


class SpringFestivalAlignmentEngine:
    """Core engine for Spring Festival temporal analysis."""

    def __init__(self, window_days: int = None):
        self.window_days = window_days or settings.spring_festival_window_days
        self.chinese_calendar = ChineseCalendar()
        self.min_years = settings.min_years_for_analysis

    def create_spring_festival_windows(
        self, years: List[int]
    ) -> List[SpringFestivalWindow]:
        """Create Spring Festival windows for given years."""
        windows = []

        for year in years:
            sf_date = self.chinese_calendar.get_spring_festival(year)
            if sf_date is None:
                logger.warning(f"No Spring Festival date available for year {year}")
                continue

            start_date = sf_date - timedelta(days=self.window_days)
            end_date = sf_date + timedelta(days=self.window_days)

            window = SpringFestivalWindow(
                year=year,
                spring_festival_date=sf_date,
                start_date=start_date,
                end_date=end_date,
                window_days=self.window_days,
            )
            windows.append(window)

        return windows

    def align_to_spring_festival(
        self, stock_data: pd.DataFrame, years: List[int] = None
    ) -> AlignedTimeSeries:
        """Align stock data to Spring Festival dates."""
        logger.info(
            f"Aligning stock data to Spring Festival for symbol: {stock_data['stock_code'].iloc[0] if not stock_data.empty else 'unknown'}"
        )

        if stock_data.empty:
            raise ValueError("Stock data is empty")

        # Get symbol
        symbol = stock_data["stock_code"].iloc[0]

        # Determine years to analyze
        if years is None:
            data_years = stock_data["trade_date"].dt.year.unique()
            available_years = self.chinese_calendar.get_available_years()
            years = sorted(set(data_years) & set(available_years))

        if len(years) < self.min_years:
            raise ValueError(
                f"Insufficient years for analysis. Need at least {self.min_years}, got {len(years)}"
            )

        # Create windows
        windows = self.create_spring_festival_windows(years)

        # Calculate baseline price (average of all closing prices)
        baseline_price = stock_data["close_price"].mean()

        # Align data points
        aligned_points = []

        for window in windows:
            # Filter data for this window
            window_data = stock_data[
                (stock_data["trade_date"].dt.date >= window.start_date)
                & (stock_data["trade_date"].dt.date <= window.end_date)
            ].copy()

            if window_data.empty:
                logger.warning(
                    f"No data found for {symbol} in {window.year} Spring Festival window"
                )
                continue

            # Create aligned data points
            for _, row in window_data.iterrows():
                trade_date = (
                    row["trade_date"].date()
                    if hasattr(row["trade_date"], "date")
                    else row["trade_date"]
                )
                relative_day = (trade_date - window.spring_festival_date).days

                # Normalize price relative to baseline
                normalized_price = (
                    (row["close_price"] - baseline_price) / baseline_price * 100
                )

                aligned_point = AlignedDataPoint(
                    original_date=trade_date,
                    relative_day=relative_day,
                    spring_festival_date=window.spring_festival_date,
                    year=window.year,
                    price=row["close_price"],
                    volume=row.get("volume"),
                    normalized_price=normalized_price,
                )
                aligned_points.append(aligned_point)

        aligned_series = AlignedTimeSeries(
            symbol=symbol,
            data_points=aligned_points,
            window_days=self.window_days,
            years=years,
            baseline_price=baseline_price,
        )

        logger.info(
            f"Aligned {len(aligned_points)} data points across {len(years)} years"
        )
        return aligned_series

    def identify_seasonal_patterns(
        self, aligned_data: AlignedTimeSeries
    ) -> SeasonalPattern:
        """Identify seasonal patterns in aligned data."""
        logger.info(f"Identifying seasonal patterns for {aligned_data.symbol}")

        if not aligned_data.data_points:
            raise ValueError("No aligned data points available")

        # Group data by relative day
        daily_returns = {}
        daily_prices = {}

        for point in aligned_data.data_points:
            if point.relative_day not in daily_returns:
                daily_returns[point.relative_day] = []
                daily_prices[point.relative_day] = []

            daily_returns[point.relative_day].append(point.normalized_price)
            daily_prices[point.relative_day].append(point.price)

        # Calculate statistics for each relative day
        daily_stats = {}
        for rel_day, returns in daily_returns.items():
            if len(returns) >= 2:  # Need at least 2 data points
                daily_stats[rel_day] = {
                    "mean_return": np.mean(returns),
                    "std_return": np.std(returns),
                    "count": len(returns),
                }

        if not daily_stats:
            raise ValueError("Insufficient data for pattern analysis")

        # Separate before and after Spring Festival
        before_sf_stats = {day: stats for day, stats in daily_stats.items() if day < 0}
        after_sf_stats = {day: stats for day, stats in daily_stats.items() if day > 0}

        # Calculate aggregate statistics
        if before_sf_stats:
            before_returns = [
                stats["mean_return"] for stats in before_sf_stats.values()
            ]
            avg_return_before = np.mean(before_returns)
            volatility_before = np.std(before_returns)
        else:
            avg_return_before = 0.0
            volatility_before = 0.0

        if after_sf_stats:
            after_returns = [stats["mean_return"] for stats in after_sf_stats.values()]
            avg_return_after = np.mean(after_returns)
            volatility_after = np.std(after_returns)
        else:
            avg_return_after = 0.0
            volatility_after = 0.0

        # Find peak and trough days
        all_returns = [
            (day, stats["mean_return"]) for day, stats in daily_stats.items()
        ]
        peak_day = max(all_returns, key=lambda x: x[1])[0]
        trough_day = min(all_returns, key=lambda x: x[1])[0]

        # Calculate pattern strength (based on consistency across years)
        pattern_strength = self._calculate_pattern_strength(aligned_data)

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(daily_stats)

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(aligned_data, daily_stats)

        pattern = SeasonalPattern(
            symbol=aligned_data.symbol,
            pattern_strength=pattern_strength,
            average_return_before=avg_return_before,
            average_return_after=avg_return_after,
            volatility_before=volatility_before,
            volatility_after=volatility_after,
            consistency_score=consistency_score,
            confidence_level=confidence_level,
            years_analyzed=aligned_data.years,
            peak_day=peak_day,
            trough_day=trough_day,
        )

        logger.info(f"Pattern analysis completed for {aligned_data.symbol}:")
        logger.info(f"  Pattern strength: {pattern_strength:.2f}")
        logger.info(f"  Avg return before SF: {avg_return_before:.2f}%")
        logger.info(f"  Avg return after SF: {avg_return_after:.2f}%")

        return pattern

    def _calculate_pattern_strength(self, aligned_data: AlignedTimeSeries) -> float:
        """Calculate pattern strength based on consistency across years."""
        if len(aligned_data.years) < 2:
            return 0.0

        # Calculate year-over-year correlation of returns
        year_returns = {}

        for year in aligned_data.years:
            year_data = aligned_data.get_year_data(year)
            if len(year_data) < 10:  # Need sufficient data points
                continue

            # Calculate daily returns for this year
            year_data.sort(key=lambda x: x.relative_day)
            returns = []
            for i in range(1, len(year_data)):
                if year_data[i].price > 0 and year_data[i - 1].price > 0:
                    ret = (year_data[i].price - year_data[i - 1].price) / year_data[
                        i - 1
                    ].price
                    returns.append(ret)

            if len(returns) >= 5:  # Need minimum returns
                year_returns[year] = returns

        if len(year_returns) < 2:
            return 0.0

        # Calculate average correlation between years
        correlations = []
        years_list = list(year_returns.keys())

        for i in range(len(years_list)):
            for j in range(i + 1, len(years_list)):
                year1_returns = year_returns[years_list[i]]
                year2_returns = year_returns[years_list[j]]

                # Align returns by taking minimum length
                min_len = min(len(year1_returns), len(year2_returns))
                if min_len >= 5:
                    corr = np.corrcoef(
                        year1_returns[:min_len], year2_returns[:min_len]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute correlation

        if not correlations:
            return 0.0

        return np.mean(correlations)

    def _calculate_consistency_score(
        self, daily_stats: Dict[int, Dict[str, float]]
    ) -> float:
        """Calculate consistency score based on standard deviation of returns."""
        if not daily_stats:
            return 0.0

        # Calculate coefficient of variation for each day
        cv_scores = []
        for day, stats in daily_stats.items():
            if stats["mean_return"] != 0 and stats["count"] >= 2:
                cv = abs(stats["std_return"] / stats["mean_return"])
                cv_scores.append(1.0 / (1.0 + cv))  # Convert to consistency score

        if not cv_scores:
            return 0.0

        return np.mean(cv_scores)

    def _calculate_confidence_level(
        self, aligned_data: AlignedTimeSeries, daily_stats: Dict[int, Dict[str, float]]
    ) -> float:
        """Calculate statistical confidence level."""
        # Base confidence on number of years and data points
        years_factor = min(len(aligned_data.years) / 10.0, 1.0)  # Max at 10 years

        # Data density factor
        total_possible_points = len(aligned_data.years) * (self.window_days * 2)
        actual_points = len(aligned_data.data_points)
        density_factor = min(actual_points / total_possible_points, 1.0)

        # Statistical significance factor
        significant_days = sum(
            1 for stats in daily_stats.values() if stats["count"] >= 3
        )
        total_days = len(daily_stats)
        significance_factor = significant_days / total_days if total_days > 0 else 0.0

        # Combined confidence
        confidence = (
            years_factor * 0.4 + density_factor * 0.3 + significance_factor * 0.3
        )

        return min(confidence, 1.0)

    def get_current_position(
        self, symbol: str, current_date: date = None
    ) -> Dict[str, Any]:
        """Get current position relative to Spring Festival cycle."""
        if current_date is None:
            current_date = date.today()

        year = current_date.year
        sf_date = self.chinese_calendar.get_spring_festival(year)

        if sf_date is None:
            return {
                "symbol": symbol,
                "current_date": current_date,
                "spring_festival_date": None,
                "days_to_spring_festival": None,
                "position": "unknown",
                "in_analysis_window": False,
            }

        days_to_sf = (sf_date - current_date).days

        # Determine position
        if abs(days_to_sf) <= self.window_days:
            in_window = True
            if days_to_sf > 15:
                position = "approaching"
            elif days_to_sf > 0:
                position = "pre_festival"
            elif days_to_sf == 0:
                position = "festival_day"
            elif days_to_sf > -15:
                position = "post_festival"
            else:
                position = "recovery"
        else:
            in_window = False
            position = "normal_period"

        return {
            "symbol": symbol,
            "current_date": current_date,
            "spring_festival_date": sf_date,
            "days_to_spring_festival": days_to_sf,
            "position": position,
            "in_analysis_window": in_window,
            "relative_day": -days_to_sf,  # Negative days_to_sf becomes positive relative_day
        }

    def generate_trading_signals(
        self, pattern: SeasonalPattern, current_position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on seasonal patterns."""
        if not current_position["in_analysis_window"]:
            return {
                "signal": "neutral",
                "strength": 0.0,
                "reason": "Outside Spring Festival analysis window",
                "recommended_action": "hold",
            }

        relative_day = current_position["relative_day"]
        position = current_position["position"]

        # Base signal on pattern and current position
        signal_strength = 0.0
        signal_type = "neutral"
        reason = ""
        action = "hold"

        # Pre-festival signals
        if position in ["approaching", "pre_festival"]:
            if pattern.is_bullish_before and pattern.pattern_strength > 0.5:
                signal_type = "bullish"
                signal_strength = pattern.pattern_strength * pattern.confidence_level
                reason = f"Historical pattern shows {pattern.average_return_before:.1f}% avg return before SF"
                action = "buy" if signal_strength > 0.6 else "watch"
            elif pattern.average_return_before < -2.0:
                signal_type = "bearish"
                signal_strength = pattern.pattern_strength * pattern.confidence_level
                reason = f"Historical pattern shows {pattern.average_return_before:.1f}% avg decline before SF"
                action = "sell" if signal_strength > 0.6 else "reduce"

        # Post-festival signals
        elif position in ["post_festival", "recovery"]:
            if pattern.is_bullish_after and pattern.pattern_strength > 0.5:
                signal_type = "bullish"
                signal_strength = pattern.pattern_strength * pattern.confidence_level
                reason = f"Historical pattern shows {pattern.average_return_after:.1f}% avg return after SF"
                action = "buy" if signal_strength > 0.6 else "watch"
            elif pattern.average_return_after < -2.0:
                signal_type = "bearish"
                signal_strength = pattern.pattern_strength * pattern.confidence_level
                reason = f"Historical pattern shows {pattern.average_return_after:.1f}% avg decline after SF"
                action = "sell" if signal_strength > 0.6 else "reduce"

        # Festival day
        elif position == "festival_day":
            signal_type = "neutral"
            reason = "Spring Festival day - markets typically closed"
            action = "hold"

        # Adjust for volatility
        if pattern.volatility_ratio > 2.0:
            signal_strength *= 0.8  # Reduce strength for high volatility
            reason += f" (High volatility: {pattern.volatility_ratio:.1f}x)"

        return {
            "signal": signal_type,
            "strength": signal_strength,
            "reason": reason,
            "recommended_action": action,
            "pattern_strength": pattern.pattern_strength,
            "confidence_level": pattern.confidence_level,
            "volatility_warning": pattern.volatility_ratio > 2.0,
        }
