"""
Advanced Risk Metrics Calculator

This module extends the Enhanced Risk Management Engine with advanced risk metrics
including enhanced Sharpe/Sortino/Calmar ratios, beta calculations, liquidity risk
scoring, and seasonal risk scoring integration with Spring Festival analysis.

Requirements addressed: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class RiskAdjustmentMethod(str, Enum):
    """Risk adjustment methods for metrics calculation"""

    STANDARD = "standard"
    ROBUST = "robust"  # Using median and MAD instead of mean and std
    BOOTSTRAP = "bootstrap"  # Bootstrap confidence intervals


class SeasonalRiskLevel(str, Enum):
    """Seasonal risk levels"""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AdvancedRiskMetrics:
    """Advanced risk metrics with confidence intervals and seasonal adjustments"""

    # Enhanced ratio metrics (required)
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    liquidity_risk_score: float  # 0-100 scale
    liquidity_risk_level: str  # Categorical assessment
    seasonal_risk_score: float  # 0-100 scale
    seasonal_risk_level: SeasonalRiskLevel
    spring_festival_risk_adjustment: float  # Multiplier for current period
    historical_seasonal_volatility: Dict[str, float]  # By month/period

    # Optional metrics with defaults
    sharpe_confidence_interval: Optional[Tuple[float, float]] = None
    sortino_confidence_interval: Optional[Tuple[float, float]] = None
    calmar_confidence_interval: Optional[Tuple[float, float]] = None

    # Market risk metrics
    beta: Optional[float] = None
    beta_confidence_interval: Optional[Tuple[float, float]] = None
    alpha: Optional[float] = None  # Jensen's alpha
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

    # Liquidity risk metrics
    bid_ask_spread_proxy: Optional[float] = None
    market_impact_score: Optional[float] = None

    # Additional advanced metrics
    omega_ratio: Optional[float] = None  # Probability-weighted ratio
    kappa_3: Optional[float] = None  # Third moment (skewness) risk
    tail_ratio: Optional[float] = None  # Tail risk measure

    calculation_date: Optional[datetime] = None


class AdvancedRiskMetricsCalculator:
    """
    Advanced Risk Metrics Calculator with seasonal integration
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 1000,
        seasonal_window_years: int = 5,
    ):
        """
        Initialize the Advanced Risk Metrics Calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            confidence_level: Confidence level for intervals
            bootstrap_iterations: Number of bootstrap iterations
            seasonal_window_years: Years of data for seasonal analysis
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.seasonal_window_years = seasonal_window_years

        # Spring Festival date cache
        self._spring_festival_cache = {}

    async def calculate_advanced_metrics(
        self,
        price_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        volume_data: Optional[pd.DataFrame] = None,
        spring_festival_engine=None,
    ) -> AdvancedRiskMetrics:
        """
        Calculate comprehensive advanced risk metrics.

        Args:
            price_data: Stock price data
            benchmark_data: Benchmark price data for beta calculation
            volume_data: Volume data for liquidity analysis
            spring_festival_engine: Spring Festival analysis engine

        Returns:
            AdvancedRiskMetrics object with all calculated metrics
        """

        try:
            # Calculate returns
            returns = self._calculate_returns(price_data)

            # Calculate enhanced ratio metrics with confidence intervals
            sharpe_ratio, sharpe_ci = await self._calculate_enhanced_sharpe_ratio(
                returns
            )
            sortino_ratio, sortino_ci = await self._calculate_enhanced_sortino_ratio(
                returns
            )
            calmar_ratio, calmar_ci = await self._calculate_enhanced_calmar_ratio(
                price_data, returns
            )

            # Calculate market risk metrics
            beta, beta_ci, alpha, tracking_error, info_ratio = (
                await self._calculate_market_risk_metrics(returns, benchmark_data)
            )

            # Calculate liquidity risk metrics
            liquidity_metrics = await self._calculate_liquidity_risk_metrics(
                price_data, volume_data
            )

            # Calculate seasonal risk metrics
            seasonal_metrics = await self._calculate_seasonal_risk_metrics(
                price_data, returns, spring_festival_engine
            )

            # Calculate additional advanced metrics
            omega_ratio = await self._calculate_omega_ratio(returns)
            kappa_3 = await self._calculate_kappa_3(returns)
            tail_ratio = await self._calculate_tail_ratio(returns)

            return AdvancedRiskMetrics(
                sharpe_ratio=sharpe_ratio,
                sharpe_confidence_interval=sharpe_ci,
                sortino_ratio=sortino_ratio,
                sortino_confidence_interval=sortino_ci,
                calmar_ratio=calmar_ratio,
                calmar_confidence_interval=calmar_ci,
                beta=beta,
                beta_confidence_interval=beta_ci,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=info_ratio,
                liquidity_risk_score=liquidity_metrics["score"],
                liquidity_risk_level=liquidity_metrics["level"],
                bid_ask_spread_proxy=liquidity_metrics["spread_proxy"],
                market_impact_score=liquidity_metrics["impact_score"],
                seasonal_risk_score=seasonal_metrics["score"],
                seasonal_risk_level=seasonal_metrics["level"],
                spring_festival_risk_adjustment=seasonal_metrics["sf_adjustment"],
                historical_seasonal_volatility=seasonal_metrics["seasonal_volatility"],
                omega_ratio=omega_ratio,
                kappa_3=kappa_3,
                tail_ratio=tail_ratio,
                calculation_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating advanced risk metrics: {e}")
            raise

    async def _calculate_enhanced_sharpe_ratio(
        self, returns: pd.Series
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Calculate Sharpe ratio with confidence interval."""

        if len(returns) < 30:
            return 0.0, None

        excess_returns = returns - (self.risk_free_rate / 252)

        if returns.std() == 0:
            return 0.0, None

        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)

        # Bootstrap confidence interval
        sharpe_bootstrap = []
        for _ in range(self.bootstrap_iterations):
            bootstrap_sample = np.random.choice(
                returns, size=len(returns), replace=True
            )
            bootstrap_excess = bootstrap_sample - (self.risk_free_rate / 252)

            if bootstrap_sample.std() > 0:
                bootstrap_sharpe = (
                    bootstrap_excess.mean() / bootstrap_sample.std()
                ) * np.sqrt(252)
                sharpe_bootstrap.append(bootstrap_sharpe)

        if sharpe_bootstrap:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(sharpe_bootstrap, (alpha / 2) * 100)
            ci_upper = np.percentile(sharpe_bootstrap, (1 - alpha / 2) * 100)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None

        return sharpe, confidence_interval

    async def _calculate_enhanced_sortino_ratio(
        self, returns: pd.Series
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Calculate Sortino ratio with confidence interval."""

        if len(returns) < 30:
            return 0.0, None

        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0, None

        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

        # Bootstrap confidence interval
        sortino_bootstrap = []
        for _ in range(self.bootstrap_iterations):
            bootstrap_sample = np.random.choice(
                returns, size=len(returns), replace=True
            )
            bootstrap_excess = bootstrap_sample - (self.risk_free_rate / 252)
            bootstrap_downside = bootstrap_sample[bootstrap_sample < 0]

            if len(bootstrap_downside) > 0 and bootstrap_downside.std() > 0:
                bootstrap_sortino = (
                    bootstrap_excess.mean() / bootstrap_downside.std()
                ) * np.sqrt(252)
                if not np.isinf(bootstrap_sortino) and not np.isnan(bootstrap_sortino):
                    sortino_bootstrap.append(bootstrap_sortino)

        if len(sortino_bootstrap) > 10:  # Need sufficient samples
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(sortino_bootstrap, (alpha / 2) * 100)
            ci_upper = np.percentile(sortino_bootstrap, (1 - alpha / 2) * 100)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None

        return sortino, confidence_interval

    async def _calculate_enhanced_calmar_ratio(
        self, price_data: pd.DataFrame, returns: pd.Series
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Calculate Calmar ratio with confidence interval."""

        if len(returns) < 30:
            return 0.0, None

        # Calculate maximum drawdown
        prices = price_data["close"]
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        if max_drawdown == 0:
            return 0.0, None

        annual_return = returns.mean() * 252
        calmar = annual_return / max_drawdown

        # Bootstrap confidence interval using rolling windows
        calmar_bootstrap = []
        window_size = min(252, len(returns) // 2)  # Use 1 year or half the data

        for _ in range(self.bootstrap_iterations):
            # Random starting point for rolling window
            start_idx = np.random.randint(0, len(returns) - window_size + 1)
            window_returns = returns.iloc[start_idx : start_idx + window_size]
            window_prices = prices.iloc[start_idx : start_idx + window_size]

            # Calculate metrics for this window
            window_running_max = window_prices.expanding().max()
            window_drawdown = (window_prices - window_running_max) / window_running_max
            window_max_dd = abs(window_drawdown.min())

            if window_max_dd > 0:
                window_annual_return = window_returns.mean() * 252
                window_calmar = window_annual_return / window_max_dd

                if not np.isinf(window_calmar) and not np.isnan(window_calmar):
                    calmar_bootstrap.append(window_calmar)

        if len(calmar_bootstrap) > 10:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(calmar_bootstrap, (alpha / 2) * 100)
            ci_upper = np.percentile(calmar_bootstrap, (1 - alpha / 2) * 100)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None

        return calmar, confidence_interval

    async def _calculate_market_risk_metrics(
        self, returns: pd.Series, benchmark_data: Optional[pd.DataFrame]
    ) -> Tuple[
        Optional[float],
        Optional[Tuple[float, float]],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Calculate comprehensive market risk metrics."""

        if benchmark_data is None or len(benchmark_data) < 30:
            return None, None, None, None, None

        try:
            # Calculate benchmark returns
            benchmark_returns = self._calculate_returns(benchmark_data)

            # Align dates
            aligned_data = pd.concat([returns, benchmark_returns], axis=1, join="inner")
            aligned_data.columns = ["stock_returns", "benchmark_returns"]

            if len(aligned_data) < 30:
                return None, None, None, None, None

            stock_rets = aligned_data["stock_returns"]
            bench_rets = aligned_data["benchmark_returns"]

            # Calculate beta using linear regression
            covariance = stock_rets.cov(bench_rets)
            benchmark_variance = bench_rets.var()

            if benchmark_variance == 0:
                return None, None, None, None, None

            beta = covariance / benchmark_variance

            # Calculate alpha (Jensen's alpha)
            stock_mean = stock_rets.mean() * 252  # Annualized
            bench_mean = bench_rets.mean() * 252  # Annualized
            alpha = stock_mean - (
                self.risk_free_rate + beta * (bench_mean - self.risk_free_rate)
            )

            # Calculate tracking error
            active_returns = stock_rets - bench_rets
            tracking_error = active_returns.std() * np.sqrt(252)

            # Calculate information ratio
            information_ratio = (
                (active_returns.mean() * 252) / tracking_error
                if tracking_error > 0
                else 0
            )

            # Bootstrap confidence interval for beta
            beta_bootstrap = []
            for _ in range(self.bootstrap_iterations):
                bootstrap_indices = np.random.choice(
                    len(aligned_data), size=len(aligned_data), replace=True
                )
                bootstrap_stock = stock_rets.iloc[bootstrap_indices]
                bootstrap_bench = bench_rets.iloc[bootstrap_indices]

                bootstrap_cov = bootstrap_stock.cov(bootstrap_bench)
                bootstrap_var = bootstrap_bench.var()

                if bootstrap_var > 0:
                    bootstrap_beta = bootstrap_cov / bootstrap_var
                    beta_bootstrap.append(bootstrap_beta)

            if beta_bootstrap:
                alpha_level = 1 - self.confidence_level
                beta_ci_lower = np.percentile(beta_bootstrap, (alpha_level / 2) * 100)
                beta_ci_upper = np.percentile(
                    beta_bootstrap, (1 - alpha_level / 2) * 100
                )
                beta_confidence_interval = (beta_ci_lower, beta_ci_upper)
            else:
                beta_confidence_interval = None

            return (
                beta,
                beta_confidence_interval,
                alpha,
                tracking_error,
                information_ratio,
            )

        except Exception as e:
            logger.error(f"Error calculating market risk metrics: {e}")
            return None, None, None, None, None

    async def _calculate_liquidity_risk_metrics(
        self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame]
    ) -> Dict[str, Union[float, str]]:
        """Calculate comprehensive liquidity risk metrics."""

        try:
            if volume_data is None or len(volume_data) < 30:
                # Use price-based liquidity proxies only
                return await self._calculate_price_based_liquidity_metrics(price_data)

            # Merge price and volume data
            merged_data = pd.merge(price_data, volume_data, on="date", how="inner")

            if len(merged_data) < 30:
                return await self._calculate_price_based_liquidity_metrics(price_data)

            # Calculate volume-based metrics
            avg_volume = merged_data["volume"].mean()
            volume_volatility = (
                merged_data["volume"].std() / avg_volume if avg_volume > 0 else 1.0
            )

            # Calculate Amihud illiquidity measure
            merged_data["price_change"] = merged_data["close"].pct_change().abs()
            merged_data["dollar_volume"] = merged_data["close"] * merged_data["volume"]

            # Amihud measure: |return| / dollar_volume
            merged_data["amihud"] = merged_data["price_change"] / merged_data[
                "dollar_volume"
            ].replace(0, np.nan)
            amihud_illiquidity = merged_data[
                "amihud"
            ].median()  # Use median to avoid outliers

            # Calculate bid-ask spread proxy using high-low spread
            merged_data["spread_proxy"] = (
                merged_data["high"] - merged_data["low"]
            ) / merged_data["close"]
            avg_spread = merged_data["spread_proxy"].mean()

            # Calculate zero-volume days ratio
            zero_volume_ratio = (merged_data["volume"] == 0).sum() / len(merged_data)

            # Calculate market impact score
            # Higher price volatility relative to volume indicates higher market impact
            merged_data["volume_normalized"] = (
                merged_data["volume"] / merged_data["volume"].rolling(20).mean()
            )
            merged_data["price_volatility"] = merged_data["close"].pct_change().abs()

            # Market impact = price_volatility / volume_normalized
            merged_data["market_impact"] = merged_data[
                "price_volatility"
            ] / merged_data["volume_normalized"].replace(0, np.nan)
            market_impact_score = merged_data["market_impact"].median()

            # Combine metrics into liquidity risk score (0-100 scale)
            # Higher values indicate higher liquidity risk (lower liquidity)

            # Normalize each component
            volume_score = min(100, volume_volatility * 50)
            amihud_score = (
                min(100, amihud_illiquidity * 1e6)
                if not np.isnan(amihud_illiquidity)
                else 50
            )
            spread_score = min(100, avg_spread * 500)
            zero_volume_score = zero_volume_ratio * 100
            impact_score = (
                min(100, market_impact_score * 1000)
                if not np.isnan(market_impact_score)
                else 50
            )

            # Weighted combination
            liquidity_risk_score = (
                volume_score * 0.25
                + amihud_score * 0.25
                + spread_score * 0.20
                + zero_volume_score * 0.15
                + impact_score * 0.15
            )

            # Determine risk level
            if liquidity_risk_score < 20:
                risk_level = "Very High Liquidity"
            elif liquidity_risk_score < 40:
                risk_level = "High Liquidity"
            elif liquidity_risk_score < 60:
                risk_level = "Moderate Liquidity"
            elif liquidity_risk_score < 80:
                risk_level = "Low Liquidity"
            else:
                risk_level = "Very Low Liquidity"

            return {
                "score": min(100, max(0, liquidity_risk_score)),
                "level": risk_level,
                "spread_proxy": avg_spread,
                "impact_score": market_impact_score,
            }

        except Exception as e:
            logger.error(f"Error calculating liquidity risk metrics: {e}")
            return {
                "score": 50.0,
                "level": "Moderate Liquidity",
                "spread_proxy": None,
                "impact_score": None,
            }

    async def _calculate_price_based_liquidity_metrics(
        self, price_data: pd.DataFrame
    ) -> Dict[str, Union[float, str]]:
        """Calculate liquidity metrics using only price data."""

        # Use high-low spread as primary liquidity proxy
        price_data = price_data.copy()
        price_data["spread_proxy"] = (
            price_data["high"] - price_data["low"]
        ) / price_data["close"]
        avg_spread = price_data["spread_proxy"].mean()

        # Use price volatility as secondary proxy
        price_data["returns"] = price_data["close"].pct_change()
        price_volatility = price_data["returns"].std()

        # Simple liquidity risk score based on spread and volatility
        spread_score = min(100, avg_spread * 500)
        volatility_score = min(100, price_volatility * 1000)

        liquidity_risk_score = spread_score * 0.6 + volatility_score * 0.4

        if liquidity_risk_score < 30:
            risk_level = "High Liquidity (Estimated)"
        elif liquidity_risk_score < 60:
            risk_level = "Moderate Liquidity (Estimated)"
        else:
            risk_level = "Low Liquidity (Estimated)"

        return {
            "score": min(100, max(0, liquidity_risk_score)),
            "level": risk_level,
            "spread_proxy": avg_spread,
            "impact_score": None,
        }

    async def _calculate_seasonal_risk_metrics(
        self, price_data: pd.DataFrame, returns: pd.Series, spring_festival_engine
    ) -> Dict[str, Union[float, str, dict]]:
        """Calculate seasonal risk metrics with Spring Festival integration."""

        try:
            # Ensure date column is datetime
            if "date" in price_data.columns:
                price_data = price_data.copy()
                price_data["date"] = pd.to_datetime(price_data["date"])
                price_data.set_index("date", inplace=True)

            # Calculate monthly volatility patterns
            monthly_volatility = {}
            returns_with_date = returns.copy()

            if hasattr(returns_with_date.index, "month"):
                for month in range(1, 13):
                    month_returns = returns_with_date[
                        returns_with_date.index.month == month
                    ]
                    if len(month_returns) > 5:  # Need minimum data
                        monthly_volatility[f"month_{month}"] = (
                            month_returns.std() * np.sqrt(252)
                        )
                    else:
                        monthly_volatility[f"month_{month}"] = returns.std() * np.sqrt(
                            252
                        )

            # Calculate Spring Festival specific risk if engine is available
            spring_festival_adjustment = 1.0
            seasonal_risk_score = 50.0  # Default moderate risk

            if spring_festival_engine is not None:
                try:
                    # Get current date and determine Spring Festival proximity
                    current_date = datetime.now()

                    # Get Spring Festival dates for recent years
                    sf_dates = await self._get_spring_festival_dates(
                        current_date.year - 2, current_date.year + 1
                    )

                    # Calculate proximity to nearest Spring Festival
                    nearest_sf_date = min(
                        sf_dates, key=lambda x: abs((x - current_date).days)
                    )
                    days_to_sf = (nearest_sf_date - current_date).days

                    # Spring Festival risk adjustment based on proximity
                    # Higher risk 30 days before and 15 days after Spring Festival
                    if -15 <= days_to_sf <= 30:
                        # Peak risk period
                        if -5 <= days_to_sf <= 10:
                            spring_festival_adjustment = 1.5  # 50% higher risk
                            seasonal_risk_score = 85.0
                        else:
                            spring_festival_adjustment = 1.2  # 20% higher risk
                            seasonal_risk_score = 70.0
                    elif -45 <= days_to_sf <= 60:
                        # Moderate risk period
                        spring_festival_adjustment = 1.1  # 10% higher risk
                        seasonal_risk_score = 60.0
                    else:
                        # Normal risk period
                        spring_festival_adjustment = 1.0
                        seasonal_risk_score = 45.0

                    # Analyze historical Spring Festival volatility patterns
                    if len(price_data) > 500:  # Need sufficient historical data
                        sf_volatility_analysis = (
                            await self._analyze_spring_festival_volatility(
                                price_data, returns, sf_dates
                            )
                        )

                        # Adjust seasonal risk score based on historical patterns
                        if (
                            sf_volatility_analysis["avg_sf_volatility"]
                            > sf_volatility_analysis["normal_volatility"] * 1.3
                        ):
                            seasonal_risk_score = min(95.0, seasonal_risk_score * 1.2)
                            spring_festival_adjustment = min(
                                2.0, spring_festival_adjustment * 1.1
                            )

                except Exception as e:
                    logger.warning(f"Spring Festival analysis failed: {e}")

            # Determine seasonal risk level
            if seasonal_risk_score < 30:
                risk_level = SeasonalRiskLevel.VERY_LOW
            elif seasonal_risk_score < 45:
                risk_level = SeasonalRiskLevel.LOW
            elif seasonal_risk_score < 65:
                risk_level = SeasonalRiskLevel.MODERATE
            elif seasonal_risk_score < 80:
                risk_level = SeasonalRiskLevel.HIGH
            else:
                risk_level = SeasonalRiskLevel.VERY_HIGH

            return {
                "score": seasonal_risk_score,
                "level": risk_level,
                "sf_adjustment": spring_festival_adjustment,
                "seasonal_volatility": monthly_volatility,
            }

        except Exception as e:
            logger.error(f"Error calculating seasonal risk metrics: {e}")
            return {
                "score": 50.0,
                "level": SeasonalRiskLevel.MODERATE,
                "sf_adjustment": 1.0,
                "seasonal_volatility": {},
            }

    async def _get_spring_festival_dates(
        self, start_year: int, end_year: int
    ) -> List[datetime]:
        """Get Spring Festival dates for specified years."""

        # Simplified Spring Festival date calculation
        # In production, use a proper Chinese calendar library
        spring_festival_dates = []

        # Approximate Spring Festival dates (these should be replaced with actual dates)
        sf_dates_by_year = {
            2020: datetime(2020, 1, 25),
            2021: datetime(2021, 2, 12),
            2022: datetime(2022, 2, 1),
            2023: datetime(2023, 1, 22),
            2024: datetime(2024, 2, 10),
            2025: datetime(2025, 1, 29),
            2026: datetime(2026, 2, 17),
        }

        for year in range(start_year, end_year + 1):
            if year in sf_dates_by_year:
                spring_festival_dates.append(sf_dates_by_year[year])
            else:
                # Fallback: approximate as late January/early February
                spring_festival_dates.append(datetime(year, 2, 1))

        return spring_festival_dates

    async def _analyze_spring_festival_volatility(
        self, price_data: pd.DataFrame, returns: pd.Series, sf_dates: List[datetime]
    ) -> Dict[str, float]:
        """Analyze volatility patterns around Spring Festival dates."""

        try:
            sf_period_returns = []
            normal_period_returns = []

            for sf_date in sf_dates:
                # Define Spring Festival period (30 days before to 15 days after)
                sf_start = sf_date - timedelta(days=30)
                sf_end = sf_date + timedelta(days=15)

                # Get returns in Spring Festival period
                sf_mask = (returns.index >= sf_start) & (returns.index <= sf_end)
                sf_returns = returns[sf_mask]
                sf_period_returns.extend(sf_returns.tolist())

                # Get normal period returns (avoiding Spring Festival periods)
                normal_start1 = sf_date - timedelta(days=120)
                normal_end1 = sf_date - timedelta(days=45)
                normal_start2 = sf_date + timedelta(days=30)
                normal_end2 = sf_date + timedelta(days=75)

                normal_mask1 = (returns.index >= normal_start1) & (
                    returns.index <= normal_end1
                )
                normal_mask2 = (returns.index >= normal_start2) & (
                    returns.index <= normal_end2
                )

                normal_returns1 = returns[normal_mask1]
                normal_returns2 = returns[normal_mask2]

                normal_period_returns.extend(normal_returns1.tolist())
                normal_period_returns.extend(normal_returns2.tolist())

            # Calculate volatilities
            sf_volatility = (
                np.std(sf_period_returns) * np.sqrt(252) if sf_period_returns else 0
            )
            normal_volatility = (
                np.std(normal_period_returns) * np.sqrt(252)
                if normal_period_returns
                else 0
            )

            return {
                "avg_sf_volatility": sf_volatility,
                "normal_volatility": normal_volatility,
                "volatility_ratio": (
                    sf_volatility / normal_volatility if normal_volatility > 0 else 1.0
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing Spring Festival volatility: {e}")
            return {
                "avg_sf_volatility": 0.0,
                "normal_volatility": 0.0,
                "volatility_ratio": 1.0,
            }

    async def _calculate_omega_ratio(
        self, returns: pd.Series, threshold: float = 0.0
    ) -> Optional[float]:
        """Calculate Omega ratio (probability-weighted ratio of gains to losses)."""

        try:
            if len(returns) < 30:
                return None

            gains = returns[returns > threshold]
            losses = returns[returns <= threshold]

            if len(losses) == 0:
                return float("inf")

            if len(gains) == 0:
                return 0.0

            # Calculate probability-weighted gains and losses
            prob_gains = len(gains) / len(returns)
            prob_losses = len(losses) / len(returns)

            avg_gain = gains.mean()
            avg_loss = abs(losses.mean())

            if avg_loss == 0:
                return float("inf")

            omega_ratio = (prob_gains * avg_gain) / (prob_losses * avg_loss)

            return omega_ratio

        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {e}")
            return None

    async def _calculate_kappa_3(self, returns: pd.Series) -> Optional[float]:
        """Calculate Kappa 3 (third moment risk measure)."""

        try:
            if len(returns) < 30:
                return None

            # Calculate skewness-adjusted risk measure
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = stats.skew(returns)

            if std_return == 0:
                return None

            # Kappa 3 incorporates skewness into risk assessment
            # Negative skewness increases risk
            kappa_3 = mean_return / (std_return * (1 - skewness / 6))

            return kappa_3

        except Exception as e:
            logger.error(f"Error calculating Kappa 3: {e}")
            return None

    async def _calculate_tail_ratio(self, returns: pd.Series) -> Optional[float]:
        """Calculate tail ratio (95th percentile / 5th percentile)."""

        try:
            if len(returns) < 30:
                return None

            percentile_95 = np.percentile(returns, 95)
            percentile_5 = np.percentile(returns, 5)

            if percentile_5 == 0:
                return None

            tail_ratio = abs(percentile_95 / percentile_5)

            return tail_ratio

        except Exception as e:
            logger.error(f"Error calculating tail ratio: {e}")
            return None

    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""

        if "close" not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")

        returns = price_data["close"].pct_change().dropna()

        # Set index to date if available
        if "date" in price_data.columns:
            returns.index = pd.to_datetime(price_data["date"].iloc[1:])

        return returns


# Integration function to enhance the main risk management engine
async def calculate_comprehensive_risk_profile(
    price_data: pd.DataFrame,
    benchmark_data: Optional[pd.DataFrame] = None,
    volume_data: Optional[pd.DataFrame] = None,
    spring_festival_engine=None,
    risk_engine=None,
) -> Dict[str, any]:
    """
    Calculate comprehensive risk profile combining basic and advanced metrics.

    Args:
        price_data: Stock price data
        benchmark_data: Benchmark data for market risk metrics
        volume_data: Volume data for liquidity analysis
        spring_festival_engine: Spring Festival analysis engine
        risk_engine: Basic risk management engine

    Returns:
        Dictionary containing all risk metrics
    """

    # Initialize calculators
    if risk_engine is None:
        from .risk_management_engine import EnhancedRiskManagementEngine

        risk_engine = EnhancedRiskManagementEngine()

    advanced_calculator = AdvancedRiskMetricsCalculator()

    # Calculate basic risk metrics
    basic_metrics = await risk_engine.calculate_comprehensive_risk_metrics(
        price_data=price_data, benchmark_data=benchmark_data, volume_data=volume_data
    )

    # Calculate advanced risk metrics
    advanced_metrics = await advanced_calculator.calculate_advanced_metrics(
        price_data=price_data,
        benchmark_data=benchmark_data,
        volume_data=volume_data,
        spring_festival_engine=spring_festival_engine,
    )

    # Combine results
    comprehensive_profile = {
        "basic_metrics": basic_metrics,
        "advanced_metrics": advanced_metrics,
        "summary": {
            "overall_risk_score": _calculate_overall_risk_score(
                basic_metrics, advanced_metrics
            ),
            "risk_level": _determine_overall_risk_level(
                basic_metrics, advanced_metrics
            ),
            "key_risk_factors": _identify_key_risk_factors(
                basic_metrics, advanced_metrics
            ),
            "recommendations": _generate_risk_recommendations(
                basic_metrics, advanced_metrics
            ),
        },
    }

    return comprehensive_profile


def _calculate_overall_risk_score(basic_metrics, advanced_metrics) -> float:
    """Calculate overall risk score (0-100 scale)."""

    try:
        # Weight different risk components
        var_score = (
            basic_metrics.var_results.get(
                "historical", type("obj", (object,), {"var_95": 0.05})
            ).var_95
            * 1000
        )  # Scale VaR
        volatility_score = (
            basic_metrics.volatility_results.get(
                "historical", type("obj", (object,), {"annualized_volatility": 0.3})
            ).annualized_volatility
            * 100
        )
        drawdown_score = basic_metrics.max_drawdown * 100
        liquidity_score = advanced_metrics.liquidity_risk_score
        seasonal_score = advanced_metrics.seasonal_risk_score

        # Weighted combination
        overall_score = (
            var_score * 0.25
            + volatility_score * 0.25
            + drawdown_score * 0.20
            + liquidity_score * 0.15
            + seasonal_score * 0.15
        )

        return min(100, max(0, overall_score))

    except Exception:
        return 50.0  # Default moderate risk


def _determine_overall_risk_level(basic_metrics, advanced_metrics) -> str:
    """Determine overall risk level."""

    overall_score = _calculate_overall_risk_score(basic_metrics, advanced_metrics)

    if overall_score < 20:
        return "Very Low Risk"
    elif overall_score < 40:
        return "Low Risk"
    elif overall_score < 60:
        return "Moderate Risk"
    elif overall_score < 80:
        return "High Risk"
    else:
        return "Very High Risk"


def _identify_key_risk_factors(basic_metrics, advanced_metrics) -> List[str]:
    """Identify key risk factors."""

    risk_factors = []

    try:
        # Check VaR levels
        historical_var = basic_metrics.var_results.get("historical")
        if historical_var and historical_var.var_95 > 0.05:  # 5% daily VaR
            risk_factors.append("High Value at Risk")

        # Check volatility
        historical_vol = basic_metrics.volatility_results.get("historical")
        if (
            historical_vol and historical_vol.annualized_volatility > 0.4
        ):  # 40% annual volatility
            risk_factors.append("High Volatility")

        # Check drawdown
        if basic_metrics.max_drawdown > 0.3:  # 30% max drawdown
            risk_factors.append("Large Historical Drawdowns")

        # Check liquidity
        if advanced_metrics.liquidity_risk_score > 70:
            risk_factors.append("Low Liquidity")

        # Check seasonal risk
        if advanced_metrics.seasonal_risk_score > 70:
            risk_factors.append("High Seasonal Risk")

        # Check Sharpe ratio
        if basic_metrics.sharpe_ratio < 0.5:
            risk_factors.append("Poor Risk-Adjusted Returns")

    except Exception:
        risk_factors.append("Risk Assessment Incomplete")

    return risk_factors if risk_factors else ["Standard Market Risk"]


def _generate_risk_recommendations(basic_metrics, advanced_metrics) -> List[str]:
    """Generate risk management recommendations."""

    recommendations = []

    try:
        # VaR-based recommendations
        historical_var = basic_metrics.var_results.get("historical")
        if historical_var and historical_var.var_95 > 0.03:
            recommendations.append("Consider position size reduction due to high VaR")

        # Volatility-based recommendations
        if basic_metrics.max_drawdown > 0.25:
            recommendations.append("Implement stop-loss orders to limit drawdowns")

        # Liquidity-based recommendations
        if advanced_metrics.liquidity_risk_score > 60:
            recommendations.append(
                "Monitor liquidity conditions and avoid large position sizes"
            )

        # Seasonal recommendations
        if advanced_metrics.seasonal_risk_score > 70:
            recommendations.append("Exercise caution during high seasonal risk periods")

        # Sharpe ratio recommendations
        if basic_metrics.sharpe_ratio < 1.0:
            recommendations.append(
                "Consider diversification to improve risk-adjusted returns"
            )

    except Exception:
        recommendations.append("Conduct regular risk monitoring")

    return (
        recommendations
        if recommendations
        else ["Maintain standard risk management practices"]
    )
