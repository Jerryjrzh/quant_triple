"""
Enhanced Risk Management Engine

This module implements comprehensive Value at Risk (VaR) calculations and risk metrics
for the stock analysis system. It provides multiple VaR calculation methods including
historical, parametric, and Monte Carlo approaches, along with Conditional VaR (CVaR)
for tail risk assessment.

Requirements addressed: 4.1, 4.2, 4.3, 4.4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class VaRMethod(str, Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"

class VolatilityMethod(str, Enum):
    """Volatility calculation methods"""
    HISTORICAL = "historical"
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    GARCH = "garch"  # GARCH(1,1) model

@dataclass
class VaRResult:
    """VaR calculation result"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% Conditional VaR (Expected Shortfall)
    cvar_99: float  # 99% Conditional VaR
    method: VaRMethod
    confidence_interval: Optional[Tuple[float, float]] = None
    calculation_date: datetime = None

@dataclass
class VolatilityResult:
    """Volatility calculation result"""
    daily_volatility: float
    annualized_volatility: float
    method: VolatilityMethod
    window_size: int
    realized_volatility: Optional[float] = None

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_results: Dict[str, VaRResult]  # VaR results by method
    volatility_results: Dict[str, VolatilityResult]  # Volatility by method
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: Optional[float] = None
    liquidity_risk_score: Optional[float] = None
    seasonal_risk_score: Optional[float] = None

class EnhancedRiskManagementEngine:
    """
    Enhanced Risk Management Engine with comprehensive VaR calculations
    and risk metrics computation.
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 var_window: int = 252,  # 1 year of trading days
                 volatility_window: int = 30,
                 monte_carlo_simulations: int = 10000):
        """
        Initialize the Enhanced Risk Management Engine.
        
        Args:
            confidence_levels: List of confidence levels for VaR calculation
            var_window: Window size for VaR calculations (trading days)
            volatility_window: Window size for volatility calculations
            monte_carlo_simulations: Number of simulations for Monte Carlo VaR
        """
        self.confidence_levels = confidence_levels
        self.var_window = var_window
        self.volatility_window = volatility_window
        self.monte_carlo_simulations = monte_carlo_simulations
        
        # Risk-free rate (can be updated dynamically)
        self.risk_free_rate = 0.03  # 3% annual
        
        # EWMA lambda parameter for volatility calculation
        self.ewma_lambda = 0.94
        
    async def calculate_comprehensive_risk_metrics(self, 
                                                 price_data: pd.DataFrame,
                                                 benchmark_data: Optional[pd.DataFrame] = None,
                                                 volume_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a stock.
        
        Args:
            price_data: DataFrame with columns ['date', 'close', 'high', 'low', 'open']
            benchmark_data: Optional benchmark data for beta calculation
            volume_data: Optional volume data for liquidity risk assessment
            
        Returns:
            RiskMetrics object with all calculated risk measures
        """
        try:
            # Validate input data
            self._validate_price_data(price_data)
            
            # Calculate returns
            returns = self._calculate_returns(price_data)
            
            # Calculate VaR using multiple methods
            var_results = {}
            for method in VaRMethod:
                try:
                    var_result = await self._calculate_var(returns, method)
                    var_results[method.value] = var_result
                except Exception as e:
                    logger.warning(f"Failed to calculate VaR using {method.value}: {e}")
            
            # Calculate volatility using multiple methods
            volatility_results = {}
            for method in VolatilityMethod:
                try:
                    vol_result = await self._calculate_volatility(returns, method)
                    volatility_results[method.value] = vol_result
                except Exception as e:
                    logger.warning(f"Failed to calculate volatility using {method.value}: {e}")
            
            # Calculate additional risk metrics
            max_drawdown = self._calculate_max_drawdown(price_data)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Calculate beta if benchmark data is provided
            beta = None
            if benchmark_data is not None:
                beta = await self._calculate_beta(returns, benchmark_data)
            
            # Calculate liquidity risk score if volume data is provided
            liquidity_risk_score = None
            if volume_data is not None:
                liquidity_risk_score = await self._calculate_liquidity_risk_score(
                    price_data, volume_data
                )
            
            return RiskMetrics(
                var_results=var_results,
                volatility_results=volatility_results,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                liquidity_risk_score=liquidity_risk_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive risk metrics: {e}")
            raise
    
    async def _calculate_var(self, returns: pd.Series, method: VaRMethod) -> VaRResult:
        """Calculate VaR using the specified method."""
        
        if method == VaRMethod.HISTORICAL:
            return await self._calculate_historical_var(returns)
        elif method == VaRMethod.PARAMETRIC:
            return await self._calculate_parametric_var(returns)
        elif method == VaRMethod.MONTE_CARLO:
            return await self._calculate_monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    async def _calculate_historical_var(self, returns: pd.Series) -> VaRResult:
        """Calculate Historical VaR."""
        
        # Use the most recent var_window returns
        recent_returns = returns.tail(self.var_window)
        
        if len(recent_returns) < 30:  # Minimum data requirement
            raise ValueError("Insufficient data for Historical VaR calculation")
        
        # Calculate VaR at different confidence levels
        var_95 = np.percentile(recent_returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(recent_returns, 1)  # 1st percentile for 99% VaR
        
        # Calculate Conditional VaR (Expected Shortfall)
        cvar_95 = recent_returns[recent_returns <= var_95].mean()
        cvar_99 = recent_returns[recent_returns <= var_99].mean()
        
        return VaRResult(
            var_95=abs(var_95),  # Convert to positive value
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            cvar_99=abs(cvar_99),
            method=VaRMethod.HISTORICAL,
            calculation_date=datetime.now()
        )
    
    async def _calculate_parametric_var(self, returns: pd.Series) -> VaRResult:
        """Calculate Parametric VaR assuming normal distribution."""
        
        recent_returns = returns.tail(self.var_window)
        
        if len(recent_returns) < 30:
            raise ValueError("Insufficient data for Parametric VaR calculation")
        
        # Calculate mean and standard deviation
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        
        # Calculate VaR using normal distribution
        var_95 = abs(mean_return + stats.norm.ppf(0.05) * std_return)
        var_99 = abs(mean_return + stats.norm.ppf(0.01) * std_return)
        
        # Calculate Conditional VaR for normal distribution
        # CVaR = μ + σ * φ(Φ^(-1)(α)) / α
        phi_95 = stats.norm.pdf(stats.norm.ppf(0.05))
        phi_99 = stats.norm.pdf(stats.norm.ppf(0.01))
        
        cvar_95 = abs(mean_return + std_return * phi_95 / 0.05)
        cvar_99 = abs(mean_return + std_return * phi_99 / 0.01)
        
        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method=VaRMethod.PARAMETRIC,
            calculation_date=datetime.now()
        )
    
    async def _calculate_monte_carlo_var(self, returns: pd.Series) -> VaRResult:
        """Calculate Monte Carlo VaR."""
        
        recent_returns = returns.tail(self.var_window)
        
        if len(recent_returns) < 30:
            raise ValueError("Insufficient data for Monte Carlo VaR calculation")
        
        # Fit distribution parameters
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return, std_return, self.monte_carlo_simulations
        )
        
        # Calculate VaR from simulated returns
        var_95 = abs(np.percentile(simulated_returns, 5))
        var_99 = abs(np.percentile(simulated_returns, 1))
        
        # Calculate Conditional VaR
        cvar_95 = abs(simulated_returns[simulated_returns <= -var_95].mean())
        cvar_99 = abs(simulated_returns[simulated_returns <= -var_99].mean())
        
        # Calculate confidence intervals using bootstrap
        bootstrap_vars_95 = []
        bootstrap_vars_99 = []
        
        for _ in range(1000):  # Bootstrap iterations
            bootstrap_sample = np.random.choice(
                recent_returns, size=len(recent_returns), replace=True
            )
            bootstrap_mean = bootstrap_sample.mean()
            bootstrap_std = bootstrap_sample.std()
            
            bootstrap_sim = np.random.normal(
                bootstrap_mean, bootstrap_std, 1000
            )
            bootstrap_vars_95.append(abs(np.percentile(bootstrap_sim, 5)))
            bootstrap_vars_99.append(abs(np.percentile(bootstrap_sim, 1)))
        
        confidence_interval_95 = (
            np.percentile(bootstrap_vars_95, 2.5),
            np.percentile(bootstrap_vars_95, 97.5)
        )
        
        return VaRResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            method=VaRMethod.MONTE_CARLO,
            confidence_interval=confidence_interval_95,
            calculation_date=datetime.now()
        ) 
   
    async def _calculate_volatility(self, returns: pd.Series, method: VolatilityMethod) -> VolatilityResult:
        """Calculate volatility using the specified method."""
        
        if method == VolatilityMethod.HISTORICAL:
            return await self._calculate_historical_volatility(returns)
        elif method == VolatilityMethod.EWMA:
            return await self._calculate_ewma_volatility(returns)
        elif method == VolatilityMethod.GARCH:
            return await self._calculate_garch_volatility(returns)
        else:
            raise ValueError(f"Unknown volatility method: {method}")
    
    async def _calculate_historical_volatility(self, returns: pd.Series) -> VolatilityResult:
        """Calculate historical volatility."""
        
        recent_returns = returns.tail(self.volatility_window)
        
        if len(recent_returns) < 10:
            raise ValueError("Insufficient data for historical volatility calculation")
        
        daily_vol = recent_returns.std()
        annualized_vol = daily_vol * np.sqrt(252)  # Annualize assuming 252 trading days
        
        # Calculate realized volatility (using high-frequency intraday data if available)
        # For now, use close-to-close volatility as proxy
        realized_vol = daily_vol * np.sqrt(252)
        
        return VolatilityResult(
            daily_volatility=daily_vol,
            annualized_volatility=annualized_vol,
            method=VolatilityMethod.HISTORICAL,
            window_size=len(recent_returns),
            realized_volatility=realized_vol
        )
    
    async def _calculate_ewma_volatility(self, returns: pd.Series) -> VolatilityResult:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility."""
        
        recent_returns = returns.tail(self.volatility_window * 2)  # Use more data for EWMA
        
        if len(recent_returns) < 20:
            raise ValueError("Insufficient data for EWMA volatility calculation")
        
        # Calculate EWMA variance
        ewma_var = 0
        for i, ret in enumerate(reversed(recent_returns)):
            weight = (1 - self.ewma_lambda) * (self.ewma_lambda ** i)
            ewma_var += weight * (ret ** 2)
        
        daily_vol = np.sqrt(ewma_var)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return VolatilityResult(
            daily_volatility=daily_vol,
            annualized_volatility=annualized_vol,
            method=VolatilityMethod.EWMA,
            window_size=len(recent_returns)
        )
    
    async def _calculate_garch_volatility(self, returns: pd.Series) -> VolatilityResult:
        """Calculate GARCH(1,1) volatility."""
        
        recent_returns = returns.tail(self.volatility_window * 3)  # Use more data for GARCH
        
        if len(recent_returns) < 50:
            raise ValueError("Insufficient data for GARCH volatility calculation")
        
        try:
            # Simplified GARCH(1,1) implementation
            # In production, consider using arch library for more robust implementation
            
            # Initialize parameters
            omega = 0.000001  # Long-term variance
            alpha = 0.1       # ARCH parameter
            beta = 0.85       # GARCH parameter
            
            # Calculate conditional variances
            variances = []
            long_term_var = recent_returns.var()
            
            for i, ret in enumerate(recent_returns):
                if i == 0:
                    var_t = long_term_var
                else:
                    var_t = omega + alpha * (recent_returns.iloc[i-1] ** 2) + beta * variances[i-1]
                variances.append(var_t)
            
            # Current volatility is the square root of the last variance
            daily_vol = np.sqrt(variances[-1])
            annualized_vol = daily_vol * np.sqrt(252)
            
            return VolatilityResult(
                daily_volatility=daily_vol,
                annualized_volatility=annualized_vol,
                method=VolatilityMethod.GARCH,
                window_size=len(recent_returns)
            )
            
        except Exception as e:
            logger.warning(f"GARCH calculation failed, falling back to historical: {e}")
            return await self._calculate_historical_volatility(returns)
    
    def _calculate_max_drawdown(self, price_data: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        prices = price_data['close']
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Return maximum drawdown (as positive value)
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        
        if len(returns) < 30:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        
        if len(returns) < 30:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / 252)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        
        return (excess_returns.mean() / downside_deviation) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        
        if len(returns) < 30 or max_drawdown == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        
        return annual_return / max_drawdown
    
    async def _calculate_beta(self, returns: pd.Series, benchmark_data: pd.DataFrame) -> float:
        """Calculate beta relative to benchmark."""
        
        try:
            # Calculate benchmark returns
            benchmark_returns = self._calculate_returns(benchmark_data)
            
            # Align dates
            aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
            aligned_data.columns = ['stock_returns', 'benchmark_returns']
            
            if len(aligned_data) < 30:
                logger.warning("Insufficient aligned data for beta calculation")
                return None
            
            # Calculate beta using linear regression
            covariance = aligned_data['stock_returns'].cov(aligned_data['benchmark_returns'])
            benchmark_variance = aligned_data['benchmark_returns'].var()
            
            if benchmark_variance == 0:
                return 0.0
            
            beta = covariance / benchmark_variance
            
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return None
    
    async def _calculate_liquidity_risk_score(self, 
                                            price_data: pd.DataFrame, 
                                            volume_data: pd.DataFrame) -> float:
        """
        Calculate liquidity risk score based on volume patterns.
        Score ranges from 0 (highly liquid) to 100 (highly illiquid).
        """
        
        try:
            # Merge price and volume data
            merged_data = pd.merge(price_data, volume_data, on='date', how='inner')
            
            if len(merged_data) < 30:
                logger.warning("Insufficient data for liquidity risk calculation")
                return 50.0  # Neutral score
            
            # Calculate various liquidity metrics
            
            # 1. Volume-based metrics
            avg_volume = merged_data['volume'].mean()
            volume_volatility = merged_data['volume'].std() / avg_volume if avg_volume > 0 else 1.0
            
            # 2. Price impact metrics (simplified)
            # Calculate price changes relative to volume
            merged_data['price_change'] = merged_data['close'].pct_change()
            merged_data['volume_normalized'] = merged_data['volume'] / merged_data['volume'].rolling(20).mean()
            
            # Price impact = |price_change| / volume_normalized
            merged_data['price_impact'] = (
                abs(merged_data['price_change']) / 
                merged_data['volume_normalized'].replace(0, np.nan)
            )
            
            avg_price_impact = merged_data['price_impact'].median()
            
            # 3. Bid-ask spread proxy (using high-low spread)
            merged_data['spread_proxy'] = (merged_data['high'] - merged_data['low']) / merged_data['close']
            avg_spread = merged_data['spread_proxy'].mean()
            
            # 4. Zero-volume days
            zero_volume_ratio = (merged_data['volume'] == 0).sum() / len(merged_data)
            
            # Combine metrics into liquidity risk score
            # Higher values indicate higher liquidity risk (lower liquidity)
            
            # Normalize each component (0-100 scale)
            volume_score = min(100, volume_volatility * 50)  # Volume volatility component
            impact_score = min(100, avg_price_impact * 1000) if not np.isnan(avg_price_impact) else 50
            spread_score = min(100, avg_spread * 500)
            zero_volume_score = zero_volume_ratio * 100
            
            # Weighted combination
            liquidity_risk_score = (
                volume_score * 0.3 +
                impact_score * 0.3 +
                spread_score * 0.2 +
                zero_volume_score * 0.2
            )
            
            return min(100, max(0, liquidity_risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk score: {e}")
            return 50.0  # Return neutral score on error
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        returns = price_data['close'].pct_change().dropna()
        
        # Set index to date if available
        if 'date' in price_data.columns:
            returns.index = pd.to_datetime(price_data['date'].iloc[1:])  # Skip first row due to pct_change
        
        return returns
    
    def _validate_price_data(self, price_data: pd.DataFrame) -> None:
        """Validate input price data."""
        
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(price_data) < 30:
            raise ValueError("Insufficient data: minimum 30 data points required")
        
        # Check for non-numeric data
        if not pd.api.types.is_numeric_dtype(price_data['close']):
            raise ValueError("Close prices must be numeric")
        
        # Check for negative prices
        if (price_data['close'] <= 0).any():
            raise ValueError("Close prices must be positive")


# Utility functions for risk management

def calculate_portfolio_var(individual_vars: List[float], 
                          correlations: np.ndarray, 
                          weights: np.ndarray) -> float:
    """
    Calculate portfolio VaR using individual VaRs and correlation matrix.
    
    Args:
        individual_vars: List of individual asset VaRs
        correlations: Correlation matrix between assets
        weights: Portfolio weights
        
    Returns:
        Portfolio VaR
    """
    
    if len(individual_vars) != len(weights) or len(weights) != correlations.shape[0]:
        raise ValueError("Dimension mismatch between VaRs, weights, and correlation matrix")
    
    # Convert to numpy arrays
    vars_array = np.array(individual_vars)
    
    # Calculate portfolio VaR using quadratic form
    # Portfolio VaR = sqrt(w' * Σ * w) where Σ is the VaR covariance matrix
    var_covariance = np.outer(vars_array, vars_array) * correlations
    
    portfolio_var = np.sqrt(np.dot(weights, np.dot(var_covariance, weights)))
    
    return portfolio_var


def calculate_component_var(individual_vars: List[float],
                          correlations: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """
    Calculate component VaR for each asset in the portfolio.
    
    Args:
        individual_vars: List of individual asset VaRs
        correlations: Correlation matrix between assets
        weights: Portfolio weights
        
    Returns:
        Array of component VaRs
    """
    
    portfolio_var = calculate_portfolio_var(individual_vars, correlations, weights)
    
    if portfolio_var == 0:
        return np.zeros(len(weights))
    
    vars_array = np.array(individual_vars)
    var_covariance = np.outer(vars_array, vars_array) * correlations
    
    # Component VaR = (w_i * (Σ * w)_i) / Portfolio VaR
    marginal_vars = np.dot(var_covariance, weights)
    component_vars = (weights * marginal_vars) / portfolio_var
    
    return component_vars


async def stress_test_portfolio(risk_engine: EnhancedRiskManagementEngine,
                              portfolio_data: Dict[str, pd.DataFrame],
                              stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Perform stress testing on a portfolio.
    
    Args:
        risk_engine: Risk management engine instance
        portfolio_data: Dictionary mapping asset symbols to price data
        stress_scenarios: Dictionary mapping scenario names to shock parameters
        
    Returns:
        Dictionary mapping scenario names to stress test results
    """
    
    results = {}
    
    for scenario_name, shocks in stress_scenarios.items():
        scenario_results = {}
        
        for symbol, price_data in portfolio_data.items():
            try:
                # Apply shock to price data
                shocked_data = price_data.copy()
                
                if symbol in shocks:
                    shock_factor = 1 + shocks[symbol]  # e.g., -0.2 for 20% decline
                    shocked_data['close'] = shocked_data['close'] * shock_factor
                
                # Calculate risk metrics for shocked data
                risk_metrics = await risk_engine.calculate_comprehensive_risk_metrics(shocked_data)
                
                scenario_results[symbol] = {
                    'var_95': risk_metrics.var_results.get('historical', VaRResult(0, 0, 0, 0, VaRMethod.HISTORICAL)).var_95,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'volatility': risk_metrics.volatility_results.get('historical', VolatilityResult(0, 0, VolatilityMethod.HISTORICAL, 0)).annualized_volatility
                }
                
            except Exception as e:
                logger.error(f"Error in stress test for {symbol} in scenario {scenario_name}: {e}")
                scenario_results[symbol] = {'error': str(e)}
        
        results[scenario_name] = scenario_results
    
    return results