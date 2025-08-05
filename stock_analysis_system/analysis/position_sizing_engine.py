"""
Dynamic Position Sizing Engine

This module implements comprehensive position sizing strategies including Kelly Criterion,
risk-adjusted position sizing, portfolio concentration monitoring, and risk budget management.

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
from scipy.optimize import minimize, minimize_scalar

logger = logging.getLogger(__name__)


class PositionSizingMethod(str, Enum):
    """Position sizing methods"""

    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    VAR_BASED = "var_based"
    RISK_PARITY = "risk_parity"
    MAXIMUM_DIVERSIFICATION = "max_diversification"


class RiskBudgetMethod(str, Enum):
    """Risk budget allocation methods"""

    EQUAL_RISK = "equal_risk"
    INVERSE_VOLATILITY = "inverse_volatility"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


class ConcentrationRiskLevel(str, Enum):
    """Portfolio concentration risk levels"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation with rationale"""

    symbol: str
    recommended_weight: float  # Portfolio weight (0-1)
    recommended_shares: Optional[int] = None  # Number of shares
    recommended_dollar_amount: Optional[float] = None  # Dollar amount

    # Sizing rationale
    method_used: PositionSizingMethod = None
    kelly_fraction: Optional[float] = None
    risk_contribution: Optional[float] = None
    concentration_adjustment: Optional[float] = None

    # Risk metrics
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    var_95: Optional[float] = None
    max_loss_estimate: Optional[float] = None

    # Constraints applied
    min_weight_constraint: Optional[float] = None
    max_weight_constraint: Optional[float] = None
    liquidity_constraint: Optional[float] = None

    # Confidence and warnings
    confidence_level: Optional[float] = None
    warnings: List[str] = None

    calculation_date: datetime = None


@dataclass
class PortfolioRiskBudget:
    """Portfolio risk budget allocation"""

    total_risk_budget: float  # Total portfolio risk (e.g., 10% VaR)
    asset_risk_budgets: Dict[str, float]  # Risk budget per asset
    asset_weights: Dict[str, float]  # Optimal weights
    risk_contributions: Dict[str, float]  # Actual risk contributions

    # Budget utilization
    budget_utilization: float  # Percentage of budget used
    diversification_ratio: float  # Portfolio diversification measure
    concentration_metrics: Dict[str, float]  # Concentration measures

    # Method and constraints
    method_used: RiskBudgetMethod = None
    constraints_applied: List[str] = None

    calculation_date: datetime = None


@dataclass
class ConcentrationRiskAnalysis:
    """Portfolio concentration risk analysis"""

    concentration_level: ConcentrationRiskLevel
    concentration_score: float  # 0-100 scale

    # Concentration metrics
    herfindahl_index: float  # Sum of squared weights
    effective_number_of_assets: float  # 1/HHI
    max_weight: float  # Largest position weight
    top_5_concentration: float  # Weight of top 5 positions

    # Sector/industry concentration
    sector_concentration: Optional[Dict[str, float]] = None
    industry_concentration: Optional[Dict[str, float]] = None

    # Risk concentration
    risk_concentration: Optional[Dict[str, float]] = None  # Risk contribution by asset

    # Recommendations
    concentration_warnings: List[str] = None
    diversification_recommendations: List[str] = None

    calculation_date: datetime = None


class DynamicPositionSizingEngine:
    """
    Dynamic Position Sizing Engine with multiple sizing methods and risk management
    """

    def __init__(
        self,
        default_risk_budget: float = 0.02,  # 2% portfolio risk budget
        max_position_weight: float = 0.20,  # 20% max position
        min_position_weight: float = 0.01,  # 1% min position
        kelly_multiplier: float = 0.25,  # Kelly fraction multiplier
        concentration_threshold: float = 0.60,
    ):  # 60% concentration warning
        """
        Initialize the Dynamic Position Sizing Engine.

        Args:
            default_risk_budget: Default portfolio risk budget (VaR)
            max_position_weight: Maximum weight for any single position
            min_position_weight: Minimum weight for any position
            kelly_multiplier: Multiplier for Kelly Criterion (for safety)
            concentration_threshold: Threshold for concentration warnings
        """
        self.default_risk_budget = default_risk_budget
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight
        self.kelly_multiplier = kelly_multiplier
        self.concentration_threshold = concentration_threshold

        # Risk-free rate for calculations
        self.risk_free_rate = 0.03

        # Liquidity constraints
        self.min_liquidity_score = 30  # Minimum liquidity score for full position

    async def calculate_position_size(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        portfolio_value: float,
        method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION,
        risk_metrics: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size for a single asset.

        Args:
            symbol: Asset symbol
            price_data: Historical price data
            portfolio_value: Total portfolio value
            method: Position sizing method to use
            risk_metrics: Pre-calculated risk metrics
            market_data: Additional market data

        Returns:
            PositionSizeRecommendation with optimal sizing
        """

        try:
            # Calculate or extract risk metrics
            if risk_metrics is None:
                risk_metrics = await self._calculate_basic_risk_metrics(price_data)

            # Get current price
            current_price = price_data["close"].iloc[-1]

            # Calculate position size based on method
            if method == PositionSizingMethod.KELLY_CRITERION:
                weight, kelly_fraction = await self._calculate_kelly_position(
                    risk_metrics
                )
            elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                weight = await self._calculate_volatility_adjusted_position(
                    risk_metrics
                )
                kelly_fraction = None
            elif method == PositionSizingMethod.VAR_BASED:
                weight = await self._calculate_var_based_position(risk_metrics)
                kelly_fraction = None
            elif method == PositionSizingMethod.FIXED_FRACTIONAL:
                weight = await self._calculate_fixed_fractional_position(risk_metrics)
                kelly_fraction = None
            else:
                weight = 0.05  # Default 5%
                kelly_fraction = None

            # Apply constraints
            original_weight = weight
            weight = self._apply_position_constraints(weight, risk_metrics)

            # Calculate dollar amount and shares
            dollar_amount = weight * portfolio_value
            shares = int(dollar_amount / current_price) if current_price > 0 else 0

            # Adjust for actual shares that can be purchased
            actual_dollar_amount = shares * current_price
            actual_weight = (
                actual_dollar_amount / portfolio_value if portfolio_value > 0 else 0
            )

            # Generate warnings
            warnings = []
            if weight != original_weight:
                warnings.append(
                    f"Position size adjusted from {original_weight:.3f} to {weight:.3f} due to constraints"
                )

            if risk_metrics.get("liquidity_score", 100) < self.min_liquidity_score:
                warnings.append("Low liquidity - consider reducing position size")

            if risk_metrics.get("volatility", 0) > 0.5:  # 50% annual volatility
                warnings.append("High volatility asset - increased risk")

            # Calculate confidence level
            confidence = self._calculate_confidence_level(risk_metrics, method)

            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_weight=actual_weight,
                recommended_shares=shares,
                recommended_dollar_amount=actual_dollar_amount,
                method_used=method,
                kelly_fraction=kelly_fraction,
                expected_return=risk_metrics.get("expected_return"),
                volatility=risk_metrics.get("volatility"),
                var_95=risk_metrics.get("var_95"),
                max_loss_estimate=actual_dollar_amount
                * risk_metrics.get("var_95", 0.05),
                min_weight_constraint=self.min_position_weight,
                max_weight_constraint=self.max_position_weight,
                confidence_level=confidence,
                warnings=warnings,
                calculation_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            raise

    async def optimize_portfolio_risk_budget(
        self,
        assets: Dict[str, pd.DataFrame],
        portfolio_value: float,
        risk_budget: Optional[float] = None,
        method: RiskBudgetMethod = RiskBudgetMethod.RISK_PARITY,
        constraints: Optional[Dict] = None,
    ) -> PortfolioRiskBudget:
        """
        Optimize portfolio weights based on risk budget allocation.

        Args:
            assets: Dictionary of asset symbols to price data
            portfolio_value: Total portfolio value
            risk_budget: Total portfolio risk budget (default: self.default_risk_budget)
            method: Risk budget allocation method
            constraints: Additional constraints

        Returns:
            PortfolioRiskBudget with optimal allocation
        """

        try:
            if risk_budget is None:
                risk_budget = self.default_risk_budget

            # Calculate risk metrics for all assets
            asset_metrics = {}
            for symbol, price_data in assets.items():
                asset_metrics[symbol] = await self._calculate_basic_risk_metrics(
                    price_data
                )

            # Build covariance matrix
            returns_data = {}
            for symbol, price_data in assets.items():
                returns = price_data["close"].pct_change().dropna()
                returns_data[symbol] = returns

            # Align returns data
            aligned_returns = pd.DataFrame(returns_data).dropna()

            if len(aligned_returns) < 30:
                raise ValueError("Insufficient aligned data for portfolio optimization")

            # Calculate covariance matrix (annualized)
            cov_matrix = aligned_returns.cov() * 252

            # Optimize based on method
            if method == RiskBudgetMethod.RISK_PARITY:
                weights = await self._optimize_risk_parity(cov_matrix, constraints)
            elif method == RiskBudgetMethod.EQUAL_RISK:
                weights = await self._optimize_equal_risk_contribution(
                    cov_matrix, constraints
                )
            elif method == RiskBudgetMethod.INVERSE_VOLATILITY:
                weights = await self._optimize_inverse_volatility(
                    asset_metrics, constraints
                )
            else:
                # Default to equal weights
                n_assets = len(assets)
                weights = {symbol: 1.0 / n_assets for symbol in assets.keys()}

            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)

            # Calculate portfolio metrics
            portfolio_volatility = self._calculate_portfolio_volatility(
                weights, cov_matrix
            )
            diversification_ratio = self._calculate_diversification_ratio(
                weights, asset_metrics, cov_matrix
            )

            # Calculate concentration metrics
            concentration_metrics = self._calculate_concentration_metrics(weights)

            # Calculate risk budgets
            asset_risk_budgets = {}
            for symbol in assets.keys():
                asset_risk_budgets[symbol] = risk_contributions[symbol] * risk_budget

            # Budget utilization
            total_risk_contribution = sum(risk_contributions.values())
            budget_utilization = min(1.0, total_risk_contribution)

            return PortfolioRiskBudget(
                total_risk_budget=risk_budget,
                asset_risk_budgets=asset_risk_budgets,
                asset_weights=weights,
                risk_contributions=risk_contributions,
                budget_utilization=budget_utilization,
                diversification_ratio=diversification_ratio,
                concentration_metrics=concentration_metrics,
                method_used=method,
                constraints_applied=list(constraints.keys()) if constraints else [],
                calculation_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error optimizing portfolio risk budget: {e}")
            raise

    async def analyze_concentration_risk(
        self,
        portfolio_weights: Dict[str, float],
        asset_sectors: Optional[Dict[str, str]] = None,
        risk_contributions: Optional[Dict[str, float]] = None,
    ) -> ConcentrationRiskAnalysis:
        """
        Analyze portfolio concentration risk.

        Args:
            portfolio_weights: Current portfolio weights
            asset_sectors: Asset to sector mapping
            risk_contributions: Risk contribution by asset

        Returns:
            ConcentrationRiskAnalysis with detailed concentration metrics
        """

        try:
            # Calculate basic concentration metrics
            weights = np.array(list(portfolio_weights.values()))

            # Herfindahl-Hirschman Index
            hhi = np.sum(weights**2)
            effective_n_assets = 1.0 / hhi if hhi > 0 else 0

            # Maximum weight
            max_weight = np.max(weights)

            # Top 5 concentration
            sorted_weights = np.sort(weights)[::-1]  # Descending order
            top_5_concentration = np.sum(sorted_weights[: min(5, len(sorted_weights))])

            # Calculate concentration score (0-100)
            concentration_score = self._calculate_concentration_score(
                hhi, max_weight, top_5_concentration
            )

            # Determine concentration level
            if concentration_score < 30:
                concentration_level = ConcentrationRiskLevel.LOW
            elif concentration_score < 50:
                concentration_level = ConcentrationRiskLevel.MODERATE
            elif concentration_score < 75:
                concentration_level = ConcentrationRiskLevel.HIGH
            else:
                concentration_level = ConcentrationRiskLevel.EXTREME

            # Sector concentration analysis
            sector_concentration = None
            if asset_sectors:
                sector_weights = {}
                for asset, weight in portfolio_weights.items():
                    sector = asset_sectors.get(asset, "Unknown")
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight
                sector_concentration = sector_weights

            # Generate warnings and recommendations
            warnings = []
            recommendations = []

            if max_weight > self.max_position_weight:
                warnings.append(
                    f"Single position exceeds maximum weight limit ({max_weight:.1%} > {self.max_position_weight:.1%})"
                )

            if top_5_concentration > self.concentration_threshold:
                warnings.append(
                    f"Top 5 positions represent {top_5_concentration:.1%} of portfolio"
                )

            if effective_n_assets < 5:
                warnings.append(
                    f"Portfolio effectively contains only {effective_n_assets:.1f} independent positions"
                )
                recommendations.append("Consider adding more uncorrelated assets")

            if concentration_level in [
                ConcentrationRiskLevel.HIGH,
                ConcentrationRiskLevel.EXTREME,
            ]:
                recommendations.append("Reduce position sizes in largest holdings")
                recommendations.append(
                    "Increase diversification across sectors and asset classes"
                )

            return ConcentrationRiskAnalysis(
                concentration_level=concentration_level,
                concentration_score=concentration_score,
                herfindahl_index=hhi,
                effective_number_of_assets=effective_n_assets,
                max_weight=max_weight,
                top_5_concentration=top_5_concentration,
                sector_concentration=sector_concentration,
                risk_concentration=risk_contributions,
                concentration_warnings=warnings,
                diversification_recommendations=recommendations,
                calculation_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error analyzing concentration risk: {e}")
            raise

    async def generate_portfolio_recommendations(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """
        Generate specific trading recommendations to move from current to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            transaction_costs: Transaction costs by asset

        Returns:
            Dictionary of trading recommendations
        """

        try:
            recommendations = {}

            # Get all assets (current + target)
            all_assets = set(current_weights.keys()) | set(target_weights.keys())

            for asset in all_assets:
                current_weight = current_weights.get(asset, 0.0)
                target_weight = target_weights.get(asset, 0.0)

                weight_diff = target_weight - current_weight
                dollar_diff = weight_diff * portfolio_value

                if abs(weight_diff) < 0.001:  # Less than 0.1% difference
                    continue

                # Calculate transaction cost
                transaction_cost = 0.0
                if transaction_costs and asset in transaction_costs:
                    transaction_cost = abs(dollar_diff) * transaction_costs[asset]

                # Determine action
                if weight_diff > 0:
                    action = "BUY"
                    action_description = f"Increase position by {weight_diff:.2%}"
                else:
                    action = "SELL"
                    action_description = f"Reduce position by {abs(weight_diff):.2%}"

                # Calculate priority (larger differences = higher priority)
                priority = min(10, int(abs(weight_diff) * 100))  # 1-10 scale

                recommendations[asset] = {
                    "action": action,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_diff,
                    "dollar_amount": abs(dollar_diff),
                    "transaction_cost": transaction_cost,
                    "net_amount": abs(dollar_diff) - transaction_cost,
                    "priority": priority,
                    "description": action_description,
                }

            return recommendations

        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            raise

    # Private helper methods

    async def _calculate_basic_risk_metrics(self, price_data: pd.DataFrame) -> Dict:
        """Calculate basic risk metrics for an asset."""

        returns = price_data["close"].pct_change().dropna()

        if len(returns) < 30:
            raise ValueError("Insufficient data for risk metric calculation")

        # Basic metrics
        expected_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = abs(np.percentile(returns, 5))  # 5th percentile

        # Sharpe ratio
        excess_return = expected_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        prices = price_data["close"]
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "var_95": var_95,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "liquidity_score": 75.0,  # Default liquidity score
        }

    async def _calculate_kelly_position(
        self, risk_metrics: Dict
    ) -> Tuple[float, float]:
        """Calculate Kelly Criterion position size."""

        expected_return = risk_metrics.get("expected_return", 0)
        volatility = risk_metrics.get("volatility", 0.2)

        if volatility == 0:
            return 0.0, 0.0

        # Kelly fraction = (expected return - risk free rate) / variance
        excess_return = expected_return - self.risk_free_rate
        kelly_fraction = excess_return / (volatility**2)

        # Apply safety multiplier and constraints
        kelly_fraction = max(0, kelly_fraction)  # No short positions
        safe_kelly = kelly_fraction * self.kelly_multiplier

        return safe_kelly, kelly_fraction

    async def _calculate_volatility_adjusted_position(
        self, risk_metrics: Dict
    ) -> float:
        """Calculate volatility-adjusted position size."""

        volatility = risk_metrics.get("volatility", 0.2)
        target_volatility = 0.15  # 15% target volatility

        if volatility == 0:
            return 0.0

        # Scale position inversely with volatility
        weight = target_volatility / volatility
        return min(weight, self.max_position_weight)

    async def _calculate_var_based_position(self, risk_metrics: Dict) -> float:
        """Calculate VaR-based position size."""

        var_95 = risk_metrics.get("var_95", 0.05)
        target_var = 0.02  # 2% target VaR

        if var_95 == 0:
            return 0.0

        # Scale position to achieve target VaR
        weight = target_var / var_95
        return min(weight, self.max_position_weight)

    async def _calculate_fixed_fractional_position(self, risk_metrics: Dict) -> float:
        """Calculate fixed fractional position size."""

        # Simple fixed fraction based on risk level
        volatility = risk_metrics.get("volatility", 0.2)

        if volatility < 0.15:  # Low volatility
            return 0.10  # 10%
        elif volatility < 0.30:  # Medium volatility
            return 0.07  # 7%
        else:  # High volatility
            return 0.05  # 5%

    def _apply_position_constraints(self, weight: float, risk_metrics: Dict) -> float:
        """Apply position size constraints."""

        # Basic weight constraints
        weight = max(self.min_position_weight, weight)
        weight = min(self.max_position_weight, weight)

        # Liquidity constraints
        liquidity_score = risk_metrics.get("liquidity_score", 100)
        if liquidity_score < 50:  # Low liquidity
            max_illiquid_weight = 0.05  # 5% max for illiquid assets
            weight = min(weight, max_illiquid_weight)

        # Volatility constraints
        volatility = risk_metrics.get("volatility", 0)
        if volatility > 0.5:  # Very high volatility
            max_volatile_weight = 0.03  # 3% max for very volatile assets
            weight = min(weight, max_volatile_weight)

        return weight

    def _calculate_confidence_level(
        self, risk_metrics: Dict, method: PositionSizingMethod
    ) -> float:
        """Calculate confidence level for position size recommendation."""

        base_confidence = 0.7  # 70% base confidence

        # Adjust based on data quality
        if len(risk_metrics) >= 5:  # Good risk metrics available
            base_confidence += 0.1

        # Adjust based on method
        if method == PositionSizingMethod.KELLY_CRITERION:
            # Kelly requires good return estimates
            sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.0:
                base_confidence += 0.1
            elif sharpe_ratio < 0:
                base_confidence -= 0.2

        # Adjust based on volatility
        volatility = risk_metrics.get("volatility", 0.2)
        if volatility > 0.4:  # High volatility reduces confidence
            base_confidence -= 0.1

        return max(0.3, min(0.95, base_confidence))  # Keep between 30% and 95%

    async def _optimize_risk_parity(
        self, cov_matrix: pd.DataFrame, constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize portfolio for risk parity (equal risk contribution)."""

        try:
            assets = cov_matrix.columns.tolist()
            n_assets = len(assets)

            # Objective function: minimize sum of squared differences in risk contributions
            def risk_parity_objective(weights):
                weights = np.array(weights)

                # Portfolio variance
                portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))

                if portfolio_var <= 0:
                    return 1e6  # Large penalty for invalid portfolio

                # Risk contributions
                marginal_contrib = np.dot(cov_matrix.values, weights)
                risk_contrib = weights * marginal_contrib / portfolio_var

                # Target equal risk contribution
                target_contrib = 1.0 / n_assets

                # Sum of squared deviations from target
                return np.sum((risk_contrib - target_contrib) ** 2)

            # Constraints
            constraints_list = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]

            # Bounds (non-negative weights)
            bounds = [(0.01, self.max_position_weight) for _ in range(n_assets)]

            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"maxiter": 1000},
            )

            if result.success:
                weights_dict = {
                    asset: weight for asset, weight in zip(assets, result.x)
                }
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                weights_dict = {asset: 1.0 / n_assets for asset in assets}

            return weights_dict

        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            # Fallback to equal weights
            return {
                asset: 1.0 / len(cov_matrix.columns) for asset in cov_matrix.columns
            }

    async def _optimize_equal_risk_contribution(
        self, cov_matrix: pd.DataFrame, constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize for equal risk contribution (similar to risk parity but different implementation)."""

        try:
            assets = cov_matrix.columns.tolist()
            n_assets = len(assets)

            # Use inverse volatility as starting point
            volatilities = np.sqrt(np.diag(cov_matrix.values))
            inv_vol_weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)

            # Iterative approach to achieve equal risk contribution
            weights = inv_vol_weights.copy()

            for iteration in range(50):  # Max 50 iterations
                # Calculate risk contributions
                portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))

                if portfolio_var <= 0:
                    break

                marginal_contrib = np.dot(cov_matrix.values, weights)
                risk_contrib = weights * marginal_contrib / portfolio_var

                # Adjust weights to equalize risk contributions
                target_contrib = 1.0 / n_assets
                adjustment_factor = np.sqrt(target_contrib / (risk_contrib + 1e-8))

                # Update weights
                new_weights = weights * adjustment_factor
                new_weights = new_weights / np.sum(new_weights)  # Normalize

                # Apply constraints
                new_weights = np.clip(new_weights, 0.01, self.max_position_weight)
                new_weights = new_weights / np.sum(new_weights)  # Renormalize

                # Check convergence
                if np.max(np.abs(new_weights - weights)) < 1e-6:
                    break

                weights = new_weights

            return {asset: weight for asset, weight in zip(assets, weights)}

        except Exception as e:
            logger.error(f"Error in equal risk contribution optimization: {e}")
            return {
                asset: 1.0 / len(cov_matrix.columns) for asset in cov_matrix.columns
            }

    async def _optimize_inverse_volatility(
        self, asset_metrics: Dict, constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize using inverse volatility weighting."""

        try:
            # Extract volatilities
            volatilities = {}
            for asset, metrics in asset_metrics.items():
                vol = metrics.get("volatility", 0.2)
                volatilities[asset] = max(
                    vol, 0.01
                )  # Minimum volatility to avoid division by zero

            # Calculate inverse volatility weights
            inv_volatilities = {asset: 1.0 / vol for asset, vol in volatilities.items()}
            total_inv_vol = sum(inv_volatilities.values())

            # Normalize to sum to 1
            weights = {
                asset: inv_vol / total_inv_vol
                for asset, inv_vol in inv_volatilities.items()
            }

            # Apply position limits
            for asset in weights:
                weights[asset] = min(weights[asset], self.max_position_weight)
                weights[asset] = max(weights[asset], self.min_position_weight)

            # Renormalize after applying constraints
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {
                    asset: weight / total_weight for asset, weight in weights.items()
                }

            return weights

        except Exception as e:
            logger.error(f"Error in inverse volatility optimization: {e}")
            return {asset: 1.0 / len(asset_metrics) for asset in asset_metrics.keys()}

    def _calculate_risk_contributions(
        self, weights: Dict[str, float], cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk contributions for each asset."""

        try:
            assets = list(weights.keys())
            weight_vector = np.array([weights[asset] for asset in assets])

            # Portfolio variance
            portfolio_var = np.dot(
                weight_vector, np.dot(cov_matrix.values, weight_vector)
            )

            if portfolio_var <= 0:
                return {asset: 0.0 for asset in assets}

            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix.values, weight_vector)

            # Risk contributions (weight * marginal contribution / portfolio variance)
            risk_contributions = {}
            for i, asset in enumerate(assets):
                risk_contrib = weight_vector[i] * marginal_contrib[i] / portfolio_var
                risk_contributions[asset] = risk_contrib

            return risk_contributions

        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return {asset: 0.0 for asset in weights.keys()}

    def _calculate_portfolio_volatility(
        self, weights: Dict[str, float], cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility."""

        try:
            assets = list(weights.keys())
            weight_vector = np.array([weights[asset] for asset in assets])

            portfolio_var = np.dot(
                weight_vector, np.dot(cov_matrix.values, weight_vector)
            )
            return np.sqrt(max(0, portfolio_var))

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    def _calculate_diversification_ratio(
        self, weights: Dict[str, float], asset_metrics: Dict, cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate diversification ratio (weighted average volatility / portfolio volatility)."""

        try:
            # Weighted average of individual volatilities
            weighted_avg_vol = 0.0
            for asset, weight in weights.items():
                vol = asset_metrics[asset].get("volatility", 0.2)
                weighted_avg_vol += weight * vol

            # Portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(weights, cov_matrix)

            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0

    def _calculate_concentration_metrics(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate various concentration metrics."""

        try:
            weight_values = np.array(list(weights.values()))

            # Herfindahl-Hirschman Index
            hhi = np.sum(weight_values**2)

            # Effective number of assets
            effective_n = 1.0 / hhi if hhi > 0 else 0

            # Maximum weight
            max_weight = np.max(weight_values)

            # Top N concentrations
            sorted_weights = np.sort(weight_values)[::-1]
            top_3 = np.sum(sorted_weights[: min(3, len(sorted_weights))])
            top_5 = np.sum(sorted_weights[: min(5, len(sorted_weights))])

            # Gini coefficient (measure of inequality)
            n = len(weight_values)
            if n > 1:
                sorted_weights_asc = np.sort(weight_values)
                cumsum = np.cumsum(sorted_weights_asc)
                gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights_asc))) / (
                    n * np.sum(sorted_weights_asc)
                ) - (n + 1) / n
            else:
                gini = 0.0

            return {
                "herfindahl_index": hhi,
                "effective_number_assets": effective_n,
                "max_weight": max_weight,
                "top_3_concentration": top_3,
                "top_5_concentration": top_5,
                "gini_coefficient": gini,
            }

        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {
                "herfindahl_index": 0.0,
                "effective_number_assets": 0.0,
                "max_weight": 0.0,
                "top_3_concentration": 0.0,
                "top_5_concentration": 0.0,
                "gini_coefficient": 0.0,
            }

    def _calculate_concentration_score(
        self, hhi: float, max_weight: float, top_5_concentration: float
    ) -> float:
        """Calculate overall concentration score (0-100)."""

        try:
            # HHI component (0-40 points)
            hhi_score = min(40, hhi * 100)  # HHI of 0.4 = 40 points

            # Max weight component (0-30 points)
            max_weight_score = min(30, max_weight * 100)  # 30% max weight = 30 points

            # Top 5 concentration component (0-30 points)
            top_5_score = min(
                30, (top_5_concentration - 0.5) * 60
            )  # Above 50% starts scoring
            top_5_score = max(0, top_5_score)

            total_score = hhi_score + max_weight_score + top_5_score
            return min(100, total_score)

        except Exception as e:
            logger.error(f"Error calculating concentration score: {e}")
            return 50.0  # Default moderate concentration


# Utility functions for position sizing


def calculate_optimal_rebalancing_frequency(
    portfolio_weights: Dict[str, float],
    transaction_costs: Dict[str, float],
    volatilities: Dict[str, float],
) -> int:
    """
    Calculate optimal rebalancing frequency based on transaction costs and volatilities.

    Args:
        portfolio_weights: Current portfolio weights
        transaction_costs: Transaction costs by asset (as fraction)
        volatilities: Asset volatilities (annualized)

    Returns:
        Optimal rebalancing frequency in days
    """

    try:
        # Weighted average transaction cost
        avg_transaction_cost = sum(
            portfolio_weights[asset] * transaction_costs.get(asset, 0.001)
            for asset in portfolio_weights
        )

        # Weighted average volatility
        avg_volatility = sum(
            portfolio_weights[asset] * volatilities.get(asset, 0.2)
            for asset in portfolio_weights
        )

        # Optimal frequency (simplified model)
        # Higher costs -> less frequent rebalancing
        # Higher volatility -> more frequent rebalancing

        if avg_volatility > 0 and avg_transaction_cost > 0:
            # Rule of thumb: rebalance when drift cost equals transaction cost
            optimal_days = int(np.sqrt(avg_transaction_cost / avg_volatility) * 252)
            return max(7, min(365, optimal_days))  # Between weekly and yearly
        else:
            return 30  # Default monthly

    except Exception as e:
        logger.error(f"Error calculating optimal rebalancing frequency: {e}")
        return 30  # Default monthly


def calculate_position_size_impact(
    position_size: float,
    daily_volume: float,
    price: float,
    participation_rate: float = 0.1,
) -> Dict[str, float]:
    """
    Calculate market impact of a position size.

    Args:
        position_size: Position size in shares
        daily_volume: Average daily volume
        price: Current price
        participation_rate: Maximum participation rate in daily volume

    Returns:
        Dictionary with impact metrics
    """

    try:
        # Position value
        position_value = position_size * price

        # Volume participation
        volume_participation = position_size / daily_volume if daily_volume > 0 else 1.0

        # Estimated market impact (simplified model)
        # Impact increases non-linearly with participation rate
        if volume_participation <= participation_rate:
            market_impact = (
                volume_participation * 0.01
            )  # 1% impact per 10% participation
        else:
            # Higher impact for large trades
            excess_participation = volume_participation - participation_rate
            market_impact = participation_rate * 0.01 + excess_participation * 0.05

        # Trading days required (assuming max participation rate)
        days_to_trade = max(1, int(np.ceil(volume_participation / participation_rate)))

        # Impact cost in dollars
        impact_cost = position_value * market_impact

        return {
            "volume_participation": volume_participation,
            "market_impact_pct": market_impact,
            "impact_cost_dollars": impact_cost,
            "days_to_trade": days_to_trade,
            "recommended_max_daily_shares": daily_volume * participation_rate,
        }

    except Exception as e:
        logger.error(f"Error calculating position size impact: {e}")
        return {
            "volume_participation": 0.0,
            "market_impact_pct": 0.0,
            "impact_cost_dollars": 0.0,
            "days_to_trade": 1,
            "recommended_max_daily_shares": 0.0,
        }


async def backtest_position_sizing_strategy(
    sizing_engine: DynamicPositionSizingEngine,
    historical_data: Dict[str, pd.DataFrame],
    initial_capital: float = 1000000,
    rebalance_frequency: int = 30,
) -> Dict[str, any]:
    """
    Backtest a position sizing strategy.

    Args:
        sizing_engine: Position sizing engine
        historical_data: Historical price data by asset
        initial_capital: Initial portfolio capital
        rebalance_frequency: Rebalancing frequency in days

    Returns:
        Backtest results
    """

    try:
        # Align all data to common dates
        all_dates = None
        aligned_data = {}

        for symbol, data in historical_data.items():
            data = data.copy()
            data["date"] = pd.to_datetime(data["date"])
            data.set_index("date", inplace=True)

            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)

            aligned_data[symbol] = data

        # Filter to common dates
        for symbol in aligned_data:
            aligned_data[symbol] = aligned_data[symbol].loc[all_dates]

        if len(all_dates) < 100:
            raise ValueError("Insufficient aligned historical data")

        # Backtest parameters
        portfolio_values = []
        portfolio_weights = []
        rebalance_dates = []

        current_capital = initial_capital
        current_weights = {symbol: 0.0 for symbol in aligned_data.keys()}

        # Run backtest
        for i, date in enumerate(all_dates):
            # Check if rebalancing day
            if i == 0 or i % rebalance_frequency == 0:
                # Calculate new target weights
                try:
                    # Use data up to current date for position sizing
                    lookback_data = {}
                    for symbol, data in aligned_data.items():
                        lookback_data[symbol] = data.iloc[: i + 1]

                    # Optimize portfolio
                    risk_budget = await sizing_engine.optimize_portfolio_risk_budget(
                        lookback_data, current_capital
                    )

                    current_weights = risk_budget.asset_weights
                    rebalance_dates.append(date)

                except Exception as e:
                    logger.warning(f"Rebalancing failed on {date}: {e}")
                    # Keep current weights

            # Calculate portfolio value
            portfolio_value = 0.0
            for symbol, weight in current_weights.items():
                if symbol in aligned_data:
                    price = aligned_data[symbol].loc[date, "close"]
                    portfolio_value += (
                        weight
                        * current_capital
                        * (price / aligned_data[symbol].iloc[0]["close"])
                    )

            portfolio_values.append(portfolio_value)
            portfolio_weights.append(current_weights.copy())

        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0

        max_drawdown = 0.0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "portfolio_values": portfolio_values,
            "portfolio_weights": portfolio_weights,
            "rebalance_dates": rebalance_dates,
            "final_capital": portfolio_values[-1],
        }

    except Exception as e:
        logger.error(f"Error in position sizing backtest: {e}")
        raise
