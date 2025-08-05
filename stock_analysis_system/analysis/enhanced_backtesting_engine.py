"""Enhanced Backtesting Engine with event-driven simulation."""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Position representation."""

    symbol: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Portfolio:
    """Portfolio representation."""

    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    buying_power: float = 0.0


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    benchmark: str = "000300.SH"  # CSI300
    rebalance_frequency: str = "monthly"
    max_position_size: float = 0.1  # 10% max position
    risk_free_rate: float = 0.03
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    commission_per_share: float = 0.0
    minimum_commission: float = 5.0
    margin_requirement: float = 1.0  # 100% margin requirement


@dataclass
class BacktestResult:
    """Backtesting results."""

    strategy_name: str
    config: BacktestConfig
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    equity_curve: pd.Series
    monthly_returns: pd.Series
    trade_log: List[Dict]
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]


class BaseStrategy(ABC):
    """Base strategy class for backtesting."""

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.indicators = {}
        self.state = {}

    @abstractmethod
    async def generate_signals(
        self, data: pd.Series, portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Generate trading signals based on current market data."""
        pass

    @abstractmethod
    async def calculate_position_size(
        self, signal: Dict[str, Any], portfolio: Portfolio, current_price: float
    ) -> float:
        """Calculate position size for a given signal."""
        pass

    async def on_bar(self, data: pd.Series, portfolio: Portfolio) -> List[Order]:
        """Called on each bar of data."""
        signals = await self.generate_signals(data, portfolio)
        orders = []

        for signal_type, signal_data in signals.items():
            if signal_type == "buy_signal" and signal_data:
                size = await self.calculate_position_size(
                    signal_data, portfolio, data["close_price"]
                )
                if size > 0:
                    order = Order(
                        order_id=f"buy_{data['stock_code']}_{data['trade_date']}",
                        symbol=data["stock_code"],
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=size,
                        timestamp=pd.to_datetime(data["trade_date"]),
                    )
                    orders.append(order)

            elif signal_type == "sell_signal" and signal_data:
                # Sell existing position
                if data["stock_code"] in portfolio.positions:
                    position = portfolio.positions[data["stock_code"]]
                    if position.quantity > 0:
                        order = Order(
                            order_id=f"sell_{data['stock_code']}_{data['trade_date']}",
                            symbol=data["stock_code"],
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=position.quantity,
                            timestamp=pd.to_datetime(data["trade_date"]),
                        )
                        orders.append(order)

        return orders


class SimpleMovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy."""

    def __init__(self, params: Dict[str, Any] = None):
        default_params = {"ma_short": 10, "ma_long": 30, "position_size": 0.1}
        if params:
            default_params.update(params)
        super().__init__(default_params)

    async def generate_signals(
        self, data: pd.Series, portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Generate signals based on moving average crossover."""
        # This is a simplified implementation
        # In practice, you'd need historical data to calculate moving averages

        # For demo purposes, generate random signals
        import random

        signals = {
            "buy_signal": random.random() > 0.95,  # 5% chance
            "sell_signal": random.random() > 0.98,  # 2% chance
        }

        return signals

    async def calculate_position_size(
        self, signal: Dict[str, Any], portfolio: Portfolio, current_price: float
    ) -> float:
        """Calculate position size based on available capital."""
        max_position_value = portfolio.cash * self.params["position_size"]
        return max_position_value / current_price if current_price > 0 else 0


class EnhancedBacktestingEngine:
    """Enhanced backtesting engine with event-driven simulation."""

    def __init__(self, data_source_manager=None):
        self.data_source = data_source_manager
        self.benchmark_cache = {}
        self.performance_cache = {}

    async def run_comprehensive_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        stock_data: pd.DataFrame = None,
    ) -> BacktestResult:
        """Run comprehensive backtesting with multiple validation methods."""

        # Fetch historical data if not provided
        if stock_data is None:
            stock_data = await self._fetch_stock_data(config)

        # Fetch benchmark data
        benchmark_data = await self._fetch_benchmark_data(config)

        # Run main backtest
        main_result = await self._run_single_backtest(
            strategy, config, stock_data, benchmark_data
        )

        # Run walk-forward analysis to check for overfitting
        wf_results = await self._run_walk_forward_analysis(strategy, config, stock_data)

        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(main_result, wf_results)

        # Add stability metrics to main result
        main_result.risk_metrics.update(stability_metrics)

        return main_result

    async def _run_single_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        stock_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
    ) -> BacktestResult:
        """Run a single backtest with event-driven simulation."""

        # Initialize portfolio
        portfolio = Portfolio(
            cash=config.initial_capital,
            total_value=config.initial_capital,
            equity=config.initial_capital,
            buying_power=config.initial_capital,
        )

        trades = []
        equity_values = []
        daily_returns = []

        # Sort data by date
        stock_data = stock_data.sort_values("trade_date").reset_index(drop=True)

        # Event-driven simulation
        for idx, row in stock_data.iterrows():
            # Generate orders from strategy
            orders = await strategy.on_bar(row, portfolio)

            # Execute orders
            for order in orders:
                trade = await self._execute_order(order, row, portfolio, config)
                if trade:
                    trades.append(trade)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(row, portfolio)
            portfolio.total_value = portfolio_value
            portfolio.equity = portfolio_value

            # Calculate daily return
            if idx > 0:
                prev_value = (
                    equity_values[-1]["value"]
                    if equity_values
                    else config.initial_capital
                )
                daily_return = (portfolio_value - prev_value) / prev_value
                daily_returns.append(daily_return)

            equity_values.append({"date": row["trade_date"], "value": portfolio_value})

        # Create equity curve
        equity_curve = pd.Series(
            [ev["value"] for ev in equity_values],
            index=pd.to_datetime([ev["date"] for ev in equity_values]),
        )

        # Calculate performance metrics
        metrics = self._calculate_comprehensive_metrics(
            equity_curve, benchmark_data, trades, config, daily_returns
        )

        return BacktestResult(
            strategy_name=config.strategy_name,
            config=config,
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            volatility=metrics["volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            max_drawdown=metrics["max_drawdown"],
            max_drawdown_duration=metrics["max_drawdown_duration"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            total_trades=len(trades),
            avg_trade_return=metrics["avg_trade_return"],
            best_trade=metrics["best_trade"],
            worst_trade=metrics["worst_trade"],
            benchmark_return=metrics["benchmark_return"],
            alpha=metrics["alpha"],
            beta=metrics["beta"],
            information_ratio=metrics["information_ratio"],
            tracking_error=metrics["tracking_error"],
            equity_curve=equity_curve,
            monthly_returns=metrics["monthly_returns"],
            trade_log=trades,
            risk_metrics=metrics["risk_metrics"],
            performance_attribution=metrics["performance_attribution"],
        )

    async def _execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        portfolio: Portfolio,
        config: BacktestConfig,
    ) -> Optional[Dict]:
        """Execute an order with realistic transaction costs and slippage."""

        current_price = market_data["close_price"]

        # Apply slippage
        if order.side == OrderSide.BUY:
            execution_price = current_price * (1 + config.slippage)
        else:
            execution_price = current_price * (1 - config.slippage)

        # Calculate commission
        commission = max(
            order.quantity * execution_price * config.transaction_cost,
            config.minimum_commission,
        )

        # Check if order can be executed
        if order.side == OrderSide.BUY:
            total_cost = order.quantity * execution_price + commission
            if total_cost > portfolio.cash:
                order.status = OrderStatus.REJECTED
                return None
        else:
            # Check if we have the position to sell
            if order.symbol not in portfolio.positions:
                order.status = OrderStatus.REJECTED
                return None

            position = portfolio.positions[order.symbol]
            if position.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return None

        # Execute the order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        order.slippage = abs(execution_price - current_price)

        # Update portfolio
        if order.side == OrderSide.BUY:
            # Buy order
            total_cost = order.quantity * execution_price + commission
            portfolio.cash -= total_cost

            if order.symbol in portfolio.positions:
                # Add to existing position
                position = portfolio.positions[order.symbol]
                total_quantity = position.quantity + order.quantity
                total_cost_basis = (
                    position.quantity * position.avg_price
                    + order.quantity * execution_price
                )
                position.avg_price = total_cost_basis / total_quantity
                position.quantity = total_quantity
            else:
                # Create new position
                portfolio.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=execution_price,
                )

        else:
            # Sell order
            position = portfolio.positions[order.symbol]
            proceeds = order.quantity * execution_price - commission
            portfolio.cash += proceeds

            # Calculate realized P&L
            realized_pnl = (
                execution_price - position.avg_price
            ) * order.quantity - commission
            position.realized_pnl += realized_pnl

            # Update position
            position.quantity -= order.quantity
            if position.quantity <= 0:
                del portfolio.positions[order.symbol]

        # Create trade record
        trade = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "commission": order.commission,
            "slippage": order.slippage,
            "timestamp": order.timestamp,
            "pnl": realized_pnl if order.side == OrderSide.SELL else 0.0,
        }

        return trade

    def _calculate_portfolio_value(
        self, market_data: pd.Series, portfolio: Portfolio
    ) -> float:
        """Calculate current portfolio value."""
        total_value = portfolio.cash

        # Add value of positions
        for symbol, position in portfolio.positions.items():
            if symbol == market_data["stock_code"]:
                market_value = position.quantity * market_data["close_price"]
                position.market_value = market_value
                position.unrealized_pnl = market_value - (
                    position.quantity * position.avg_price
                )
                total_value += market_value
            else:
                # For simplicity, assume other positions maintain their value
                # In a real implementation, you'd fetch current prices for all positions
                total_value += position.market_value

        return total_value

    def _calculate_comprehensive_metrics(
        self,
        equity_curve: pd.Series,
        benchmark_data: pd.DataFrame,
        trades: List[Dict],
        config: BacktestConfig,
        daily_returns: List[float],
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        if len(equity_curve) < 2:
            return self._get_empty_metrics()

        # Basic returns
        total_return = (
            equity_curve.iloc[-1] - equity_curve.iloc[0]
        ) / equity_curve.iloc[0]

        # Calculate daily returns if not provided
        if not daily_returns:
            daily_returns = equity_curve.pct_change().dropna().tolist()

        daily_returns = np.array(daily_returns)

        # Annualized metrics
        trading_days = 252
        annual_return = (1 + total_return) ** (trading_days / len(equity_curve)) - 1
        volatility = np.std(daily_returns) * np.sqrt(trading_days)

        # Risk-adjusted metrics
        excess_returns = daily_returns - (config.risk_free_rate / trading_days)
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(trading_days)
            if np.std(daily_returns) > 0
            else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = (
            np.std(downside_returns) * np.sqrt(trading_days)
            if len(downside_returns) > 0
            else 0
        )
        sortino_ratio = (
            (annual_return - config.risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0
        )

        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_duration = self._calculate_max_drawdown_duration(drawdown)

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)

        # Benchmark comparison
        benchmark_metrics = self._calculate_benchmark_metrics(
            daily_returns, benchmark_data, config
        )

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(daily_returns, equity_curve)

        # Performance attribution
        performance_attribution = self._calculate_performance_attribution(trades)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": drawdown_duration,
            "win_rate": trade_stats["win_rate"],
            "profit_factor": trade_stats["profit_factor"],
            "avg_trade_return": trade_stats["avg_return"],
            "best_trade": trade_stats["best_trade"],
            "worst_trade": trade_stats["worst_trade"],
            "benchmark_return": benchmark_metrics["return"],
            "alpha": benchmark_metrics["alpha"],
            "beta": benchmark_metrics["beta"],
            "information_ratio": benchmark_metrics["information_ratio"],
            "tracking_error": benchmark_metrics["tracking_error"],
            "monthly_returns": monthly_returns,
            "risk_metrics": risk_metrics,
            "performance_attribution": performance_attribution,
        }

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)

        return max(drawdown_periods) if drawdown_periods else 0

    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trade-level statistics."""
        if not trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_return": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        # Filter sell trades (which have P&L)
        sell_trades = [
            trade for trade in trades if trade["side"] == "sell" and "pnl" in trade
        ]

        if not sell_trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_return": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        pnls = [trade["pnl"] for trade in sell_trades]

        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]

        win_rate = len(winning_trades) / len(pnls) if pnls else 0

        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf") if gross_profit > 0 else 0
        )

        avg_return = np.mean(pnls) if pnls else 0
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_return": avg_return,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        }

    def _calculate_benchmark_metrics(
        self,
        strategy_returns: np.ndarray,
        benchmark_data: pd.DataFrame,
        config: BacktestConfig,
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""

        # For simplicity, assume benchmark returns are 0.05% daily
        # In a real implementation, you'd calculate actual benchmark returns
        benchmark_returns = np.full(len(strategy_returns), 0.0005)
        benchmark_annual_return = 0.08  # 8% annual return

        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return {
                "return": benchmark_annual_return,
                "alpha": 0.0,
                "beta": 1.0,
                "information_ratio": 0.0,
                "tracking_error": 0.0,
            }

        # Align lengths
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        # Calculate beta
        if np.var(benchmark_returns) > 0:
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(
                benchmark_returns
            )
        else:
            beta = 1.0

        # Calculate alpha
        strategy_annual_return = np.mean(strategy_returns) * 252
        alpha = strategy_annual_return - (
            config.risk_free_rate
            + beta * (benchmark_annual_return - config.risk_free_rate)
        )

        # Tracking error and information ratio
        active_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = (
            np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
            if np.std(active_returns) > 0
            else 0
        )

        return {
            "return": benchmark_annual_return,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
        }

    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns."""
        try:
            monthly_equity = equity_curve.resample("M").last()
            monthly_returns = monthly_equity.pct_change().dropna()
            return monthly_returns
        except Exception:
            # Return empty series if resampling fails
            return pd.Series(dtype=float)

    def _calculate_risk_metrics(
        self, daily_returns: np.ndarray, equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate additional risk metrics."""

        if len(daily_returns) == 0:
            return {}

        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95])
        cvar_99 = np.mean(daily_returns[daily_returns <= var_99])

        # Skewness and Kurtosis
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def _calculate_performance_attribution(
        self, trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate performance attribution by symbol."""
        attribution = {}

        for trade in trades:
            if trade["side"] == "sell" and "pnl" in trade:
                symbol = trade["symbol"]
                if symbol not in attribution:
                    attribution[symbol] = 0.0
                attribution[symbol] += trade["pnl"]

        return attribution

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for edge cases."""
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "benchmark_return": 0.0,
            "alpha": 0.0,
            "beta": 1.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
            "monthly_returns": pd.Series(dtype=float),
            "risk_metrics": {},
            "performance_attribution": {},
        }

    async def _run_walk_forward_analysis(
        self, strategy: BaseStrategy, config: BacktestConfig, stock_data: pd.DataFrame
    ) -> List[BacktestResult]:
        """Run walk-forward analysis to test strategy robustness."""

        if len(stock_data) < 100:  # Need minimum data for walk-forward
            logger.warning("Insufficient data for walk-forward analysis")
            return []

        # Use TimeSeriesSplit for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        wf_results = []

        try:
            for train_index, test_index in tscv.split(stock_data):
                # Create train and test datasets
                train_data = stock_data.iloc[train_index].copy()
                test_data = stock_data.iloc[test_index].copy()

                if len(test_data) < 10:  # Skip if test set too small
                    continue

                # Create test configuration
                start_date = test_data["trade_date"].min()
                end_date = test_data["trade_date"].max()

                # Convert to date if they are datetime objects
                if hasattr(start_date, "date"):
                    start_date = start_date.date()
                if hasattr(end_date, "date"):
                    end_date = end_date.date()

                test_config = BacktestConfig(
                    strategy_name=f"{config.strategy_name}_WF",
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=config.initial_capital,
                    transaction_cost=config.transaction_cost,
                    slippage=config.slippage,
                    benchmark=config.benchmark,
                    strategy_params=config.strategy_params.copy(),
                )

                # Fetch benchmark data for test period
                benchmark_data = await self._fetch_benchmark_data(test_config)

                # Run backtest on test data
                wf_result = await self._run_single_backtest(
                    strategy, test_config, test_data, benchmark_data
                )
                wf_results.append(wf_result)

        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            return []

        return wf_results

    def _calculate_stability_metrics(
        self, main_result: BacktestResult, wf_results: List[BacktestResult]
    ) -> Dict[str, float]:
        """Calculate stability metrics to assess overfitting risk."""

        if not wf_results:
            logger.warning("No walk-forward results for stability analysis")
            return {
                "return_stability": 0.0,
                "sharpe_stability": 0.0,
                "performance_degradation": 0.0,
                "overfitting_risk": 1.0,  # High risk if no validation
                "wf_mean_return": 0.0,
                "wf_std_return": 0.0,
                "consistency_score": 0.0,
                "robustness_score": 0.0,
            }

        if len(wf_results) < 2:
            logger.warning("Insufficient walk-forward results for stability analysis")
            # With only one result, we can still calculate some basic metrics
            single_result = wf_results[0]
            return {
                "return_stability": 1.0,  # Perfect stability with one result
                "sharpe_stability": 1.0,
                "performance_degradation": (
                    (main_result.annual_return - single_result.annual_return)
                    / abs(main_result.annual_return)
                    if abs(main_result.annual_return) > 1e-6
                    else 0.0
                ),
                "overfitting_risk": 0.5,  # Medium risk with insufficient validation
                "wf_mean_return": single_result.annual_return,
                "wf_std_return": 0.0,
                "consistency_score": 1.0 if single_result.annual_return > 0 else 0.0,
                "robustness_score": 0.5,
            }

        # Extract key metrics from walk-forward results
        wf_returns = [
            result.annual_return
            for result in wf_results
            if not np.isnan(result.annual_return)
        ]
        wf_sharpe = [
            result.sharpe_ratio
            for result in wf_results
            if not np.isnan(result.sharpe_ratio)
        ]
        wf_max_dd = [
            result.max_drawdown
            for result in wf_results
            if not np.isnan(result.max_drawdown)
        ]

        if not wf_returns:
            logger.warning("No valid walk-forward returns for stability analysis")
            return {
                "return_stability": 0.0,
                "sharpe_stability": 0.0,
                "performance_degradation": 0.0,
                "overfitting_risk": 1.0,
                "wf_mean_return": 0.0,
                "wf_std_return": 0.0,
                "consistency_score": 0.0,
                "robustness_score": 0.0,
            }

        # Calculate stability metrics
        wf_mean_return = np.mean(wf_returns)
        wf_std_return = np.std(wf_returns)

        # Return stability: 1 - coefficient of variation (capped at 1.0)
        return_stability = (
            1.0 - (wf_std_return / abs(wf_mean_return))
            if abs(wf_mean_return) > 1e-6
            else 0.0
        )
        return_stability = max(0.0, min(1.0, return_stability))

        # Sharpe stability
        if len(wf_sharpe) > 1:
            wf_mean_sharpe = np.mean(wf_sharpe)
            wf_std_sharpe = np.std(wf_sharpe)
            sharpe_stability = (
                1.0 - (wf_std_sharpe / abs(wf_mean_sharpe))
                if abs(wf_mean_sharpe) > 1e-6
                else 0.0
            )
            sharpe_stability = max(0.0, min(1.0, sharpe_stability))
        else:
            sharpe_stability = 0.0

        # Performance degradation (main vs walk-forward average)
        if abs(main_result.annual_return) > 1e-6:
            performance_degradation = (
                main_result.annual_return - wf_mean_return
            ) / abs(main_result.annual_return)
        else:
            performance_degradation = 0.0

        # Overfitting risk assessment
        # High degradation (>20%) indicates potential overfitting
        overfitting_risk = (
            max(0.0, min(1.0, performance_degradation))
            if performance_degradation > 0.2
            else 0.0
        )

        # Consistency score: percentage of periods with positive returns
        positive_periods = sum(1 for ret in wf_returns if ret > 0)
        consistency_score = positive_periods / len(wf_returns) if wf_returns else 0.0

        # Robustness score: combination of stability and consistency
        robustness_score = (
            return_stability * 0.4
            + sharpe_stability * 0.3
            + consistency_score * 0.2
            + (1 - overfitting_risk) * 0.1
        )

        # Additional metrics for comprehensive analysis
        metrics = {
            "return_stability": return_stability,
            "sharpe_stability": sharpe_stability,
            "performance_degradation": performance_degradation,
            "overfitting_risk": overfitting_risk,
            "wf_mean_return": wf_mean_return,
            "wf_std_return": wf_std_return,
            "consistency_score": consistency_score,
            "robustness_score": robustness_score,
        }

        # Add drawdown stability if available
        if len(wf_max_dd) > 1:
            wf_mean_dd = np.mean(wf_max_dd)
            wf_std_dd = np.std(wf_max_dd)
            dd_stability = (
                1.0 - (wf_std_dd / abs(wf_mean_dd)) if abs(wf_mean_dd) > 1e-6 else 0.0
            )
            metrics["drawdown_stability"] = max(0.0, min(1.0, dd_stability))

        return metrics

    async def run_parameter_optimization(
        self,
        strategy_class,
        param_grid: Dict[str, List],
        config: BacktestConfig,
        stock_data: pd.DataFrame = None,
        optimization_metric: str = "sharpe_ratio",
        cv_folds: int = 3,
    ) -> Dict[str, Any]:
        """
        Run parameter optimization with walk-forward validation.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameters to optimize
            config: Backtesting configuration
            stock_data: Historical data for optimization
            optimization_metric: Metric to optimize ('sharpe_ratio', 'annual_return', etc.)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing optimization results
        """

        if stock_data is None:
            stock_data = await self._fetch_stock_data(config)

        if len(stock_data) < 100:
            logger.warning("Insufficient data for parameter optimization")
            return {"best_params": {}, "best_score": 0.0, "optimization_results": []}

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        if not param_combinations:
            logger.warning("No parameter combinations to test")
            return {"best_params": {}, "best_score": 0.0, "optimization_results": []}

        optimization_results = []
        best_score = float("-inf")
        best_params = {}

        # Use TimeSeriesSplit for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        logger.info(
            f"Starting parameter optimization with {len(param_combinations)} combinations"
        )

        for i, params in enumerate(param_combinations):
            try:
                # Create strategy with current parameters
                strategy = strategy_class(params)

                # Run cross-validation
                cv_scores = []

                for train_index, test_index in tscv.split(stock_data):
                    # Create train and test datasets
                    test_data = stock_data.iloc[test_index].copy()

                    if len(test_data) < 10:  # Skip if test set too small
                        continue

                    # Create test configuration
                    start_date = test_data["trade_date"].min()
                    end_date = test_data["trade_date"].max()

                    # Convert to date if they are datetime objects
                    if hasattr(start_date, "date"):
                        start_date = start_date.date()
                    if hasattr(end_date, "date"):
                        end_date = end_date.date()

                    test_config = BacktestConfig(
                        strategy_name=f"{config.strategy_name}_opt_{i}",
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=config.initial_capital,
                        transaction_cost=config.transaction_cost,
                        slippage=config.slippage,
                        benchmark=config.benchmark,
                        strategy_params=params.copy(),
                    )

                    # Fetch benchmark data for test period
                    benchmark_data = await self._fetch_benchmark_data(test_config)

                    # Run backtest on test data
                    result = await self._run_single_backtest(
                        strategy, test_config, test_data, benchmark_data
                    )

                    # Extract optimization metric
                    score = getattr(result, optimization_metric, 0.0)
                    if not np.isnan(score) and np.isfinite(score):
                        cv_scores.append(score)

                # Calculate average score across folds
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)

                    optimization_results.append(
                        {
                            "params": params,
                            "mean_score": avg_score,
                            "std_score": std_score,
                            "cv_scores": cv_scores,
                        }
                    )

                    # Update best parameters
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params.copy()

                logger.info(f"Completed optimization {i+1}/{len(param_combinations)}")

            except Exception as e:
                logger.error(
                    f"Error in parameter optimization for params {params}: {e}"
                )
                continue

        # Sort results by score
        optimization_results.sort(key=lambda x: x["mean_score"], reverse=True)

        logger.info(f"Parameter optimization completed. Best score: {best_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_results": optimization_results,
            "optimization_metric": optimization_metric,
        }

    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters from grid."""

        if not param_grid:
            return []

        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate all combinations
        combinations = []

        def generate_combinations(current_params, remaining_names, remaining_values):
            if not remaining_names:
                combinations.append(current_params.copy())
                return

            param_name = remaining_names[0]
            param_vals = remaining_values[0]

            for val in param_vals:
                current_params[param_name] = val
                generate_combinations(
                    current_params, remaining_names[1:], remaining_values[1:]
                )

        generate_combinations({}, param_names, param_values)

        return combinations

    def assess_overfitting_risk(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Assess overfitting risk based on stability metrics and other indicators.

        Args:
            result: Backtesting result with stability metrics

        Returns:
            Dictionary containing overfitting risk assessment
        """

        risk_factors = []
        risk_score = 0.0
        warnings = []
        recommendations = []

        # Check if stability metrics are available
        if "return_stability" not in result.risk_metrics:
            warnings.append(
                "No walk-forward analysis performed - high overfitting risk"
            )
            risk_score += 0.5
            risk_factors.append("No validation performed")
        else:
            # Analyze stability metrics
            return_stability = result.risk_metrics.get("return_stability", 0.0)
            sharpe_stability = result.risk_metrics.get("sharpe_stability", 0.0)
            performance_degradation = result.risk_metrics.get(
                "performance_degradation", 0.0
            )
            consistency_score = result.risk_metrics.get("consistency_score", 0.0)

            # Return stability risk
            if return_stability < 0.3:
                risk_factors.append("Low return stability")
                risk_score += 0.3
                warnings.append(f"Return stability is low ({return_stability:.2f})")
                recommendations.append(
                    "Consider simplifying strategy or using more robust parameters"
                )

            # Sharpe stability risk
            if sharpe_stability < 0.3:
                risk_factors.append("Low Sharpe ratio stability")
                risk_score += 0.2
                warnings.append(
                    f"Sharpe ratio stability is low ({sharpe_stability:.2f})"
                )

            # Performance degradation risk
            if performance_degradation > 0.2:
                risk_factors.append("High performance degradation")
                risk_score += 0.4
                warnings.append(
                    f"Performance degradation is high ({performance_degradation:.2%})"
                )
                recommendations.append("Strategy may be overfitted to historical data")

            # Consistency risk
            if consistency_score < 0.4:
                risk_factors.append("Low consistency across periods")
                risk_score += 0.2
                warnings.append(
                    f"Strategy is inconsistent across periods ({consistency_score:.2%})"
                )

        # Check for other overfitting indicators

        # Extremely high Sharpe ratio (>3.0) can indicate overfitting
        if result.sharpe_ratio > 3.0:
            risk_factors.append("Unusually high Sharpe ratio")
            risk_score += 0.3
            warnings.append(
                f"Sharpe ratio is unusually high ({result.sharpe_ratio:.2f})"
            )
            recommendations.append("Verify strategy logic and data quality")

        # Very low number of trades might indicate curve fitting
        if result.total_trades < 10:
            risk_factors.append("Very few trades")
            risk_score += 0.2
            warnings.append(f"Very few trades ({result.total_trades})")
            recommendations.append(
                "Ensure strategy generates sufficient trading opportunities"
            )

        # Perfect win rate (100%) is suspicious
        if result.win_rate >= 0.99:
            risk_factors.append("Perfect or near-perfect win rate")
            risk_score += 0.4
            warnings.append(f"Win rate is suspiciously high ({result.win_rate:.2%})")
            recommendations.append("Check for look-ahead bias or data snooping")

        # Cap risk score at 1.0
        risk_score = min(1.0, risk_score)

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "warnings": warnings,
            "recommendations": recommendations,
            "assessment_summary": self._generate_risk_assessment_summary(
                risk_score, risk_level, risk_factors
            ),
        }

    def _generate_risk_assessment_summary(
        self, risk_score: float, risk_level: str, risk_factors: List[str]
    ) -> str:
        """Generate a human-readable risk assessment summary."""

        summary = (
            f"Overfitting Risk Assessment: {risk_level} (Score: {risk_score:.2f})\n\n"
        )

        if risk_level == "LOW":
            summary += "✅ The strategy shows good stability and low overfitting risk. "
            summary += "The walk-forward analysis indicates consistent performance across different time periods."
        elif risk_level == "MEDIUM":
            summary += "⚠️ The strategy shows moderate overfitting risk. "
            summary += (
                "Some stability concerns were identified that should be addressed."
            )
        else:
            summary += "🚨 The strategy shows high overfitting risk. "
            summary += "Significant stability issues were detected that require immediate attention."

        if risk_factors:
            summary += f"\n\nKey Risk Factors:\n"
            for factor in risk_factors:
                summary += f"• {factor}\n"

        return summary

    async def _fetch_stock_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Fetch stock data for backtesting."""

        if self.data_source is None:
            # Generate sample data for testing
            logger.warning("No data source provided, generating sample data")
            return self._generate_sample_data(config)

        try:
            # In a real implementation, you'd extract symbol from config or pass it separately
            # For now, use a default symbol
            symbol = "000001.SZ"  # Default to Ping An Bank

            data = await self.data_source.get_stock_data(
                symbol=symbol, start_date=config.start_date, end_date=config.end_date
            )

            return data

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return self._generate_sample_data(config)

    async def _fetch_benchmark_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Fetch benchmark data for comparison."""

        # Check cache first
        cache_key = f"{config.benchmark}_{config.start_date}_{config.end_date}"
        if cache_key in self.benchmark_cache:
            return self.benchmark_cache[cache_key]

        try:
            if self.data_source is not None:
                data = await self.data_source.get_stock_data(
                    symbol=config.benchmark,
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
                self.benchmark_cache[cache_key] = data
                return data
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")

        # Generate sample benchmark data
        benchmark_data = self._generate_sample_benchmark_data(config)
        self.benchmark_cache[cache_key] = benchmark_data
        return benchmark_data

    def _generate_sample_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Generate sample stock data for testing."""

        date_range = pd.date_range(
            start=config.start_date, end=config.end_date, freq="D"
        )

        # Filter business days
        business_days = [d for d in date_range if d.weekday() < 5]

        np.random.seed(42)  # For reproducible results

        # Generate realistic stock price data
        n_days = len(business_days)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with drift

        prices = [100.0]  # Starting price
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLC data
        data = []
        for i, (date, price) in enumerate(zip(business_days, prices)):
            # Generate intraday volatility
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = (
                prices[i - 1] * (1 + np.random.normal(0, 0.005)) if i > 0 else price
            )

            # Ensure OHLC consistency
            high = max(high, price, open_price)
            low = min(low, price, open_price)

            volume = int(np.random.lognormal(15, 1))  # Log-normal volume distribution
            amount = volume * price

            data.append(
                {
                    "stock_code": "000001.SZ",
                    "trade_date": date.date(),
                    "open_price": round(open_price, 2),
                    "high_price": round(high, 2),
                    "low_price": round(low, 2),
                    "close_price": round(price, 2),
                    "volume": volume,
                    "amount": round(amount, 2),
                    "adj_factor": 1.0,
                }
            )

        return pd.DataFrame(data)

    def _generate_sample_benchmark_data(self, config: BacktestConfig) -> pd.DataFrame:
        """Generate sample benchmark data."""

        date_range = pd.date_range(
            start=config.start_date, end=config.end_date, freq="D"
        )

        # Filter business days
        business_days = [d for d in date_range if d.weekday() < 5]

        np.random.seed(123)  # Different seed for benchmark

        # Generate benchmark returns (slightly lower volatility)
        n_days = len(business_days)
        returns = np.random.normal(0.0003, 0.015, n_days)  # Lower drift and volatility

        prices = [3000.0]  # Starting index value
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = []
        for date, price in zip(business_days, prices):
            data.append(
                {
                    "stock_code": config.benchmark,
                    "trade_date": date.date(),
                    "close_price": round(price, 2),
                }
            )

        return pd.DataFrame(data)

    async def run_multiple_benchmarks(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        benchmarks: List[str],
        stock_data: pd.DataFrame = None,
    ) -> Dict[str, BacktestResult]:
        """Run backtest against multiple benchmarks."""

        results = {}

        for benchmark in benchmarks:
            benchmark_config = BacktestConfig(
                strategy_name=config.strategy_name,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                transaction_cost=config.transaction_cost,
                slippage=config.slippage,
                benchmark=benchmark,
                rebalance_frequency=config.rebalance_frequency,
                max_position_size=config.max_position_size,
                risk_free_rate=config.risk_free_rate,
                strategy_params=config.strategy_params.copy(),
            )

            try:
                result = await self.run_comprehensive_backtest(
                    strategy, benchmark_config, stock_data
                )
                results[benchmark] = result
            except Exception as e:
                logger.error(f"Error running backtest for benchmark {benchmark}: {e}")
                continue

        return results

    def generate_performance_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive performance report."""

        report = f"""
=== BACKTESTING PERFORMANCE REPORT ===

Strategy: {result.strategy_name}
Period: {result.config.start_date} to {result.config.end_date}
Initial Capital: ${result.config.initial_capital:,.2f}

=== RETURNS ===
Total Return: {result.total_return:.2%}
Annualized Return: {result.annual_return:.2%}
Benchmark Return: {result.benchmark_return:.2%}
Alpha: {result.alpha:.2%}

=== RISK METRICS ===
Volatility: {result.volatility:.2%}
Maximum Drawdown: {result.max_drawdown:.2%}
Max Drawdown Duration: {result.max_drawdown_duration} days
Beta: {result.beta:.2f}
Tracking Error: {result.tracking_error:.2%}

=== RISK-ADJUSTED RETURNS ===
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Calmar Ratio: {result.calmar_ratio:.2f}
Information Ratio: {result.information_ratio:.2f}

=== TRADING STATISTICS ===
Total Trades: {result.total_trades}
Win Rate: {result.win_rate:.2%}
Profit Factor: {result.profit_factor:.2f}
Average Trade Return: ${result.avg_trade_return:.2f}
Best Trade: ${result.best_trade:.2f}
Worst Trade: ${result.worst_trade:.2f}

=== STABILITY METRICS ===
"""

        if "return_stability" in result.risk_metrics:
            report += f"""Return Stability: {result.risk_metrics['return_stability']:.2f}
Sharpe Stability: {result.risk_metrics['sharpe_stability']:.2f}
Performance Degradation: {result.risk_metrics['performance_degradation']:.2%}
Overfitting Risk: {result.risk_metrics['overfitting_risk']:.2f}
"""

        if "var_95" in result.risk_metrics:
            report += f"""
=== RISK METRICS ===
VaR (95%): {result.risk_metrics['var_95']:.2%}
VaR (99%): {result.risk_metrics['var_99']:.2%}
CVaR (95%): {result.risk_metrics['cvar_95']:.2%}
CVaR (99%): {result.risk_metrics['cvar_99']:.2%}
Skewness: {result.risk_metrics['skewness']:.2f}
Kurtosis: {result.risk_metrics['kurtosis']:.2f}
"""

        return report

    def generate_walk_forward_report(
        self, main_result: BacktestResult, wf_results: List[BacktestResult]
    ) -> str:
        """Generate a detailed walk-forward analysis report."""

        if not wf_results:
            return "Walk-forward analysis was not performed or failed."

        report = f"""
=== WALK-FORWARD ANALYSIS REPORT ===

Strategy: {main_result.strategy_name}
Analysis Period: {main_result.config.start_date} to {main_result.config.end_date}
Walk-Forward Periods: {len(wf_results)}

=== MAIN BACKTEST RESULTS ===
Total Return: {main_result.total_return:.2%}
Annualized Return: {main_result.annual_return:.2%}
Sharpe Ratio: {main_result.sharpe_ratio:.2f}
Maximum Drawdown: {main_result.max_drawdown:.2%}

=== WALK-FORWARD VALIDATION RESULTS ===
"""

        # Calculate walk-forward statistics
        wf_returns = [
            r.annual_return for r in wf_results if not np.isnan(r.annual_return)
        ]
        wf_sharpe = [r.sharpe_ratio for r in wf_results if not np.isnan(r.sharpe_ratio)]
        wf_drawdowns = [
            r.max_drawdown for r in wf_results if not np.isnan(r.max_drawdown)
        ]

        if wf_returns:
            report += f"""
Average Return: {np.mean(wf_returns):.2%}
Return Std Dev: {np.std(wf_returns):.2%}
Best Period: {max(wf_returns):.2%}
Worst Period: {min(wf_returns):.2%}
Positive Periods: {sum(1 for r in wf_returns if r > 0)}/{len(wf_returns)}
"""

        if wf_sharpe:
            report += f"""
Average Sharpe: {np.mean(wf_sharpe):.2f}
Sharpe Std Dev: {np.std(wf_sharpe):.2f}
Best Sharpe: {max(wf_sharpe):.2f}
Worst Sharpe: {min(wf_sharpe):.2f}
"""

        if wf_drawdowns:
            report += f"""
Average Max DD: {np.mean(wf_drawdowns):.2%}
DD Std Dev: {np.std(wf_drawdowns):.2%}
Best DD: {max(wf_drawdowns):.2%}
Worst DD: {min(wf_drawdowns):.2%}
"""

        # Add stability metrics if available
        if "return_stability" in main_result.risk_metrics:
            report += f"""
=== STABILITY METRICS ===
Return Stability: {main_result.risk_metrics['return_stability']:.2f}
Sharpe Stability: {main_result.risk_metrics['sharpe_stability']:.2f}
Performance Degradation: {main_result.risk_metrics['performance_degradation']:.2%}
Consistency Score: {main_result.risk_metrics['consistency_score']:.2%}
Robustness Score: {main_result.risk_metrics['robustness_score']:.2f}
Overfitting Risk: {main_result.risk_metrics['overfitting_risk']:.2f}
"""

        # Add period-by-period breakdown
        report += f"""
=== PERIOD-BY-PERIOD BREAKDOWN ===
"""
        for i, result in enumerate(wf_results):
            report += f"""
Period {i+1}: {result.config.start_date} to {result.config.end_date}
  Return: {result.annual_return:.2%}
  Sharpe: {result.sharpe_ratio:.2f}
  Max DD: {result.max_drawdown:.2%}
  Trades: {result.total_trades}
"""

        return report

    async def run_comprehensive_validation(
        self,
        strategy_class,
        config: BacktestConfig,
        stock_data: pd.DataFrame = None,
        param_grid: Dict[str, List] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive strategy validation including walk-forward analysis,
        parameter optimization, and overfitting risk assessment.

        Args:
            strategy_class: Strategy class to validate
            config: Backtesting configuration
            stock_data: Historical data for validation
            param_grid: Optional parameter grid for optimization

        Returns:
            Dictionary containing comprehensive validation results
        """

        validation_results = {
            "main_backtest": None,
            "walk_forward_results": [],
            "parameter_optimization": None,
            "overfitting_assessment": None,
            "validation_summary": None,
        }

        try:
            # Fetch data if not provided
            if stock_data is None:
                stock_data = await self._fetch_stock_data(config)

            # Create strategy instance
            strategy = strategy_class(config.strategy_params)

            # Run main backtest
            logger.info("Running main backtest...")
            main_result = await self.run_comprehensive_backtest(
                strategy, config, stock_data
            )
            validation_results["main_backtest"] = main_result

            # Run parameter optimization if param_grid provided
            if param_grid:
                logger.info("Running parameter optimization...")
                opt_results = await self.run_parameter_optimization(
                    strategy_class, param_grid, config, stock_data
                )
                validation_results["parameter_optimization"] = opt_results

                # Re-run backtest with optimized parameters if better ones found
                if (
                    opt_results["best_params"]
                    and opt_results["best_score"] > main_result.sharpe_ratio
                ):
                    logger.info("Re-running backtest with optimized parameters...")
                    optimized_config = BacktestConfig(
                        strategy_name=f"{config.strategy_name}_optimized",
                        start_date=config.start_date,
                        end_date=config.end_date,
                        initial_capital=config.initial_capital,
                        transaction_cost=config.transaction_cost,
                        slippage=config.slippage,
                        benchmark=config.benchmark,
                        strategy_params=opt_results["best_params"],
                    )

                    optimized_strategy = strategy_class(opt_results["best_params"])
                    optimized_result = await self.run_comprehensive_backtest(
                        optimized_strategy, optimized_config, stock_data
                    )
                    validation_results["optimized_backtest"] = optimized_result

            # Assess overfitting risk
            logger.info("Assessing overfitting risk...")
            overfitting_assessment = self.assess_overfitting_risk(main_result)
            validation_results["overfitting_assessment"] = overfitting_assessment

            # Generate validation summary
            validation_results["validation_summary"] = (
                self._generate_validation_summary(validation_results)
            )

            logger.info("Comprehensive validation completed successfully")

        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            validation_results["error"] = str(e)

        return validation_results

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation summary."""

        summary = "=== COMPREHENSIVE STRATEGY VALIDATION SUMMARY ===\n\n"

        main_result = validation_results.get("main_backtest")
        if main_result:
            summary += f"Strategy: {main_result.strategy_name}\n"
            summary += f"Period: {main_result.config.start_date} to {main_result.config.end_date}\n\n"

            summary += "📊 MAIN BACKTEST PERFORMANCE:\n"
            summary += f"• Total Return: {main_result.total_return:.2%}\n"
            summary += f"• Annualized Return: {main_result.annual_return:.2%}\n"
            summary += f"• Sharpe Ratio: {main_result.sharpe_ratio:.2f}\n"
            summary += f"• Maximum Drawdown: {main_result.max_drawdown:.2%}\n"
            summary += f"• Total Trades: {main_result.total_trades}\n\n"

        # Parameter optimization results
        opt_results = validation_results.get("parameter_optimization")
        if opt_results:
            summary += "🔧 PARAMETER OPTIMIZATION:\n"
            summary += f"• Best Score: {opt_results['best_score']:.4f}\n"
            summary += f"• Best Parameters: {opt_results['best_params']}\n"
            summary += (
                f"• Tested Combinations: {len(opt_results['optimization_results'])}\n\n"
            )

        # Overfitting assessment
        overfitting = validation_results.get("overfitting_assessment")
        if overfitting:
            summary += "🎯 OVERFITTING RISK ASSESSMENT:\n"
            summary += f"• Risk Level: {overfitting['risk_level']}\n"
            summary += f"• Risk Score: {overfitting['risk_score']:.2f}\n"

            if overfitting["warnings"]:
                summary += "• Warnings:\n"
                for warning in overfitting["warnings"]:
                    summary += f"  - {warning}\n"

            if overfitting["recommendations"]:
                summary += "• Recommendations:\n"
                for rec in overfitting["recommendations"]:
                    summary += f"  - {rec}\n"
            summary += "\n"

        # Overall assessment
        summary += "🏆 OVERALL ASSESSMENT:\n"

        if main_result and overfitting:
            if overfitting["risk_level"] == "LOW" and main_result.sharpe_ratio > 1.0:
                summary += (
                    "✅ Strategy shows strong performance with low overfitting risk.\n"
                )
                summary += (
                    "✅ Recommended for further development and potential deployment.\n"
                )
            elif overfitting["risk_level"] == "MEDIUM":
                summary += "⚠️ Strategy shows decent performance but moderate overfitting risk.\n"
                summary += (
                    "⚠️ Consider addressing stability concerns before deployment.\n"
                )
            else:
                summary += "🚨 Strategy shows high overfitting risk.\n"
                summary += "🚨 Significant improvements needed before considering deployment.\n"

        return summary
