"""Tests for Enhanced Backtesting Engine."""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from stock_analysis_system.analysis.enhanced_backtesting_engine import (
    BacktestConfig,
    BacktestResult,
    EnhancedBacktestingEngine,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    SimpleMovingAverageStrategy,
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    business_days = [d for d in dates if d.weekday() < 5]

    np.random.seed(42)
    n_days = len(business_days)
    returns = np.random.normal(0.0005, 0.02, n_days)

    prices = [100.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = []
    for i, (date, price) in enumerate(zip(business_days, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = (
            prices[i - 1] * (1 + np.random.normal(0, 0.005)) if i > 0 else price
        )

        high = max(high, price, open_price)
        low = min(low, price, open_price)

        volume = int(np.random.lognormal(15, 1))

        data.append(
            {
                "stock_code": "000001.SZ",
                "trade_date": date.date(),
                "open_price": round(open_price, 2),
                "high_price": round(high, 2),
                "low_price": round(low, 2),
                "close_price": round(price, 2),
                "volume": volume,
                "amount": round(volume * price, 2),
                "adj_factor": 1.0,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def backtest_config():
    """Create sample backtest configuration."""
    return BacktestConfig(
        strategy_name="Test Strategy",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1000000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        benchmark="000300.SH",
        strategy_params={"ma_short": 10, "ma_long": 30},
    )


@pytest.fixture
def backtesting_engine():
    """Create backtesting engine instance."""
    return EnhancedBacktestingEngine()


@pytest.fixture
def simple_strategy():
    """Create simple moving average strategy."""
    return SimpleMovingAverageStrategy(
        {"ma_short": 10, "ma_long": 30, "position_size": 0.1}
    )


class TestBacktestConfig:
    """Test BacktestConfig class."""

    def test_config_creation(self, backtest_config):
        """Test configuration creation."""
        assert backtest_config.strategy_name == "Test Strategy"
        assert backtest_config.initial_capital == 1000000.0
        assert backtest_config.transaction_cost == 0.001
        assert backtest_config.slippage == 0.0005

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = BacktestConfig(
            strategy_name="Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )

        assert config.initial_capital == 1000000.0
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.benchmark == "000300.SH"
        assert config.max_position_size == 0.1


class TestOrder:
    """Test Order class."""

    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            order_id="test_001",
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        assert order.order_id == "test_001"
        assert order.symbol == "000001.SZ"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 1000
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0


class TestPosition:
    """Test Position class."""

    def test_position_creation(self):
        """Test position creation."""
        position = Position(symbol="000001.SZ", quantity=1000, avg_price=100.0)

        assert position.symbol == "000001.SZ"
        assert position.quantity == 1000
        assert position.avg_price == 100.0
        assert position.market_value == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.realized_pnl == 0.0


class TestPortfolio:
    """Test Portfolio class."""

    def test_portfolio_creation(self):
        """Test portfolio creation."""
        portfolio = Portfolio(cash=1000000.0)

        assert portfolio.cash == 1000000.0
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 0.0
        assert portfolio.equity == 0.0


class TestSimpleMovingAverageStrategy:
    """Test SimpleMovingAverageStrategy class."""

    def test_strategy_creation(self, simple_strategy):
        """Test strategy creation."""
        assert simple_strategy.params["ma_short"] == 10
        assert simple_strategy.params["ma_long"] == 30
        assert simple_strategy.params["position_size"] == 0.1

    @pytest.mark.asyncio
    async def test_generate_signals(self, simple_strategy):
        """Test signal generation."""
        portfolio = Portfolio(cash=1000000.0)
        data = pd.Series(
            {
                "stock_code": "000001.SZ",
                "trade_date": date(2023, 1, 1),
                "close_price": 100.0,
            }
        )

        signals = await simple_strategy.generate_signals(data, portfolio)

        assert "buy_signal" in signals
        assert "sell_signal" in signals
        assert isinstance(signals["buy_signal"], bool)
        assert isinstance(signals["sell_signal"], bool)

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, simple_strategy):
        """Test position size calculation."""
        portfolio = Portfolio(cash=1000000.0)
        signal = {"strength": 1.0}
        current_price = 100.0

        size = await simple_strategy.calculate_position_size(
            signal, portfolio, current_price
        )

        expected_size = (1000000.0 * 0.1) / 100.0  # 10% of capital / price
        assert size == expected_size

    @pytest.mark.asyncio
    async def test_on_bar(self, simple_strategy):
        """Test on_bar method."""
        portfolio = Portfolio(cash=1000000.0)
        data = pd.Series(
            {
                "stock_code": "000001.SZ",
                "trade_date": date(2023, 1, 1),
                "close_price": 100.0,
            }
        )

        orders = await simple_strategy.on_bar(data, portfolio)

        assert isinstance(orders, list)
        # Orders may be empty due to random signal generation


class TestEnhancedBacktestingEngine:
    """Test EnhancedBacktestingEngine class."""

    def test_engine_creation(self, backtesting_engine):
        """Test engine creation."""
        assert backtesting_engine.data_source is None
        assert isinstance(backtesting_engine.benchmark_cache, dict)
        assert isinstance(backtesting_engine.performance_cache, dict)

    @pytest.mark.asyncio
    async def test_execute_order_buy(self, backtesting_engine, backtest_config):
        """Test buy order execution."""
        portfolio = Portfolio(cash=1000000.0)
        order = Order(
            order_id="buy_001",
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        market_data = pd.Series({"stock_code": "000001.SZ", "close_price": 100.0})

        trade = await backtesting_engine._execute_order(
            order, market_data, portfolio, backtest_config
        )

        assert trade is not None
        assert trade["side"] == "buy"
        assert trade["quantity"] == 1000
        assert trade["price"] > 100.0  # Should include slippage
        assert order.status == OrderStatus.FILLED
        assert portfolio.cash < 1000000.0  # Cash should decrease
        assert "000001.SZ" in portfolio.positions

    @pytest.mark.asyncio
    async def test_execute_order_sell(self, backtesting_engine, backtest_config):
        """Test sell order execution."""
        portfolio = Portfolio(cash=900000.0)
        portfolio.positions["000001.SZ"] = Position(
            symbol="000001.SZ", quantity=1000, avg_price=100.0
        )

        order = Order(
            order_id="sell_001",
            symbol="000001.SZ",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        market_data = pd.Series({"stock_code": "000001.SZ", "close_price": 110.0})

        trade = await backtesting_engine._execute_order(
            order, market_data, portfolio, backtest_config
        )

        assert trade is not None
        assert trade["side"] == "sell"
        assert trade["quantity"] == 1000
        assert trade["pnl"] > 0  # Should be profitable
        assert order.status == OrderStatus.FILLED
        assert portfolio.cash > 900000.0  # Cash should increase
        assert "000001.SZ" not in portfolio.positions  # Position should be closed

    @pytest.mark.asyncio
    async def test_execute_order_insufficient_cash(
        self, backtesting_engine, backtest_config
    ):
        """Test order rejection due to insufficient cash."""
        portfolio = Portfolio(cash=1000.0)  # Very low cash
        order = Order(
            order_id="buy_001",
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        market_data = pd.Series({"stock_code": "000001.SZ", "close_price": 100.0})

        trade = await backtesting_engine._execute_order(
            order, market_data, portfolio, backtest_config
        )

        assert trade is None
        assert order.status == OrderStatus.REJECTED
        assert portfolio.cash == 1000.0  # Cash unchanged

    def test_calculate_portfolio_value(self, backtesting_engine):
        """Test portfolio value calculation."""
        portfolio = Portfolio(cash=500000.0)
        portfolio.positions["000001.SZ"] = Position(
            symbol="000001.SZ", quantity=1000, avg_price=100.0
        )

        market_data = pd.Series({"stock_code": "000001.SZ", "close_price": 110.0})

        value = backtesting_engine._calculate_portfolio_value(market_data, portfolio)

        expected_value = 500000.0 + (1000 * 110.0)  # Cash + position value
        assert value == expected_value
        assert portfolio.positions["000001.SZ"].market_value == 110000.0
        assert portfolio.positions["000001.SZ"].unrealized_pnl == 10000.0

    def test_calculate_trade_statistics_empty(self, backtesting_engine):
        """Test trade statistics with no trades."""
        stats = backtesting_engine._calculate_trade_statistics([])

        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["avg_return"] == 0.0
        assert stats["best_trade"] == 0.0
        assert stats["worst_trade"] == 0.0

    def test_calculate_trade_statistics_with_trades(self, backtesting_engine):
        """Test trade statistics with sample trades."""
        trades = [
            {"side": "sell", "pnl": 1000.0},
            {"side": "sell", "pnl": -500.0},
            {"side": "sell", "pnl": 2000.0},
            {"side": "sell", "pnl": -300.0},
            {"side": "buy", "pnl": 0.0},  # Buy trades don't have P&L
        ]

        stats = backtesting_engine._calculate_trade_statistics(trades)

        assert stats["win_rate"] == 0.5  # 2 wins out of 4 sell trades
        assert stats["profit_factor"] == 3000.0 / 800.0  # Gross profit / gross loss
        assert stats["avg_return"] == 550.0  # (1000 - 500 + 2000 - 300) / 4
        assert stats["best_trade"] == 2000.0
        assert stats["worst_trade"] == -500.0

    def test_generate_sample_data(self, backtesting_engine, backtest_config):
        """Test sample data generation."""
        data = backtesting_engine._generate_sample_data(backtest_config)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "stock_code" in data.columns
        assert "trade_date" in data.columns
        assert "close_price" in data.columns
        assert "volume" in data.columns

        # Check OHLC consistency
        for _, row in data.iterrows():
            assert row["high_price"] >= row["close_price"]
            assert row["low_price"] <= row["close_price"]
            assert row["high_price"] >= row["open_price"]
            assert row["low_price"] <= row["open_price"]

    def test_generate_sample_benchmark_data(self, backtesting_engine, backtest_config):
        """Test sample benchmark data generation."""
        data = backtesting_engine._generate_sample_benchmark_data(backtest_config)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "stock_code" in data.columns
        assert "trade_date" in data.columns
        assert "close_price" in data.columns
        assert data["stock_code"].iloc[0] == backtest_config.benchmark

    @pytest.mark.asyncio
    async def test_run_comprehensive_backtest(
        self, backtesting_engine, simple_strategy, backtest_config, sample_stock_data
    ):
        """Test comprehensive backtesting."""
        result = await backtesting_engine.run_comprehensive_backtest(
            simple_strategy, backtest_config, sample_stock_data
        )

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == backtest_config.strategy_name
        assert result.config == backtest_config
        assert isinstance(result.total_return, float)
        assert isinstance(result.annual_return, float)
        assert isinstance(result.volatility, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.trade_log, list)
        assert isinstance(result.risk_metrics, dict)

    @pytest.mark.asyncio
    async def test_run_multiple_benchmarks(
        self, backtesting_engine, simple_strategy, backtest_config, sample_stock_data
    ):
        """Test running backtest against multiple benchmarks."""
        benchmarks = ["000300.SH", "000905.SH"]

        results = await backtesting_engine.run_multiple_benchmarks(
            simple_strategy, backtest_config, benchmarks, sample_stock_data
        )

        assert isinstance(results, dict)
        assert len(results) == len(benchmarks)

        for benchmark in benchmarks:
            assert benchmark in results
            assert isinstance(results[benchmark], BacktestResult)

    def test_generate_performance_report(self, backtesting_engine):
        """Test performance report generation."""
        # Create a mock result
        config = BacktestConfig(
            strategy_name="Test Strategy",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )

        result = BacktestResult(
            strategy_name="Test Strategy",
            config=config,
            total_return=0.15,
            annual_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=0.85,
            calmar_ratio=0.75,
            max_drawdown=-0.08,
            max_drawdown_duration=45,
            win_rate=0.55,
            profit_factor=1.25,
            total_trades=100,
            avg_trade_return=150.0,
            best_trade=5000.0,
            worst_trade=-2000.0,
            benchmark_return=0.08,
            alpha=0.04,
            beta=1.1,
            information_ratio=0.3,
            tracking_error=0.05,
            equity_curve=pd.Series([1000000, 1150000]),
            monthly_returns=pd.Series([0.01, 0.02]),
            trade_log=[],
            risk_metrics={
                "return_stability": 0.85,
                "sharpe_stability": 0.78,
                "performance_degradation": 0.05,
                "overfitting_risk": 0.1,
                "var_95": -0.025,
                "var_99": -0.045,
                "cvar_95": -0.035,
                "cvar_99": -0.055,
                "skewness": -0.2,
                "kurtosis": 3.5,
            },
            performance_attribution={},
        )

        report = backtesting_engine.generate_performance_report(result)

        assert isinstance(report, str)
        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "Test Strategy" in report
        assert "15.00%" in report  # Total return
        assert "12.00%" in report  # Annual return
        assert "0.67" in report  # Sharpe ratio
        assert "100" in report  # Total trades


class TestWalkForwardAnalysis:
    """Test walk-forward analysis functionality."""

    @pytest.mark.asyncio
    async def test_run_walk_forward_analysis(
        self, backtesting_engine, simple_strategy, backtest_config, sample_stock_data
    ):
        """Test walk-forward analysis execution."""
        wf_results = await backtesting_engine._run_walk_forward_analysis(
            simple_strategy, backtest_config, sample_stock_data
        )

        assert isinstance(wf_results, list)
        # Should have some results if data is sufficient
        if len(sample_stock_data) >= 100:
            assert len(wf_results) > 0
            for result in wf_results:
                assert isinstance(result, BacktestResult)

    def test_calculate_stability_metrics_empty(self, backtesting_engine):
        """Test stability metrics calculation with empty results."""
        main_result = BacktestResult(
            strategy_name="Test",
            config=BacktestConfig("Test", date(2023, 1, 1), date(2023, 12, 31)),
            total_return=0.1,
            annual_return=0.1,
            volatility=0.15,
            sharpe_ratio=0.67,
            sortino_ratio=0.8,
            calmar_ratio=1.25,
            max_drawdown=-0.05,
            max_drawdown_duration=30,
            win_rate=0.6,
            profit_factor=1.5,
            total_trades=50,
            avg_trade_return=100.0,
            best_trade=1000.0,
            worst_trade=-500.0,
            benchmark_return=0.08,
            alpha=0.02,
            beta=1.1,
            information_ratio=0.2,
            tracking_error=0.03,
            equity_curve=pd.Series([100000, 110000]),
            monthly_returns=pd.Series([0.01, 0.02]),
            trade_log=[],
            risk_metrics={},
            performance_attribution={},
        )

        metrics = backtesting_engine._calculate_stability_metrics(main_result, [])

        assert metrics["return_stability"] == 0.0
        assert metrics["sharpe_stability"] == 0.0
        assert metrics["overfitting_risk"] == 1.0
        assert "consistency_score" in metrics
        assert "robustness_score" in metrics

    def test_calculate_stability_metrics_with_results(self, backtesting_engine):
        """Test stability metrics calculation with walk-forward results."""
        main_result = BacktestResult(
            strategy_name="Test",
            config=BacktestConfig("Test", date(2023, 1, 1), date(2023, 12, 31)),
            total_return=0.15,
            annual_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=0.8,
            calmar_ratio=1.25,
            max_drawdown=-0.08,
            max_drawdown_duration=45,
            win_rate=0.55,
            profit_factor=1.25,
            total_trades=100,
            avg_trade_return=150.0,
            best_trade=2000.0,
            worst_trade=-800.0,
            benchmark_return=0.08,
            alpha=0.04,
            beta=1.1,
            information_ratio=0.3,
            tracking_error=0.05,
            equity_curve=pd.Series([100000, 115000]),
            monthly_returns=pd.Series([0.01, 0.02]),
            trade_log=[],
            risk_metrics={},
            performance_attribution={},
        )

        # Create mock walk-forward results
        wf_results = []
        for i in range(3):
            wf_result = BacktestResult(
                strategy_name=f"Test_WF_{i}",
                config=BacktestConfig(
                    f"Test_WF_{i}", date(2023, 1, 1), date(2023, 4, 1)
                ),
                total_return=0.08 + i * 0.02,
                annual_return=0.10 + i * 0.01,
                volatility=0.16,
                sharpe_ratio=0.6 + i * 0.05,
                sortino_ratio=0.75,
                calmar_ratio=1.2,
                max_drawdown=-0.06,
                max_drawdown_duration=20,
                win_rate=0.5,
                profit_factor=1.2,
                total_trades=30,
                avg_trade_return=50.0,
                best_trade=500.0,
                worst_trade=-200.0,
                benchmark_return=0.06,
                alpha=0.02,
                beta=1.0,
                information_ratio=0.25,
                tracking_error=0.04,
                equity_curve=pd.Series([100000, 108000 + i * 2000]),
                monthly_returns=pd.Series([0.01, 0.015]),
                trade_log=[],
                risk_metrics={},
                performance_attribution={},
            )
            wf_results.append(wf_result)

        metrics = backtesting_engine._calculate_stability_metrics(
            main_result, wf_results
        )

        assert isinstance(metrics["return_stability"], float)
        assert isinstance(metrics["sharpe_stability"], float)
        assert isinstance(metrics["performance_degradation"], float)
        assert isinstance(metrics["overfitting_risk"], float)
        assert isinstance(metrics["consistency_score"], float)
        assert isinstance(metrics["robustness_score"], float)
        assert 0.0 <= metrics["return_stability"] <= 1.0
        assert 0.0 <= metrics["overfitting_risk"] <= 1.0

    @pytest.mark.asyncio
    async def test_run_parameter_optimization(
        self, backtesting_engine, sample_stock_data
    ):
        """Test parameter optimization functionality."""
        config = BacktestConfig(
            strategy_name="Optimization Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 30),  # Shorter period for faster testing
            initial_capital=100000.0,
        )

        param_grid = {
            "ma_short": [5, 10],
            "ma_long": [20, 30],
            "position_size": [0.05, 0.1],
        }

        results = await backtesting_engine.run_parameter_optimization(
            SimpleMovingAverageStrategy,
            param_grid,
            config,
            sample_stock_data,
            cv_folds=2,
        )

        assert "best_params" in results
        assert "best_score" in results
        assert "optimization_results" in results
        assert isinstance(results["optimization_results"], list)

    def test_generate_param_combinations(self, backtesting_engine):
        """Test parameter combination generation."""
        param_grid = {"param1": [1, 2], "param2": ["a", "b"], "param3": [0.1, 0.2]}

        combinations = backtesting_engine._generate_param_combinations(param_grid)

        assert len(combinations) == 8  # 2 * 2 * 2
        assert all(isinstance(combo, dict) for combo in combinations)
        assert all(len(combo) == 3 for combo in combinations)

    def test_assess_overfitting_risk_low_risk(self, backtesting_engine):
        """Test overfitting risk assessment for low-risk strategy."""
        result = BacktestResult(
            strategy_name="Low Risk Test",
            config=BacktestConfig("Test", date(2023, 1, 1), date(2023, 12, 31)),
            total_return=0.12,
            annual_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,  # Reasonable Sharpe ratio
            sortino_ratio=0.9,
            calmar_ratio=1.5,
            max_drawdown=-0.06,
            max_drawdown_duration=30,
            win_rate=0.6,  # Reasonable win rate
            profit_factor=1.4,
            total_trades=50,  # Sufficient trades
            avg_trade_return=120.0,
            best_trade=1500.0,
            worst_trade=-600.0,
            benchmark_return=0.08,
            alpha=0.04,
            beta=1.0,
            information_ratio=0.3,
            tracking_error=0.04,
            equity_curve=pd.Series([100000, 112000]),
            monthly_returns=pd.Series([0.01, 0.02]),
            trade_log=[],
            risk_metrics={
                "return_stability": 0.8,
                "sharpe_stability": 0.75,
                "performance_degradation": 0.1,
                "consistency_score": 0.7,
                "overfitting_risk": 0.1,
            },
            performance_attribution={},
        )

        assessment = backtesting_engine.assess_overfitting_risk(result)

        assert assessment["risk_level"] == "LOW"
        assert assessment["risk_score"] < 0.3
        assert isinstance(assessment["warnings"], list)
        assert isinstance(assessment["recommendations"], list)
        assert "assessment_summary" in assessment

    def test_assess_overfitting_risk_high_risk(self, backtesting_engine):
        """Test overfitting risk assessment for high-risk strategy."""
        result = BacktestResult(
            strategy_name="High Risk Test",
            config=BacktestConfig("Test", date(2023, 1, 1), date(2023, 12, 31)),
            total_return=0.50,
            annual_return=0.50,
            volatility=0.20,
            sharpe_ratio=4.0,  # Suspiciously high Sharpe ratio
            sortino_ratio=5.0,
            calmar_ratio=10.0,
            max_drawdown=-0.02,
            max_drawdown_duration=5,
            win_rate=0.99,  # Perfect win rate
            profit_factor=10.0,
            total_trades=5,  # Very few trades
            avg_trade_return=5000.0,
            best_trade=10000.0,
            worst_trade=-100.0,
            benchmark_return=0.08,
            alpha=0.42,
            beta=0.5,
            information_ratio=2.0,
            tracking_error=0.02,
            equity_curve=pd.Series([100000, 150000]),
            monthly_returns=pd.Series([0.04, 0.05]),
            trade_log=[],
            risk_metrics={
                "return_stability": 0.2,  # Low stability
                "sharpe_stability": 0.1,
                "performance_degradation": 0.4,  # High degradation
                "consistency_score": 0.3,
                "overfitting_risk": 0.8,
            },
            performance_attribution={},
        )

        assessment = backtesting_engine.assess_overfitting_risk(result)

        assert assessment["risk_level"] == "HIGH"
        assert assessment["risk_score"] > 0.6
        assert len(assessment["warnings"]) > 0
        assert len(assessment["recommendations"]) > 0

    def test_generate_walk_forward_report(self, backtesting_engine):
        """Test walk-forward analysis report generation."""
        main_result = BacktestResult(
            strategy_name="WF Report Test",
            config=BacktestConfig("Test", date(2023, 1, 1), date(2023, 12, 31)),
            total_return=0.15,
            annual_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=0.8,
            calmar_ratio=1.25,
            max_drawdown=-0.08,
            max_drawdown_duration=45,
            win_rate=0.55,
            profit_factor=1.25,
            total_trades=100,
            avg_trade_return=150.0,
            best_trade=2000.0,
            worst_trade=-800.0,
            benchmark_return=0.08,
            alpha=0.04,
            beta=1.1,
            information_ratio=0.3,
            tracking_error=0.05,
            equity_curve=pd.Series([100000, 115000]),
            monthly_returns=pd.Series([0.01, 0.02]),
            trade_log=[],
            risk_metrics={
                "return_stability": 0.75,
                "sharpe_stability": 0.70,
                "performance_degradation": 0.05,
                "consistency_score": 0.6,
                "robustness_score": 0.68,
                "overfitting_risk": 0.2,
            },
            performance_attribution={},
        )

        # Create mock walk-forward results
        wf_results = []
        for i in range(2):
            wf_result = BacktestResult(
                strategy_name=f"WF_Test_{i}",
                config=BacktestConfig(
                    f"WF_Test_{i}", date(2023, 1, 1), date(2023, 6, 30)
                ),
                total_return=0.08 + i * 0.02,
                annual_return=0.10 + i * 0.01,
                volatility=0.16,
                sharpe_ratio=0.6 + i * 0.05,
                sortino_ratio=0.75,
                calmar_ratio=1.2,
                max_drawdown=-0.06,
                max_drawdown_duration=20,
                win_rate=0.5,
                profit_factor=1.2,
                total_trades=30,
                avg_trade_return=50.0,
                best_trade=500.0,
                worst_trade=-200.0,
                benchmark_return=0.06,
                alpha=0.02,
                beta=1.0,
                information_ratio=0.25,
                tracking_error=0.04,
                equity_curve=pd.Series([100000, 108000 + i * 2000]),
                monthly_returns=pd.Series([0.01, 0.015]),
                trade_log=[],
                risk_metrics={},
                performance_attribution={},
            )
            wf_results.append(wf_result)

        report = backtesting_engine.generate_walk_forward_report(
            main_result, wf_results
        )

        assert isinstance(report, str)
        assert "WALK-FORWARD ANALYSIS REPORT" in report
        assert "WF Report Test" in report
        assert "STABILITY METRICS" in report
        assert "PERIOD-BY-PERIOD BREAKDOWN" in report

    @pytest.mark.asyncio
    async def test_run_comprehensive_validation(
        self, backtesting_engine, sample_stock_data
    ):
        """Test comprehensive validation workflow."""
        config = BacktestConfig(
            strategy_name="Comprehensive Validation Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 30),  # Shorter period for faster testing
            initial_capital=100000.0,
            strategy_params={"ma_short": 10, "ma_long": 30, "position_size": 0.1},
        )

        param_grid = {"ma_short": [5, 10], "ma_long": [20, 30]}

        results = await backtesting_engine.run_comprehensive_validation(
            SimpleMovingAverageStrategy, config, sample_stock_data, param_grid
        )

        assert "main_backtest" in results
        assert "parameter_optimization" in results
        assert "overfitting_assessment" in results
        assert "validation_summary" in results

        if results["main_backtest"]:
            assert isinstance(results["main_backtest"], BacktestResult)

        if results["overfitting_assessment"]:
            assert "risk_level" in results["overfitting_assessment"]
            assert "risk_score" in results["overfitting_assessment"]


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_backtest_workflow(self, sample_stock_data):
        """Test complete backtesting workflow."""
        # Create components
        engine = EnhancedBacktestingEngine()
        strategy = SimpleMovingAverageStrategy(
            {"ma_short": 5, "ma_long": 20, "position_size": 0.05}
        )

        config = BacktestConfig(
            strategy_name="Integration Test Strategy",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=500000.0,
            transaction_cost=0.002,
            slippage=0.001,
        )

        # Run backtest
        result = await engine.run_comprehensive_backtest(
            strategy, config, sample_stock_data
        )

        # Verify results
        assert result.strategy_name == "Integration Test Strategy"
        assert result.config.initial_capital == 500000.0
        assert len(result.equity_curve) > 0
        assert isinstance(result.total_return, float)
        assert isinstance(result.risk_metrics, dict)

        # Verify walk-forward analysis was performed
        assert "return_stability" in result.risk_metrics
        assert "overfitting_risk" in result.risk_metrics

        # Generate and verify report
        report = engine.generate_performance_report(result)
        assert "Integration Test Strategy" in report
        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "STABILITY METRICS" in report

    @pytest.mark.asyncio
    async def test_walk_forward_integration(self, sample_stock_data):
        """Test walk-forward analysis integration."""
        engine = EnhancedBacktestingEngine()
        strategy = SimpleMovingAverageStrategy(
            {"ma_short": 10, "ma_long": 30, "position_size": 0.08}
        )

        config = BacktestConfig(
            strategy_name="Walk-Forward Integration Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=1000000.0,
        )

        # Run comprehensive backtest (includes walk-forward analysis)
        result = await engine.run_comprehensive_backtest(
            strategy, config, sample_stock_data
        )

        # Verify walk-forward metrics are present
        assert "return_stability" in result.risk_metrics
        assert "sharpe_stability" in result.risk_metrics
        assert "performance_degradation" in result.risk_metrics
        assert "overfitting_risk" in result.risk_metrics
        assert "consistency_score" in result.risk_metrics
        assert "robustness_score" in result.risk_metrics

        # Test overfitting assessment
        assessment = engine.assess_overfitting_risk(result)
        assert assessment["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert 0.0 <= assessment["risk_score"] <= 1.0

        # Generate walk-forward report
        # Note: We can't easily get wf_results from the public interface,
        # so we'll test with empty results
        wf_report = engine.generate_walk_forward_report(result, [])
        assert isinstance(wf_report, str)


if __name__ == "__main__":
    pytest.main([__file__])
