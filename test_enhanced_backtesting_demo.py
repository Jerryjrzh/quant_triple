"""Demo script for Enhanced Backtesting Engine."""

import asyncio
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from stock_analysis_system.analysis.enhanced_backtesting_engine import (
    EnhancedBacktestingEngine,
    BacktestConfig,
    SimpleMovingAverageStrategy
)


class MomentumStrategy:
    """Momentum strategy for demonstration."""
    
    def __init__(self, params=None):
        self.params = params or {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15
        }
        self.price_history = []
    
    async def generate_signals(self, data, portfolio):
        """Generate momentum-based signals."""
        self.price_history.append(data['close_price'])
        
        # Keep only required history
        if len(self.price_history) > self.params['lookback_period']:
            self.price_history = self.price_history[-self.params['lookback_period']:]
        
        if len(self.price_history) < self.params['lookback_period']:
            return {'buy_signal': False, 'sell_signal': False}
        
        # Calculate momentum
        current_price = self.price_history[-1]
        past_price = self.price_history[0]
        momentum = (current_price - past_price) / past_price
        
        # Generate signals
        buy_signal = momentum > self.params['momentum_threshold']
        
        # Sell signal based on stop loss or take profit
        sell_signal = False
        if data['stock_code'] in portfolio.positions:
            position = portfolio.positions[data['stock_code']]
            pnl_pct = (current_price - position.avg_price) / position.avg_price
            
            if pnl_pct <= -self.params['stop_loss'] or pnl_pct >= self.params['take_profit']:
                sell_signal = True
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'momentum': momentum
        }
    
    async def calculate_position_size(self, signal, portfolio, current_price):
        """Calculate position size based on momentum strength."""
        base_size = portfolio.cash * self.params['position_size']
        
        # Adjust size based on momentum strength
        momentum = abs(signal.get('momentum', 0))
        size_multiplier = min(2.0, 1.0 + momentum * 10)  # Max 2x size
        
        adjusted_size = base_size * size_multiplier
        return adjusted_size / current_price if current_price > 0 else 0
    
    async def on_bar(self, data, portfolio):
        """Process each bar of data."""
        from stock_analysis_system.analysis.enhanced_backtesting_engine import Order, OrderSide, OrderType
        
        signals = await self.generate_signals(data, portfolio)
        orders = []
        
        if signals['buy_signal'] and data['stock_code'] not in portfolio.positions:
            size = await self.calculate_position_size(signals, portfolio, data['close_price'])
            if size > 0:
                order = Order(
                    order_id=f"buy_{data['stock_code']}_{data['trade_date']}",
                    symbol=data['stock_code'],
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=size,
                    timestamp=pd.to_datetime(data['trade_date'])
                )
                orders.append(order)
        
        elif signals['sell_signal'] and data['stock_code'] in portfolio.positions:
            position = portfolio.positions[data['stock_code']]
            if position.quantity > 0:
                order = Order(
                    order_id=f"sell_{data['stock_code']}_{data['trade_date']}",
                    symbol=data['stock_code'],
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    timestamp=pd.to_datetime(data['trade_date'])
                )
                orders.append(order)
        
        return orders


async def demonstrate_basic_backtesting():
    """Demonstrate basic backtesting functionality."""
    print("=== Enhanced Backtesting Engine Demo ===\n")
    
    # Create backtesting engine
    engine = EnhancedBacktestingEngine()
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy({
        'ma_short': 10,
        'ma_long': 30,
        'position_size': 0.15
    })
    
    # Create configuration
    config = BacktestConfig(
        strategy_name="Simple MA Crossover",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1000000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        benchmark="000300.SH"
    )
    
    print(f"Running backtest for strategy: {config.strategy_name}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Transaction Cost: {config.transaction_cost:.3%}")
    print(f"Slippage: {config.slippage:.3%}\n")
    
    # Run backtest
    result = await engine.run_comprehensive_backtest(strategy, config)
    
    # Display results
    print("=== BACKTEST RESULTS ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annual_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    
    if 'return_stability' in result.risk_metrics:
        print(f"\n=== STABILITY METRICS ===")
        print(f"Return Stability: {result.risk_metrics['return_stability']:.2f}")
        print(f"Performance Degradation: {result.risk_metrics['performance_degradation']:.2%}")
        print(f"Overfitting Risk: {result.risk_metrics['overfitting_risk']:.2f}")
    
    return result


async def demonstrate_momentum_strategy():
    """Demonstrate momentum strategy backtesting."""
    print("\n=== Momentum Strategy Demo ===\n")
    
    # Create backtesting engine
    engine = EnhancedBacktestingEngine()
    
    # Create momentum strategy
    strategy = MomentumStrategy({
        'lookback_period': 15,
        'momentum_threshold': 0.03,
        'position_size': 0.12,
        'stop_loss': 0.08,
        'take_profit': 0.20
    })
    
    # Create configuration
    config = BacktestConfig(
        strategy_name="Momentum Strategy",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1000000.0,
        transaction_cost=0.0015,
        slippage=0.001,
        benchmark="000300.SH"
    )
    
    print(f"Running backtest for strategy: {config.strategy_name}")
    print(f"Lookback Period: {strategy.params['lookback_period']} days")
    print(f"Momentum Threshold: {strategy.params['momentum_threshold']:.1%}")
    print(f"Stop Loss: {strategy.params['stop_loss']:.1%}")
    print(f"Take Profit: {strategy.params['take_profit']:.1%}\n")
    
    # Run backtest
    result = await engine.run_comprehensive_backtest(strategy, config)
    
    # Display results
    print("=== MOMENTUM STRATEGY RESULTS ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annual_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"Max Drawdown Duration: {result.max_drawdown_duration} days")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Best Trade: ${result.best_trade:.2f}")
    print(f"Worst Trade: ${result.worst_trade:.2f}")
    
    return result


async def demonstrate_multiple_benchmarks():
    """Demonstrate backtesting against multiple benchmarks."""
    print("\n=== Multiple Benchmarks Demo ===\n")
    
    # Create backtesting engine
    engine = EnhancedBacktestingEngine()
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy({
        'ma_short': 8,
        'ma_long': 25,
        'position_size': 0.2
    })
    
    # Create configuration
    config = BacktestConfig(
        strategy_name="Multi-Benchmark Test",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1000000.0
    )
    
    # Define benchmarks
    benchmarks = ["000300.SH", "000905.SH", "399006.SZ"]  # CSI300, CSI500, ChiNext
    benchmark_names = {
        "000300.SH": "CSI 300",
        "000905.SH": "CSI 500", 
        "399006.SZ": "ChiNext"
    }
    
    print("Running backtest against multiple benchmarks:")
    for benchmark in benchmarks:
        print(f"  - {benchmark_names.get(benchmark, benchmark)}")
    print()
    
    # Run backtests
    results = await engine.run_multiple_benchmarks(strategy, config, benchmarks)
    
    # Display comparison
    print("=== BENCHMARK COMPARISON ===")
    print(f"{'Benchmark':<15} {'Return':<10} {'Sharpe':<8} {'Max DD':<8} {'Alpha':<8}")
    print("-" * 55)
    
    for benchmark, result in results.items():
        name = benchmark_names.get(benchmark, benchmark)[:14]
        print(f"{name:<15} {result.annual_return:>8.2%} {result.sharpe_ratio:>7.2f} "
              f"{result.max_drawdown:>7.2%} {result.alpha:>7.2%}")
    
    return results


async def demonstrate_performance_report():
    """Demonstrate comprehensive performance report generation."""
    print("\n=== Performance Report Demo ===\n")
    
    # Create backtesting engine
    engine = EnhancedBacktestingEngine()
    
    # Create strategy
    strategy = MomentumStrategy({
        'lookback_period': 20,
        'momentum_threshold': 0.025,
        'position_size': 0.15,
        'stop_loss': 0.06,
        'take_profit': 0.18
    })
    
    # Create configuration
    config = BacktestConfig(
        strategy_name="Comprehensive Report Demo",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1000000.0,
        transaction_cost=0.0012,
        slippage=0.0008
    )
    
    # Run backtest
    result = await engine.run_comprehensive_backtest(strategy, config)
    
    # Generate comprehensive report
    report = engine.generate_performance_report(result)
    print(report)
    
    return result


def create_performance_visualization(result):
    """Create performance visualization charts."""
    print("\n=== Creating Performance Visualizations ===\n")
    
    try:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtesting Results: {result.strategy_name}', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values, 
                       linewidth=2, color='blue', label='Strategy')
        axes[0, 0].axhline(y=result.config.initial_capital, color='red', 
                          linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        running_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - running_max) / running_max * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly Returns
        if len(result.monthly_returns) > 0:
            monthly_returns_pct = result.monthly_returns * 100
            colors = ['green' if x > 0 else 'red' for x in monthly_returns_pct]
            axes[1, 0].bar(range(len(monthly_returns_pct)), monthly_returns_pct, 
                          color=colors, alpha=0.7)
            axes[1, 0].set_title('Monthly Returns')
            axes[1, 0].set_ylabel('Return (%)')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Monthly Data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Monthly Returns')
        
        # 4. Performance Metrics
        metrics_text = f"""
Total Return: {result.total_return:.2%}
Annual Return: {result.annual_return:.2%}
Volatility: {result.volatility:.2%}
Sharpe Ratio: {result.sharpe_ratio:.2f}
Max Drawdown: {result.max_drawdown:.2%}
Win Rate: {result.win_rate:.2%}
Total Trades: {result.total_trades}
        """
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'backtesting_performance.png', dpi=300, bbox_inches='tight')
        print(f"Performance chart saved to: {output_dir / 'backtesting_performance.png'}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Matplotlib may not be available or configured properly.")


async def run_comprehensive_demo():
    """Run comprehensive backtesting demonstration."""
    print("Starting Enhanced Backtesting Engine Comprehensive Demo...\n")
    
    try:
        # 1. Basic backtesting
        basic_result = await demonstrate_basic_backtesting()
        
        # 2. Momentum strategy
        momentum_result = await demonstrate_momentum_strategy()
        
        # 3. Multiple benchmarks
        benchmark_results = await demonstrate_multiple_benchmarks()
        
        # 4. Performance report
        report_result = await demonstrate_performance_report()
        
        # 5. Create visualizations
        create_performance_visualization(momentum_result)
        
        print("\n=== DEMO SUMMARY ===")
        print("✅ Basic backtesting completed")
        print("✅ Momentum strategy tested")
        print("✅ Multiple benchmark comparison completed")
        print("✅ Comprehensive performance report generated")
        print("✅ Performance visualizations created")
        
        print(f"\nBest performing strategy: {momentum_result.strategy_name}")
        print(f"Annual Return: {momentum_result.annual_return:.2%}")
        print(f"Sharpe Ratio: {momentum_result.sharpe_ratio:.2f}")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo())