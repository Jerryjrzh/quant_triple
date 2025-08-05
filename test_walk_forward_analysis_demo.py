#!/usr/bin/env python3
"""
Walk-Forward Analysis Demo

This script demonstrates the walk-forward analysis functionality implemented in task 8.2.
It shows how to:
1. Run walk-forward validation to detect overfitting
2. Perform parameter optimization with cross-validation
3. Assess overfitting risk
4. Generate comprehensive validation reports

Author: Stock Analysis System
Date: 2024
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our enhanced backtesting engine
from stock_analysis_system.analysis.enhanced_backtesting_engine import (
    EnhancedBacktestingEngine,
    BacktestConfig,
    SimpleMovingAverageStrategy
)


class WalkForwardDemo:
    """Demonstration of walk-forward analysis capabilities."""
    
    def __init__(self):
        self.engine = EnhancedBacktestingEngine()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_sample_data(self, start_date: date, end_date: date, 
                           trend: float = 0.0005, volatility: float = 0.02) -> pd.DataFrame:
        """Generate sample stock data with specified characteristics."""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        business_days = [d for d in date_range if d.weekday() < 5]
        
        np.random.seed(42)  # For reproducible results
        n_days = len(business_days)
        
        # Generate returns with specified trend and volatility
        returns = np.random.normal(trend, volatility, n_days)
        
        # Add some autocorrelation to make it more realistic
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Generate prices
        prices = [100.0]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC data
        data = []
        for i, (date, price) in enumerate(zip(business_days, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005)) if i > 0 else price
            
            high = max(high, price, open_price)
            low = min(low, price, open_price)
            
            volume = int(np.random.lognormal(15, 1))
            
            data.append({
                'stock_code': '000001.SZ',
                'trade_date': date.date(),
                'open_price': round(open_price, 2),
                'high_price': round(high, 2),
                'low_price': round(low, 2),
                'close_price': round(price, 2),
                'volume': volume,
                'amount': round(volume * price, 2),
                'adj_factor': 1.0
            })
        
        return pd.DataFrame(data)
    
    async def demo_basic_walk_forward(self):
        """Demonstrate basic walk-forward analysis."""
        
        print("\n" + "="*60)
        print("DEMO 1: Basic Walk-Forward Analysis")
        print("="*60)
        
        # Generate sample data
        stock_data = self.generate_sample_data(
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            trend=0.0008,  # Positive trend
            volatility=0.018
        )
        
        # Create strategy and configuration
        strategy = SimpleMovingAverageStrategy({
            'ma_short': 10,
            'ma_long': 30,
            'position_size': 0.1
        })
        
        config = BacktestConfig(
            strategy_name="Basic WF Demo Strategy",
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=1000000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        print("Running comprehensive backtest with walk-forward analysis...")
        result = await self.engine.run_comprehensive_backtest(strategy, config, stock_data)
        
        # Display results
        print(f"\nMain Backtest Results:")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annual_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
        print(f"Total Trades: {result.total_trades}")
        
        # Display stability metrics
        if 'return_stability' in result.risk_metrics:
            print(f"\nStability Metrics:")
            print(f"Return Stability: {result.risk_metrics['return_stability']:.2f}")
            print(f"Sharpe Stability: {result.risk_metrics['sharpe_stability']:.2f}")
            print(f"Performance Degradation: {result.risk_metrics['performance_degradation']:.2%}")
            print(f"Overfitting Risk: {result.risk_metrics['overfitting_risk']:.2f}")
            print(f"Consistency Score: {result.risk_metrics['consistency_score']:.2f}")
            print(f"Robustness Score: {result.risk_metrics['robustness_score']:.2f}")
        
        # Assess overfitting risk
        overfitting_assessment = self.engine.assess_overfitting_risk(result)
        print(f"\nOverfitting Risk Assessment:")
        print(f"Risk Level: {overfitting_assessment['risk_level']}")
        print(f"Risk Score: {overfitting_assessment['risk_score']:.2f}")
        
        if overfitting_assessment['warnings']:
            print(f"\nWarnings:")
            for warning in overfitting_assessment['warnings']:
                print(f"  • {warning}")
        
        if overfitting_assessment['recommendations']:
            print(f"\nRecommendations:")
            for rec in overfitting_assessment['recommendations']:
                print(f"  • {rec}")
        
        return result
    
    async def demo_parameter_optimization(self):
        """Demonstrate parameter optimization with walk-forward validation."""
        
        print("\n" + "="*60)
        print("DEMO 2: Parameter Optimization with Walk-Forward Validation")
        print("="*60)
        
        # Generate sample data
        stock_data = self.generate_sample_data(
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            trend=0.0006,
            volatility=0.02
        )
        
        config = BacktestConfig(
            strategy_name="Parameter Optimization Demo",
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=1000000.0,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        # Define parameter grid
        param_grid = {
            'ma_short': [5, 10, 15],
            'ma_long': [20, 30, 40],
            'position_size': [0.05, 0.1, 0.15]
        }
        
        print("Running parameter optimization...")
        print(f"Testing {3 * 3 * 3} parameter combinations...")
        
        opt_results = await self.engine.run_parameter_optimization(
            SimpleMovingAverageStrategy,
            param_grid,
            config,
            stock_data,
            optimization_metric='sharpe_ratio',
            cv_folds=3
        )
        
        print(f"\nOptimization Results:")
        print(f"Best Score (Sharpe Ratio): {opt_results['best_score']:.4f}")
        print(f"Best Parameters: {opt_results['best_params']}")
        print(f"Total Combinations Tested: {len(opt_results['optimization_results'])}")
        
        # Show top 5 parameter combinations
        print(f"\nTop 5 Parameter Combinations:")
        for i, result in enumerate(opt_results['optimization_results'][:5]):
            print(f"{i+1}. Score: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
            print(f"   Params: {result['params']}")
        
        return opt_results
    
    async def demo_comprehensive_validation(self):
        """Demonstrate comprehensive strategy validation."""
        
        print("\n" + "="*60)
        print("DEMO 3: Comprehensive Strategy Validation")
        print("="*60)
        
        # Generate sample data with different characteristics for testing
        stock_data = self.generate_sample_data(
            start_date=date(2021, 1, 1),
            end_date=date(2023, 12, 31),
            trend=0.0004,
            volatility=0.025
        )
        
        config = BacktestConfig(
            strategy_name="Comprehensive Validation Demo",
            start_date=date(2021, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=1000000.0,
            transaction_cost=0.0015,
            slippage=0.0008,
            strategy_params={'ma_short': 12, 'ma_long': 26, 'position_size': 0.08}
        )
        
        # Define parameter grid for optimization
        param_grid = {
            'ma_short': [8, 12, 16],
            'ma_long': [24, 30, 36],
            'position_size': [0.06, 0.08, 0.10]
        }
        
        print("Running comprehensive validation...")
        print("This includes:")
        print("  • Main backtest")
        print("  • Walk-forward analysis")
        print("  • Parameter optimization")
        print("  • Overfitting risk assessment")
        
        validation_results = await self.engine.run_comprehensive_validation(
            SimpleMovingAverageStrategy,
            config,
            stock_data,
            param_grid
        )
        
        # Display validation summary
        if 'validation_summary' in validation_results:
            print("\n" + validation_results['validation_summary'])
        
        return validation_results
    
    def create_stability_visualization(self, results_list):
        """Create visualization of stability metrics across different scenarios."""
        
        print("\n" + "="*60)
        print("DEMO 4: Stability Metrics Visualization")
        print("="*60)
        
        # Extract stability metrics from results
        stability_data = []
        
        for i, result in enumerate(results_list):
            if hasattr(result, 'risk_metrics') and 'return_stability' in result.risk_metrics:
                stability_data.append({
                    'Scenario': f'Demo {i+1}',
                    'Return Stability': result.risk_metrics['return_stability'],
                    'Sharpe Stability': result.risk_metrics['sharpe_stability'],
                    'Consistency Score': result.risk_metrics['consistency_score'],
                    'Robustness Score': result.risk_metrics['robustness_score'],
                    'Overfitting Risk': result.risk_metrics['overfitting_risk']
                })
        
        if not stability_data:
            print("No stability data available for visualization.")
            return
        
        # Create visualization
        df = pd.DataFrame(stability_data)
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis: Stability Metrics Comparison', fontsize=16)
        
        # Return Stability
        axes[0, 0].bar(df['Scenario'], df['Return Stability'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Return Stability')
        axes[0, 0].set_ylabel('Stability Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
        axes[0, 0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor Threshold')
        axes[0, 0].legend()
        
        # Sharpe Stability
        axes[0, 1].bar(df['Scenario'], df['Sharpe Stability'], color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Sharpe Ratio Stability')
        axes[0, 1].set_ylabel('Stability Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        
        # Robustness Score
        axes[1, 0].bar(df['Scenario'], df['Robustness Score'], color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Overall Robustness Score')
        axes[1, 0].set_ylabel('Robustness Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axhline(y=0.6, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.4, color='red', linestyle='--', alpha=0.5)
        
        # Overfitting Risk
        axes[1, 1].bar(df['Scenario'], df['Overfitting Risk'], color='orange', alpha=0.7)
        axes[1, 1].set_title('Overfitting Risk')
        axes[1, 1].set_ylabel('Risk Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Low Risk')
        axes[1, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High Risk')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / "walk_forward_stability_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Stability metrics visualization saved to: {output_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, validation_results):
        """Generate a comprehensive walk-forward analysis report."""
        
        print("\n" + "="*60)
        print("DEMO 5: Comprehensive Walk-Forward Analysis Report")
        print("="*60)
        
        report_path = self.output_dir / "walk_forward_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE WALK-FORWARD ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Main backtest results
            main_result = validation_results.get('main_backtest')
            if main_result:
                f.write("MAIN BACKTEST PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Strategy: {main_result.strategy_name}\n")
                f.write(f"Period: {main_result.config.start_date} to {main_result.config.end_date}\n")
                f.write(f"Total Return: {main_result.total_return:.2%}\n")
                f.write(f"Annualized Return: {main_result.annual_return:.2%}\n")
                f.write(f"Sharpe Ratio: {main_result.sharpe_ratio:.2f}\n")
                f.write(f"Maximum Drawdown: {main_result.max_drawdown:.2%}\n")
                f.write(f"Total Trades: {main_result.total_trades}\n")
                f.write(f"Win Rate: {main_result.win_rate:.2%}\n\n")
                
                # Stability metrics
                if 'return_stability' in main_result.risk_metrics:
                    f.write("WALK-FORWARD STABILITY METRICS:\n")
                    f.write("-" * 35 + "\n")
                    f.write(f"Return Stability: {main_result.risk_metrics['return_stability']:.2f}\n")
                    f.write(f"Sharpe Stability: {main_result.risk_metrics['sharpe_stability']:.2f}\n")
                    f.write(f"Performance Degradation: {main_result.risk_metrics['performance_degradation']:.2%}\n")
                    f.write(f"Consistency Score: {main_result.risk_metrics['consistency_score']:.2f}\n")
                    f.write(f"Robustness Score: {main_result.risk_metrics['robustness_score']:.2f}\n")
                    f.write(f"Overfitting Risk: {main_result.risk_metrics['overfitting_risk']:.2f}\n\n")
            
            # Parameter optimization results
            opt_results = validation_results.get('parameter_optimization')
            if opt_results:
                f.write("PARAMETER OPTIMIZATION RESULTS:\n")
                f.write("-" * 35 + "\n")
                f.write(f"Best Score: {opt_results['best_score']:.4f}\n")
                f.write(f"Best Parameters: {opt_results['best_params']}\n")
                f.write(f"Combinations Tested: {len(opt_results['optimization_results'])}\n\n")
            
            # Overfitting assessment
            overfitting = validation_results.get('overfitting_assessment')
            if overfitting:
                f.write("OVERFITTING RISK ASSESSMENT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Risk Level: {overfitting['risk_level']}\n")
                f.write(f"Risk Score: {overfitting['risk_score']:.2f}\n")
                
                if overfitting['warnings']:
                    f.write("\nWarnings:\n")
                    for warning in overfitting['warnings']:
                        f.write(f"  • {warning}\n")
                
                if overfitting['recommendations']:
                    f.write("\nRecommendations:\n")
                    for rec in overfitting['recommendations']:
                        f.write(f"  • {rec}\n")
                
                f.write(f"\n{overfitting['assessment_summary']}\n")
            
            # Validation summary
            if 'validation_summary' in validation_results:
                f.write("\nVALIDATION SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(validation_results['validation_summary'])
        
        print(f"Comprehensive report saved to: {report_path}")
        
        # Also display key findings
        print("\nKEY FINDINGS:")
        if main_result and 'return_stability' in main_result.risk_metrics:
            stability = main_result.risk_metrics['return_stability']
            overfitting_risk = main_result.risk_metrics['overfitting_risk']
            
            if stability > 0.7 and overfitting_risk < 0.3:
                print("✅ Strategy shows excellent stability with low overfitting risk")
            elif stability > 0.5 and overfitting_risk < 0.5:
                print("⚠️ Strategy shows moderate stability with acceptable overfitting risk")
            else:
                print("🚨 Strategy shows poor stability with high overfitting risk")
    
    async def run_all_demos(self):
        """Run all walk-forward analysis demonstrations."""
        
        print("WALK-FORWARD ANALYSIS DEMONSTRATION")
        print("=" * 60)
        print("This demo showcases the walk-forward analysis functionality")
        print("implemented in task 8.2 of the Stock Analysis System.")
        print("\nFeatures demonstrated:")
        print("• Walk-forward validation using TimeSeriesSplit")
        print("• Parameter optimization with cross-validation")
        print("• Stability metrics calculation")
        print("• Overfitting risk assessment")
        print("• Comprehensive validation reports")
        
        results = []
        
        try:
            # Demo 1: Basic walk-forward analysis
            result1 = await self.demo_basic_walk_forward()
            results.append(result1)
            
            # Demo 2: Parameter optimization
            opt_results = await self.demo_parameter_optimization()
            
            # Demo 3: Comprehensive validation
            validation_results = await self.demo_comprehensive_validation()
            
            # Demo 4: Visualization
            if results:
                self.create_stability_visualization(results)
            
            # Demo 5: Comprehensive report
            if validation_results:
                self.generate_comprehensive_report(validation_results)
            
            print("\n" + "="*60)
            print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("All walk-forward analysis features have been demonstrated.")
            print(f"Output files saved to: {self.output_dir}")
            print("\nTask 8.2 implementation includes:")
            print("✅ TimeSeriesSplit for walk-forward validation")
            print("✅ Parameter optimization on training data")
            print("✅ Stability metrics calculation for strategy robustness")
            print("✅ Overfitting risk assessment and warnings")
            print("✅ Comprehensive reporting and visualization")
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
            print(f"\n❌ Demonstration failed with error: {e}")


async def main():
    """Main function to run the walk-forward analysis demo."""
    
    demo = WalkForwardDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())