#!/usr/bin/env python3
"""
高级分析示例

本示例展示了如何使用系统进行复杂的股票分析，
包括技术分析、资金流向分析、龙虎榜分析等。
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AdvancedStockAnalyzer:
    """高级股票分析器"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", token: str = None):
        self.base_url = base_url
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def login(self, username: str, password: str) -> bool:
        """登录获取令牌"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            response.raise_for_status()
            
            data = response.json()
            token = data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            return True
            
        except Exception as e:
            print(f"登录失败: {e}")
            return False
    
    def get_stock_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取股票数据并转换为DataFrame"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/data/history/{symbol}"
        params = {"start_date": start_date, "end_date": end_date}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        return df
    
    def get_fund_flow_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """获取资金流向历史数据"""
        fund_flows = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            try:
                url = f"{self.base_url}/data/fund-flow/{symbol}"
                params = {"period": "1d", "date": date}
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()["data"]
                    data['date'] = date
                    fund_flows.append(data)
            except:
                continue
        
        if fund_flows:
            df = pd.DataFrame(fund_flows)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df.sort_index()
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # KDJ
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 成交量指标
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        return df
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """趋势分析"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        analysis = {
            'price_trend': 'unknown',
            'volume_trend': 'unknown',
            'strength': 0,
            'signals': []
        }
        
        # 价格趋势
        if latest['close'] > latest['ma5'] > latest['ma20']:
            analysis['price_trend'] = 'strong_uptrend'
            analysis['strength'] += 3
        elif latest['close'] > latest['ma5']:
            analysis['price_trend'] = 'uptrend'
            analysis['strength'] += 2
        elif latest['close'] < latest['ma5'] < latest['ma20']:
            analysis['price_trend'] = 'strong_downtrend'
            analysis['strength'] -= 3
        elif latest['close'] < latest['ma5']:
            analysis['price_trend'] = 'downtrend'
            analysis['strength'] -= 2
        else:
            analysis['price_trend'] = 'sideways'
        
        # 成交量趋势
        if latest['volume_ratio'] > 1.5:
            analysis['volume_trend'] = 'high'
            analysis['strength'] += 1
        elif latest['volume_ratio'] < 0.7:
            analysis['volume_trend'] = 'low'
            analysis['strength'] -= 1
        else:
            analysis['volume_trend'] = 'normal'
        
        # 技术信号
        if latest['rsi'] > 70:
            analysis['signals'].append('RSI超买')
        elif latest['rsi'] < 30:
            analysis['signals'].append('RSI超卖')
        
        if latest['close'] > latest['bb_upper']:
            analysis['signals'].append('突破布林带上轨')
        elif latest['close'] < latest['bb_lower']:
            analysis['signals'].append('跌破布林带下轨')
        
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            analysis['signals'].append('MACD金叉')
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            analysis['signals'].append('MACD死叉')
        
        if latest['k'] > latest['d'] and latest['k'] > 80:
            analysis['signals'].append('KDJ高位金叉')
        elif latest['k'] < latest['d'] and latest['k'] < 20:
            analysis['signals'].append('KDJ低位死叉')
        
        return analysis
    
    def analyze_fund_flow(self, fund_df: pd.DataFrame) -> Dict[str, Any]:
        """资金流向分析"""
        if fund_df.empty:
            return {'status': 'no_data'}
        
        # 计算累计资金流向
        fund_df['main_cumsum'] = fund_df['main_net_inflow'].cumsum()
        fund_df['super_large_cumsum'] = fund_df['super_large_net_inflow'].cumsum()
        
        latest = fund_df.iloc[-1]
        recent_5d = fund_df.tail(5)
        
        analysis = {
            'latest_main_flow': latest['main_net_inflow'],
            'recent_5d_main_flow': recent_5d['main_net_inflow'].sum(),
            'main_flow_trend': 'unknown',
            'institution_activity': 'unknown',
            'signals': []
        }
        
        # 主力资金趋势
        if analysis['recent_5d_main_flow'] > 0:
            analysis['main_flow_trend'] = 'inflow'
            if analysis['latest_main_flow'] > 0:
                analysis['signals'].append('主力资金持续流入')
        else:
            analysis['main_flow_trend'] = 'outflow'
            if analysis['latest_main_flow'] < 0:
                analysis['signals'].append('主力资金持续流出')
        
        # 机构活跃度
        super_large_ratio = abs(latest['super_large_net_inflow']) / abs(latest['main_net_inflow']) if latest['main_net_inflow'] != 0 else 0
        if super_large_ratio > 0.6:
            analysis['institution_activity'] = 'high'
            analysis['signals'].append('机构资金活跃')
        else:
            analysis['institution_activity'] = 'normal'
        
        return analysis
    
    def get_dragon_tiger_analysis(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """龙虎榜分析"""
        dragon_tiger_data = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            try:
                url = f"{self.base_url}/data/dragon-tiger"
                params = {"date": date, "symbol": symbol}
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()["data"]
                    if data:
                        dragon_tiger_data.extend(data)
            except:
                continue
        
        if not dragon_tiger_data:
            return {'status': 'no_data'}
        
        # 分析龙虎榜数据
        total_appearances = len(dragon_tiger_data)
        total_net_buy = sum(item['net_amount'] for item in dragon_tiger_data)
        
        # 统计上榜原因
        reasons = {}
        for item in dragon_tiger_data:
            reason = item['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        
        # 统计机构参与情况
        institution_count = 0
        institution_net_buy = 0
        
        for item in dragon_tiger_data:
            for inst in item.get('institutions', []):
                if inst['type'] == 'institution':
                    institution_count += 1
                    institution_net_buy += inst['buy_amount'] - inst['sell_amount']
        
        return {
            'total_appearances': total_appearances,
            'total_net_buy': total_net_buy,
            'avg_net_buy': total_net_buy / total_appearances,
            'main_reasons': sorted(reasons.items(), key=lambda x: x[1], reverse=True),
            'institution_participation': {
                'count': institution_count,
                'net_buy': institution_net_buy
            }
        }
    
    def comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """综合分析"""
        print(f"正在分析 {symbol}...")
        
        # 获取数据
        stock_df = self.get_stock_data(symbol, days=60)
        stock_df = self.calculate_technical_indicators(stock_df)
        
        fund_df = self.get_fund_flow_history(symbol, days=20)
        dragon_tiger = self.get_dragon_tiger_analysis(symbol, days=7)
        
        # 各项分析
        trend_analysis = self.analyze_trend(stock_df)
        fund_analysis = self.analyze_fund_flow(fund_df)
        
        # 综合评分
        score = 50  # 基础分
        
        # 技术面评分
        score += trend_analysis['strength'] * 5
        
        # 资金面评分
        if fund_analysis.get('main_flow_trend') == 'inflow':
            score += 10
        elif fund_analysis.get('main_flow_trend') == 'outflow':
            score -= 10
        
        # 龙虎榜评分
        if dragon_tiger.get('status') != 'no_data':
            if dragon_tiger['avg_net_buy'] > 0:
                score += 5
            if dragon_tiger['institution_participation']['net_buy'] > 0:
                score += 5
        
        # 限制评分范围
        score = max(0, min(100, score))
        
        return {
            'symbol': symbol,
            'analysis_time': datetime.now().isoformat(),
            'comprehensive_score': score,
            'latest_price': float(stock_df['close'].iloc[-1]),
            'price_change_pct': float((stock_df['close'].iloc[-1] / stock_df['close'].iloc[-2] - 1) * 100),
            'trend_analysis': trend_analysis,
            'fund_analysis': fund_analysis,
            'dragon_tiger_analysis': dragon_tiger,
            'technical_data': {
                'rsi': float(stock_df['rsi'].iloc[-1]),
                'macd': float(stock_df['macd'].iloc[-1]),
                'volume_ratio': float(stock_df['volume_ratio'].iloc[-1])
            }
        }
    
    def plot_comprehensive_chart(self, symbol: str):
        """绘制综合分析图表"""
        # 获取数据
        stock_df = self.get_stock_data(symbol, days=60)
        stock_df = self.calculate_technical_indicators(stock_df)
        fund_df = self.get_fund_flow_history(symbol, days=30)
        
        # 创建子图
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'{symbol} 综合技术分析', fontsize=16, fontweight='bold')
        
        # 1. 价格和移动平均线
        ax1 = axes[0]
        ax1.plot(stock_df.index, stock_df['close'], label='收盘价', linewidth=2, color='black')
        ax1.plot(stock_df.index, stock_df['ma5'], label='MA5', alpha=0.7, color='red')
        ax1.plot(stock_df.index, stock_df['ma20'], label='MA20', alpha=0.7, color='blue')
        ax1.plot(stock_df.index, stock_df['ma60'], label='MA60', alpha=0.7, color='green')
        
        # 布林带
        ax1.fill_between(stock_df.index, stock_df['bb_upper'], stock_df['bb_lower'], 
                        alpha=0.1, color='gray', label='布林带')
        
        ax1.set_title('价格走势与移动平均线')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 成交量
        ax2 = axes[1]
        colors = ['red' if close >= open else 'green' 
                 for close, open in zip(stock_df['close'], stock_df['open'])]
        ax2.bar(stock_df.index, stock_df['volume'], color=colors, alpha=0.6)
        ax2.plot(stock_df.index, stock_df['volume_ma5'], label='成交量MA5', color='orange')
        
        ax2.set_title('成交量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD和RSI
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        # MACD
        ax3.plot(stock_df.index, stock_df['macd'], label='MACD', color='blue')
        ax3.plot(stock_df.index, stock_df['macd_signal'], label='Signal', color='red')
        ax3.bar(stock_df.index, stock_df['macd_histogram'], label='Histogram', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.legend(loc='upper left')
        
        # RSI
        ax3_twin.plot(stock_df.index, stock_df['rsi'], label='RSI', color='purple', linewidth=2)
        ax3_twin.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3_twin.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3_twin.set_ylabel('RSI')
        ax3_twin.set_ylim(0, 100)
        ax3_twin.legend(loc='upper right')
        
        ax3.set_title('MACD与RSI指标')
        ax3.grid(True, alpha=0.3)
        
        # 4. 资金流向
        ax4 = axes[3]
        if not fund_df.empty:
            ax4.bar(fund_df.index, fund_df['main_net_inflow']/10000, 
                   color=['red' if x > 0 else 'green' for x in fund_df['main_net_inflow']],
                   alpha=0.7, label='主力净流入')
            ax4.plot(fund_df.index, fund_df['main_net_inflow'].rolling(5).mean()/10000, 
                    color='orange', label='5日均线')
            ax4.set_ylabel('资金流入(万元)')
            ax4.legend()
        
        ax4.set_title('主力资金流向')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def batch_analysis(self, symbols: List[str]) -> pd.DataFrame:
        """批量分析多只股票"""
        results = []
        
        for symbol in symbols:
            try:
                analysis = self.comprehensive_analysis(symbol)
                results.append({
                    'symbol': symbol,
                    'score': analysis['comprehensive_score'],
                    'price': analysis['latest_price'],
                    'change_pct': analysis['price_change_pct'],
                    'rsi': analysis['technical_data']['rsi'],
                    'volume_ratio': analysis['technical_data']['volume_ratio'],
                    'trend': analysis['trend_analysis']['price_trend'],
                    'signals': ', '.join(analysis['trend_analysis']['signals'][:3])
                })
            except Exception as e:
                print(f"分析 {symbol} 失败: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df.sort_values('score', ascending=False)


def demo_advanced_analysis():
    """高级分析演示"""
    print("🔬 高级股票分析演示")
    print("=" * 50)
    
    # 创建分析器
    analyzer = AdvancedStockAnalyzer()
    
    # 登录
    if not analyzer.login("admin", "password"):
        print("❌ 登录失败")
        return
    
    # 单只股票综合分析
    print("\n1️⃣ 单只股票综合分析")
    symbol = "000001.SZ"
    analysis = analyzer.comprehensive_analysis(symbol)
    
    print(f"\n📊 {symbol} 综合分析报告")
    print(f"综合评分: {analysis['comprehensive_score']}/100")
    print(f"最新价格: {analysis['latest_price']:.2f}")
    print(f"涨跌幅: {analysis['price_change_pct']:+.2f}%")
    
    print(f"\n📈 趋势分析:")
    trend = analysis['trend_analysis']
    print(f"价格趋势: {trend['price_trend']}")
    print(f"成交量: {trend['volume_trend']}")
    print(f"技术信号: {', '.join(trend['signals']) if trend['signals'] else '无'}")
    
    print(f"\n💰 资金分析:")
    fund = analysis['fund_analysis']
    if fund.get('status') != 'no_data':
        print(f"主力资金: {fund['main_flow_trend']}")
        print(f"最新流入: {fund['latest_main_flow']/10000:.2f}万元")
        print(f"5日累计: {fund['recent_5d_main_flow']/10000:.2f}万元")
    
    print(f"\n🏆 龙虎榜:")
    dt = analysis['dragon_tiger_analysis']
    if dt.get('status') != 'no_data':
        print(f"7日上榜次数: {dt['total_appearances']}")
        print(f"平均净买入: {dt['avg_net_buy']/10000:.2f}万元")
        print(f"机构参与: {dt['institution_participation']['count']}次")
    else:
        print("近期未上龙虎榜")
    
    # 绘制图表
    print("\n2️⃣ 绘制综合分析图表")
    try:
        analyzer.plot_comprehensive_chart(symbol)
    except Exception as e:
        print(f"绘图失败: {e}")
    
    # 批量分析
    print("\n3️⃣ 批量股票分析")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "000858.SZ"]
    batch_results = analyzer.batch_analysis(symbols)
    
    print("\n📋 批量分析结果 (按评分排序):")
    print(batch_results.to_string(index=False, float_format='%.2f'))
    
    # 筛选优质股票
    print("\n4️⃣ 优质股票筛选")
    high_score_stocks = batch_results[batch_results['score'] >= 60]
    uptrend_stocks = batch_results[batch_results['trend'].str.contains('uptrend', na=False)]
    
    print(f"\n高评分股票 (≥60分): {len(high_score_stocks)}只")
    if not high_score_stocks.empty:
        print(high_score_stocks[['symbol', 'score', 'change_pct']].to_string(index=False))
    
    print(f"\n上升趋势股票: {len(uptrend_stocks)}只")
    if not uptrend_stocks.empty:
        print(uptrend_stocks[['symbol', 'trend', 'change_pct']].to_string(index=False))


def demo_custom_strategy():
    """自定义策略演示"""
    print("\n🎯 自定义策略演示")
    print("=" * 30)
    
    analyzer = AdvancedStockAnalyzer()
    if not analyzer.login("admin", "password"):
        return
    
    # 定义策略：寻找突破股票
    def find_breakout_stocks(symbols: List[str]) -> List[Dict[str, Any]]:
        """寻找突破股票的策略"""
        breakout_stocks = []
        
        for symbol in symbols:
            try:
                df = analyzer.get_stock_data(symbol, days=30)
                df = analyzer.calculate_technical_indicators(df)
                
                latest = df.iloc[-1]
                prev_5d = df.iloc[-6:-1]
                
                # 突破条件
                conditions = {
                    'price_above_ma20': latest['close'] > latest['ma20'],
                    'volume_surge': latest['volume_ratio'] > 1.5,
                    'rsi_not_overbought': latest['rsi'] < 70,
                    'recent_consolidation': prev_5d['close'].std() / prev_5d['close'].mean() < 0.05
                }
                
                score = sum(conditions.values())
                
                if score >= 3:  # 满足至少3个条件
                    breakout_stocks.append({
                        'symbol': symbol,
                        'score': score,
                        'price': latest['close'],
                        'volume_ratio': latest['volume_ratio'],
                        'rsi': latest['rsi'],
                        'conditions_met': [k for k, v in conditions.items() if v]
                    })
                    
            except Exception as e:
                print(f"分析 {symbol} 失败: {e}")
                continue
        
        return sorted(breakout_stocks, key=lambda x: x['score'], reverse=True)
    
    # 应用策略
    test_symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", 
                   "000858.SZ", "002415.SZ", "300059.SZ", "600519.SH"]
    
    breakout_stocks = find_breakout_stocks(test_symbols)
    
    print(f"发现 {len(breakout_stocks)} 只潜在突破股票:")
    for stock in breakout_stocks:
        print(f"\n📈 {stock['symbol']}")
        print(f"  评分: {stock['score']}/4")
        print(f"  价格: {stock['price']:.2f}")
        print(f"  量比: {stock['volume_ratio']:.2f}")
        print(f"  RSI: {stock['rsi']:.1f}")
        print(f"  满足条件: {', '.join(stock['conditions_met'])}")


if __name__ == "__main__":
    try:
        # 高级分析演示
        demo_advanced_analysis()
        
        # 自定义策略演示
        demo_custom_strategy()
        
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()