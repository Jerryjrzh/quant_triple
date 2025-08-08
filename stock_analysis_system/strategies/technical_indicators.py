"""
Comprehensive Technical Indicator Library

Advanced technical indicators for quantitative analysis including:
- Trend indicators (MA, EMA, MACD, ADX)
- Momentum indicators (RSI, Stochastic, Williams %R)
- Volatility indicators (Bollinger Bands, ATR, VIX)
- Volume indicators (OBV, VWAP, MFI)
- Custom composite indicators
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    ma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    williams_period: int = 14
    obv_period: int = 20
    mfi_period: int = 14
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]


class TechnicalIndicatorLibrary:
    """Comprehensive technical indicator calculation library"""
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.scaler = StandardScaler()
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given data"""
        if not self._validate_data(data):
            raise ValueError("Invalid data format. Required columns: open, high, low, close, volume")
        
        result = data.copy()
        
        # Trend indicators
        result = self._add_trend_indicators(result)
        
        # Momentum indicators
        result = self._add_momentum_indicators(result)
        
        # Volatility indicators
        result = self._add_volatility_indicators(result)
        
        # Volume indicators
        result = self._add_volume_indicators(result)
        
        # Custom composite indicators
        result = self._add_composite_indicators(result)
        
        # Pattern recognition
        result = self._add_pattern_indicators(result)
        
        return result
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Moving Averages
        for period in self.config.ma_periods:
            data[f'ma_{period}'] = talib.SMA(close, timeperiod=period)
            data[f'ma_{period}_signal'] = np.where(close > data[f'ma_{period}'], 1, -1)
        
        # Exponential Moving Averages
        for period in self.config.ema_periods:
            data[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            data[f'ema_{period}_signal'] = np.where(close > data[f'ema_{period}'], 1, -1)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, 
            fastperiod=self.config.macd_fast,
            slowperiod=self.config.macd_slow,
            signalperiod=self.config.macd_signal
        )
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_histogram'] = macd_hist
        data['macd_bullish'] = np.where(macd > macd_signal, 1, -1)
        
        # ADX (Average Directional Index)
        data['adx'] = talib.ADX(high, low, close, timeperiod=self.config.adx_period)
        data['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.config.adx_period)
        data['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.config.adx_period)
        data['adx_trend_strength'] = np.where(data['adx'] > 25, 1, 0)
        
        # Parabolic SAR
        data['sar'] = talib.SAR(high, low)
        data['sar_signal'] = np.where(close > data['sar'], 1, -1)
        
        # Ichimoku Cloud components
        data['tenkan_sen'] = self._calculate_ichimoku_line(data, 9)
        data['kijun_sen'] = self._calculate_ichimoku_line(data, 26)
        data['senkou_span_a'] = (data['tenkan_sen'] + data['kijun_sen']) / 2
        data['senkou_span_b'] = self._calculate_ichimoku_line(data, 52)
        data['chikou_span'] = close
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # RSI
        data['rsi'] = talib.RSI(close, timeperiod=self.config.rsi_period)
        data['rsi_overbought'] = np.where(data['rsi'] > 70, 1, 0)
        data['rsi_oversold'] = np.where(data['rsi'] < 30, 1, 0)
        data['rsi_signal'] = np.where(data['rsi'] > 50, 1, -1)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.config.stoch_k,
            slowk_period=self.config.stoch_d,
            slowd_period=self.config.stoch_d
        )
        data['stoch_k'] = slowk
        data['stoch_d'] = slowd
        data['stoch_overbought'] = np.where(slowk > 80, 1, 0)
        data['stoch_oversold'] = np.where(slowk < 20, 1, 0)
        
        # Williams %R
        data['williams_r'] = talib.WILLR(high, low, close, timeperiod=self.config.williams_period)
        data['williams_overbought'] = np.where(data['williams_r'] > -20, 1, 0)
        data['williams_oversold'] = np.where(data['williams_r'] < -80, 1, 0)
        
        # CCI (Commodity Channel Index)
        data['cci'] = talib.CCI(high, low, close, timeperiod=14)
        data['cci_overbought'] = np.where(data['cci'] > 100, 1, 0)
        data['cci_oversold'] = np.where(data['cci'] < -100, 1, 0)
        
        # ROC (Rate of Change)
        data['roc'] = talib.ROC(close, timeperiod=10)
        data['roc_signal'] = np.where(data['roc'] > 0, 1, -1)
        
        # Momentum
        data['momentum'] = talib.MOM(close, timeperiod=10)
        data['momentum_signal'] = np.where(data['momentum'] > 0, 1, -1)
        
        return data
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=self.config.bb_period,
            nbdevup=self.config.bb_std,
            nbdevdn=self.config.bb_std
        )
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        data['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        data['bb_squeeze'] = np.where(data['bb_width'] < data['bb_width'].rolling(20).mean(), 1, 0)
        
        # ATR (Average True Range)
        data['atr'] = talib.ATR(high, low, close, timeperiod=self.config.atr_period)
        data['atr_normalized'] = data['atr'] / close
        
        # Keltner Channels
        ema_20 = talib.EMA(close, timeperiod=20)
        data['keltner_upper'] = ema_20 + (2 * data['atr'])
        data['keltner_lower'] = ema_20 - (2 * data['atr'])
        data['keltner_position'] = (close - data['keltner_lower']) / (data['keltner_upper'] - data['keltner_lower'])
        
        # Donchian Channels
        data['donchian_upper'] = talib.MAX(high, timeperiod=20)
        data['donchian_lower'] = talib.MIN(low, timeperiod=20)
        data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
        
        # Historical Volatility
        returns = np.log(close / np.roll(close, 1))
        data['hist_volatility'] = pd.Series(returns).rolling(20).std() * np.sqrt(252)
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # OBV (On-Balance Volume)
        data['obv'] = talib.OBV(close, volume)
        data['obv_ma'] = talib.SMA(data['obv'].values, timeperiod=self.config.obv_period)
        data['obv_signal'] = np.where(data['obv'] > data['obv_ma'], 1, -1)
        
        # VWAP (Volume Weighted Average Price)
        data['vwap'] = self._calculate_vwap(data)
        data['vwap_signal'] = np.where(close > data['vwap'], 1, -1)
        
        # MFI (Money Flow Index)
        data['mfi'] = talib.MFI(high, low, close, volume, timeperiod=self.config.mfi_period)
        data['mfi_overbought'] = np.where(data['mfi'] > 80, 1, 0)
        data['mfi_oversold'] = np.where(data['mfi'] < 20, 1, 0)
        
        # A/D Line (Accumulation/Distribution)
        data['ad_line'] = talib.AD(high, low, close, volume)
        data['ad_line_ma'] = talib.SMA(data['ad_line'].values, timeperiod=20)
        data['ad_signal'] = np.where(data['ad_line'] > data['ad_line_ma'], 1, -1)
        
        # Chaikin Oscillator
        data['chaikin_osc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        data['chaikin_signal'] = np.where(data['chaikin_osc'] > 0, 1, -1)
        
        # Volume Rate of Change
        data['volume_roc'] = talib.ROC(volume.astype(float), timeperiod=10)
        
        # Price Volume Trend
        data['pvt'] = self._calculate_pvt(data)
        
        return data
    
    def _add_composite_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators"""
        close = data['close'].values
        
        # Trend Strength Composite
        trend_signals = []
        if 'ma_20' in data.columns:
            trend_signals.append(data['ma_20_signal'])
        if 'ema_12' in data.columns:
            trend_signals.append(data['ema_12_signal'])
        if 'macd_bullish' in data.columns:
            trend_signals.append(data['macd_bullish'])
        if 'adx_trend_strength' in data.columns:
            trend_signals.append(data['adx_trend_strength'])
        
        if trend_signals:
            data['trend_strength'] = np.mean(trend_signals, axis=0)
        
        # Momentum Composite
        momentum_signals = []
        if 'rsi_signal' in data.columns:
            momentum_signals.append(data['rsi_signal'])
        if 'roc_signal' in data.columns:
            momentum_signals.append(data['roc_signal'])
        if 'momentum_signal' in data.columns:
            momentum_signals.append(data['momentum_signal'])
        
        if momentum_signals:
            data['momentum_composite'] = np.mean(momentum_signals, axis=0)
        
        # Volume Strength Composite
        volume_signals = []
        if 'obv_signal' in data.columns:
            volume_signals.append(data['obv_signal'])
        if 'vwap_signal' in data.columns:
            volume_signals.append(data['vwap_signal'])
        if 'ad_signal' in data.columns:
            volume_signals.append(data['ad_signal'])
        
        if volume_signals:
            data['volume_strength'] = np.mean(volume_signals, axis=0)
        
        # Overall Technical Score
        if all(col in data.columns for col in ['trend_strength', 'momentum_composite', 'volume_strength']):
            data['technical_score'] = (
                data['trend_strength'] * 0.4 +
                data['momentum_composite'] * 0.35 +
                data['volume_strength'] * 0.25
            )
        
        return data
    
    def _add_pattern_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Major candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing_bullish': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'harami': talib.CDLHARAMI,
            'piercing': talib.CDLPIERCING,
            'dark_cloud': talib.CDLDARKCLOUDCOVER
        }
        
        pattern_signals = []
        for pattern_name, pattern_func in patterns.items():
            try:
                pattern_result = pattern_func(open_price, high, low, close)
                data[f'pattern_{pattern_name}'] = pattern_result
                # Convert pattern signals to binary (bullish=1, bearish=-1, neutral=0)
                pattern_signals.append(np.sign(pattern_result))
            except Exception as e:
                logger.warning(f"Failed to calculate pattern {pattern_name}: {e}")
        
        # Composite pattern signal
        if pattern_signals:
            data['pattern_composite'] = np.mean(pattern_signals, axis=0)
        
        return data
    
    def _calculate_ichimoku_line(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ichimoku line (Tenkan-sen, Kijun-sen, etc.)"""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        return (high_max + low_min) / 2
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    def _calculate_pvt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Price Volume Trend"""
        price_change = data['close'].pct_change()
        return (price_change * data['volume']).cumsum()
    
    def get_signal_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all technical signals"""
        if data.empty:
            return {}
        
        latest_data = data.iloc[-1]
        
        # Trend signals
        trend_signals = {}
        for col in data.columns:
            if col.endswith('_signal') and 'trend' in col.lower():
                trend_signals[col] = latest_data[col]
        
        # Momentum signals
        momentum_signals = {}
        for col in data.columns:
            if any(indicator in col for indicator in ['rsi', 'stoch', 'williams', 'cci']):
                if col.endswith('_signal') or col.endswith('_overbought') or col.endswith('_oversold'):
                    momentum_signals[col] = latest_data[col]
        
        # Volume signals
        volume_signals = {}
        for col in data.columns:
            if any(indicator in col for indicator in ['obv', 'vwap', 'mfi', 'ad']):
                if col.endswith('_signal'):
                    volume_signals[col] = latest_data[col]
        
        # Composite scores
        composite_scores = {}
        for col in ['trend_strength', 'momentum_composite', 'volume_strength', 'technical_score']:
            if col in data.columns:
                composite_scores[col] = latest_data[col]
        
        return {
            'trend_signals': trend_signals,
            'momentum_signals': momentum_signals,
            'volume_signals': volume_signals,
            'composite_scores': composite_scores,
            'overall_signal': self._calculate_overall_signal(latest_data)
        }
    
    def _calculate_overall_signal(self, latest_data: pd.Series) -> str:
        """Calculate overall trading signal"""
        if 'technical_score' in latest_data:
            score = latest_data['technical_score']
            if score > 0.3:
                return 'STRONG_BUY'
            elif score > 0.1:
                return 'BUY'
            elif score > -0.1:
                return 'HOLD'
            elif score > -0.3:
                return 'SELL'
            else:
                return 'STRONG_SELL'
        
        return 'NEUTRAL'
    
    def backtest_indicator(self, data: pd.DataFrame, indicator_column: str, 
                          signal_column: str, initial_capital: float = 10000) -> Dict[str, Any]:
        """Backtest a single indicator strategy"""
        if signal_column not in data.columns:
            raise ValueError(f"Signal column {signal_column} not found in data")
        
        # Simple backtest logic
        positions = data[signal_column].shift(1)  # Use previous signal
        returns = data['close'].pct_change()
        strategy_returns = positions * returns
        
        # Calculate performance metrics
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0])
        
        return {
            'indicator': indicator_column,
            'signal_column': signal_column,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(strategy_returns[strategy_returns != 0]),
            'cumulative_returns': cumulative_returns
        }