"""
Multi-Factor Screening Engine

This module implements the core screening engine that evaluates stocks
against multiple criteria types and generates composite scores.
"""

from typing import Dict, List, Optional, Any, Tuple
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import uuid
from dataclasses import dataclass

from .screening_criteria import ScreeningTemplate, LogicalOperator
from .screening_results import ScreeningResult, StockScore
from ..data.data_source_manager import DataSourceManager
from ..analysis.spring_festival_engine import SpringFestivalAlignmentEngine
from ..analysis.institutional_attention_scoring import InstitutionalAttentionScoringSystem
from ..analysis.risk_management_engine import EnhancedRiskManagementEngine


class ScreeningEngine:
    """
    Core screening engine that evaluates stocks against multiple criteria
    and generates ranked results with composite scoring.
    """
    
    def __init__(self, data_source_manager: DataSourceManager,
                 spring_festival_engine: SpringFestivalAlignmentEngine,
                 institutional_engine: InstitutionalAttentionScoringSystem,
                 risk_engine: EnhancedRiskManagementEngine):
        
        self.data_source = data_source_manager
        self.spring_festival_engine = spring_festival_engine
        self.institutional_engine = institutional_engine
        self.risk_engine = risk_engine
        
        # Scoring weights for composite score calculation
        self.default_weights = {
            'technical': 0.3,
            'seasonal': 0.25,
            'institutional': 0.25,
            'risk': 0.2
        }
        
        # Performance optimization
        self.batch_size = 50
        self.max_concurrent_evaluations = 10
    
    async def execute_screening(self, template: ScreeningTemplate,
                              stock_universe: List[str] = None,
                              max_results: int = 100,
                              screening_id: str = None) -> ScreeningResult:
        """
        Execute screening using the provided template.
        
        Args:
            template: Screening template with criteria
            stock_universe: List of stock codes to screen (None for all)
            max_results: Maximum number of results to return
            screening_id: Unique screening identifier
            
        Returns:
            ScreeningResult with scored and ranked stocks
        """
        start_time = datetime.now()
        screening_id = screening_id or str(uuid.uuid4())
        
        # Get stock universe
        if stock_universe is None:
            stock_universe = await self._get_default_stock_universe()
        
        total_stocks = len(stock_universe)
        
        # Process stocks in batches for better performance
        all_scores = []
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        
        for i in range(0, len(stock_universe), self.batch_size):
            batch = stock_universe[i:i + self.batch_size]
            batch_tasks = [
                self._evaluate_stock_with_semaphore(semaphore, stock_code, template)
                for stock_code in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_scores = [
                result for result in batch_results
                if not isinstance(result, Exception) and result is not None
            ]
            
            all_scores.extend(valid_scores)
        
        # Sort by composite score and limit results
        all_scores.sort(key=lambda x: x.composite_score, reverse=True)
        final_scores = all_scores[:max_results]
        
        # Calculate execution metrics
        end_time = datetime.now()
        execution_duration = int((end_time - start_time).total_seconds() * 1000)
        
        # Create result object
        result = ScreeningResult(
            screening_id=screening_id,
            template_name=template.name,
            execution_time=start_time,
            total_stocks_screened=total_stocks,
            stocks_passed=len(final_scores),
            execution_duration_ms=execution_duration,
            stock_scores=final_scores,
            criteria_summary=self._create_criteria_summary(template),
            avg_composite_score=np.mean([s.composite_score for s in final_scores]) if final_scores else 0.0
        )
        
        # Calculate score distribution
        result.score_distribution = self._calculate_score_distribution(final_scores)
        
        return result
    
    async def _evaluate_stock_with_semaphore(self, semaphore: asyncio.Semaphore,
                                           stock_code: str,
                                           template: ScreeningTemplate) -> Optional[StockScore]:
        """Evaluate a single stock with concurrency control."""
        async with semaphore:
            return await self._evaluate_single_stock(stock_code, template)
    
    async def _evaluate_single_stock(self, stock_code: str,
                                   template: ScreeningTemplate) -> Optional[StockScore]:
        """
        Evaluate a single stock against all criteria in the template.
        
        Args:
            stock_code: Stock code to evaluate
            template: Screening template with criteria
            
        Returns:
            StockScore object or None if stock doesn't pass screening
        """
        try:
            # Get basic stock data
            stock_data = await self._get_stock_data(stock_code)
            if stock_data is None:
                return None
            
            # Initialize scores
            technical_score = 0.0
            seasonal_score = 0.0
            institutional_score = 0.0
            risk_score = 0.0
            
            # Evaluate technical criteria
            if template.technical_criteria and template.technical_criteria.enabled:
                technical_score = await self._evaluate_technical_criteria(
                    stock_code, stock_data, template.technical_criteria
                )
                if technical_score == 0.0 and template.logical_operator == LogicalOperator.AND:
                    return None
            
            # Evaluate seasonal criteria
            if template.seasonal_criteria and template.seasonal_criteria.enabled:
                seasonal_score = await self._evaluate_seasonal_criteria(
                    stock_code, stock_data, template.seasonal_criteria
                )
                if seasonal_score == 0.0 and template.logical_operator == LogicalOperator.AND:
                    return None
            
            # Evaluate institutional criteria
            if template.institutional_criteria and template.institutional_criteria.enabled:
                institutional_score = await self._evaluate_institutional_criteria(
                    stock_code, stock_data, template.institutional_criteria
                )
                if institutional_score == 0.0 and template.logical_operator == LogicalOperator.AND:
                    return None
            
            # Evaluate risk criteria
            if template.risk_criteria and template.risk_criteria.enabled:
                risk_score = await self._evaluate_risk_criteria(
                    stock_code, stock_data, template.risk_criteria
                )
                if risk_score == 0.0 and template.logical_operator == LogicalOperator.AND:
                    return None
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                technical_score, seasonal_score, institutional_score, risk_score, template
            )
            
            # Apply minimum composite score threshold
            if composite_score < 10.0:  # Minimum threshold
                return None
            
            # Create stock score object
            return StockScore(
                stock_code=stock_code,
                stock_name=stock_data.get('stock_name', stock_code),
                composite_score=composite_score,
                technical_score=technical_score,
                seasonal_score=seasonal_score,
                institutional_score=institutional_score,
                risk_score=risk_score,
                current_price=stock_data.get('current_price', 0.0),
                price_change_pct=stock_data.get('price_change_pct', 0.0),
                volume_ratio=stock_data.get('volume_ratio', 0.0),
                market_cap=stock_data.get('market_cap', 0.0),
                sector=stock_data.get('sector', ''),
                industry=stock_data.get('industry', ''),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            print(f"Error evaluating stock {stock_code}: {e}")
            return None
    
    async def _evaluate_technical_criteria(self, stock_code: str, stock_data: Dict,
                                         criteria) -> float:
        """Evaluate technical analysis criteria."""
        score = 0.0
        max_score = 100.0
        criteria_count = 0
        
        try:
            # Get technical indicators
            indicators = await self._calculate_technical_indicators(stock_code, stock_data)
            
            # Price criteria
            if criteria.price_min is not None or criteria.price_max is not None:
                price = stock_data.get('current_price', 0)
                if criteria.price_min and price < criteria.price_min:
                    return 0.0
                if criteria.price_max and price > criteria.price_max:
                    return 0.0
                score += 20.0
                criteria_count += 1
            
            # Price change criteria
            if criteria.price_change_pct_min is not None or criteria.price_change_pct_max is not None:
                price_change = stock_data.get('price_change_pct', 0)
                if criteria.price_change_pct_min and price_change < criteria.price_change_pct_min:
                    return 0.0
                if criteria.price_change_pct_max and price_change > criteria.price_change_pct_max:
                    return 0.0
                score += 15.0
                criteria_count += 1
            
            # Volume criteria
            if criteria.volume_min is not None:
                volume = stock_data.get('volume', 0)
                if volume < criteria.volume_min:
                    return 0.0
                score += 10.0
                criteria_count += 1
            
            if criteria.volume_avg_ratio_min is not None:
                volume_ratio = stock_data.get('volume_ratio', 0)
                if volume_ratio < criteria.volume_avg_ratio_min:
                    return 0.0
                score += 15.0
                criteria_count += 1
            
            # Moving average criteria
            ma_score = self._evaluate_ma_criteria(indicators, criteria)
            if ma_score == 0.0:
                return 0.0
            score += ma_score
            criteria_count += 1
            
            # RSI criteria
            if criteria.rsi_min is not None or criteria.rsi_max is not None:
                rsi = indicators.get('rsi', 50)
                if criteria.rsi_min and rsi < criteria.rsi_min:
                    return 0.0
                if criteria.rsi_max and rsi > criteria.rsi_max:
                    return 0.0
                score += 20.0
                criteria_count += 1
            
            # MACD criteria
            if criteria.macd_signal:
                macd_score = self._evaluate_macd_signal(indicators, criteria.macd_signal)
                if macd_score == 0.0:
                    return 0.0
                score += macd_score
                criteria_count += 1
            
            # Normalize score
            if criteria_count > 0:
                return min(max_score, (score / criteria_count) * (max_score / 100.0))
            
            return max_score  # No criteria means full score
            
        except Exception as e:
            print(f"Error evaluating technical criteria for {stock_code}: {e}")
            return 0.0
    
    async def _evaluate_seasonal_criteria(self, stock_code: str, stock_data: Dict,
                                        criteria) -> float:
        """Evaluate seasonal/Spring Festival criteria."""
        try:
            # Get Spring Festival analysis
            current_position = self.spring_festival_engine.get_current_position(stock_code)
            sf_analysis = {
                'current_sf_position': current_position.get('days_to_spring_festival', 0),
                'pattern_strength': current_position.get('pattern_strength', 0.5),
                'historical_performance': 'neutral',
                'pattern_confidence': current_position.get('confidence_level', 0.5)
            }
            
            if not sf_analysis:
                return 0.0
            
            score = 0.0
            criteria_count = 0
            
            # Spring Festival position criteria
            if criteria.spring_festival_days_range:
                current_sf_position = sf_analysis.get('current_sf_position', 0)
                min_days, max_days = criteria.spring_festival_days_range
                
                if min_days <= current_sf_position <= max_days:
                    score += 30.0
                else:
                    return 0.0
                criteria_count += 1
            
            # Pattern strength criteria
            if criteria.spring_festival_pattern_strength:
                pattern_strength = sf_analysis.get('pattern_strength', 0)
                if pattern_strength >= criteria.spring_festival_pattern_strength:
                    score += 25.0
                else:
                    return 0.0
                criteria_count += 1
            
            # Historical performance criteria
            if criteria.spring_festival_historical_performance:
                historical_perf = sf_analysis.get('historical_performance', 'neutral')
                if historical_perf == criteria.spring_festival_historical_performance:
                    score += 25.0
                elif criteria.spring_festival_historical_performance == 'strong' and historical_perf != 'strong':
                    return 0.0
                criteria_count += 1
            
            # Pattern confidence criteria
            if criteria.pattern_confidence_min:
                confidence = sf_analysis.get('pattern_confidence', 0)
                if confidence >= criteria.pattern_confidence_min:
                    score += 20.0
                else:
                    return 0.0
                criteria_count += 1
            
            # Normalize score
            if criteria_count > 0:
                return min(100.0, (score / criteria_count) * (100.0 / 25.0))
            
            return 100.0  # No criteria means full score
            
        except Exception as e:
            print(f"Error evaluating seasonal criteria for {stock_code}: {e}")
            return 0.0
    
    async def _evaluate_institutional_criteria(self, stock_code: str, stock_data: Dict,
                                             criteria) -> float:
        """Evaluate institutional activity criteria."""
        try:
            # Get institutional analysis with required date parameters
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            inst_analysis = await self.institutional_engine.calculate_stock_attention_profile(
                stock_code, start_date, end_date
            )
            
            if not inst_analysis:
                return 0.0
            
            score = 0.0
            criteria_count = 0
            
            # Attention score criteria
            if criteria.attention_score_min is not None or criteria.attention_score_max is not None:
                attention_score = inst_analysis.overall_attention_score if inst_analysis else 0
                
                if criteria.attention_score_min and attention_score < criteria.attention_score_min:
                    return 0.0
                if criteria.attention_score_max and attention_score > criteria.attention_score_max:
                    return 0.0
                
                score += min(30.0, attention_score * 0.3)  # Scale to 30 points max
                criteria_count += 1
            
            # New institutional entry (simplified check)
            if criteria.new_institutional_entry:
                has_new_entry = inst_analysis and inst_analysis.overall_attention_score > 60
                if has_new_entry:
                    score += 25.0
                else:
                    return 0.0
                criteria_count += 1
            
            # Dragon-tiger list appearances (simplified)
            if criteria.dragon_tiger_appearances:
                dt_appearances = len(inst_analysis.institution_scores) if inst_analysis else 0
                if dt_appearances >= criteria.dragon_tiger_appearances:
                    score += 20.0
                else:
                    return 0.0
                criteria_count += 1
            
            # Fund type activity (simplified)
            fund_activity_score = 0
            fund_criteria_count = 0
            
            if criteria.mutual_fund_activity:
                has_activity = inst_analysis and inst_analysis.overall_attention_score > 50
                if has_activity:
                    fund_activity_score += 25
                else:
                    return 0.0
                fund_criteria_count += 1
            
            if criteria.qfii_activity:
                has_activity = inst_analysis and inst_analysis.overall_attention_score > 50
                if has_activity:
                    fund_activity_score += 25
                else:
                    return 0.0
                fund_criteria_count += 1
            
            if fund_criteria_count > 0:
                score += fund_activity_score / fund_criteria_count
                criteria_count += 1
            
            # Normalize score
            if criteria_count > 0:
                return min(100.0, score / criteria_count * 4)  # Scale to 100
            
            return 100.0  # No criteria means full score
            
        except Exception as e:
            print(f"Error evaluating institutional criteria for {stock_code}: {e}")
            return 0.0
    
    async def _evaluate_risk_criteria(self, stock_code: str, stock_data: Dict,
                                    criteria) -> float:
        """Evaluate risk management criteria."""
        try:
            # Get historical data for risk calculation
            historical_data = await self.data_source.get_stock_historical_data(stock_code, 60)
            if historical_data is None or len(historical_data) < 20:
                return 0.0
            
            # Get risk metrics with proper DataFrame format
            risk_metrics = await self.risk_engine.calculate_comprehensive_risk_metrics(
                price_data=historical_data
            )
            
            if not risk_metrics:
                return 0.0
            
            score = 100.0  # Start with full score, deduct for violations
            criteria_count = 0
            
            # Volatility criteria
            if criteria.volatility_min is not None or criteria.volatility_max is not None:
                volatility = getattr(risk_metrics, 'volatility', 0) if hasattr(risk_metrics, 'volatility') else 0
                
                if criteria.volatility_min and volatility < criteria.volatility_min:
                    return 0.0
                if criteria.volatility_max and volatility > criteria.volatility_max:
                    return 0.0
                
                criteria_count += 1
            
            # VaR criteria
            if criteria.var_max is not None:
                var = getattr(risk_metrics, 'var_95', 0) if hasattr(risk_metrics, 'var_95') else 0
                if var > criteria.var_max:
                    return 0.0
                criteria_count += 1
            
            # Sharpe ratio criteria
            if criteria.sharpe_ratio_min is not None:
                sharpe = getattr(risk_metrics, 'sharpe_ratio', 0) if hasattr(risk_metrics, 'sharpe_ratio') else 0
                if sharpe < criteria.sharpe_ratio_min:
                    return 0.0
                criteria_count += 1
            
            # Beta criteria
            if criteria.beta_min is not None or criteria.beta_max is not None:
                beta = getattr(risk_metrics, 'beta', 1.0) if hasattr(risk_metrics, 'beta') else 1.0
                
                if criteria.beta_min and beta < criteria.beta_min:
                    return 0.0
                if criteria.beta_max and beta > criteria.beta_max:
                    return 0.0
                
                criteria_count += 1
            
            # Drawdown criteria
            if criteria.max_drawdown_max is not None:
                max_dd = getattr(risk_metrics, 'max_drawdown', 0) if hasattr(risk_metrics, 'max_drawdown') else 0
                if max_dd > criteria.max_drawdown_max:
                    return 0.0
                criteria_count += 1
            
            return score if criteria_count > 0 else 100.0
            
        except Exception as e:
            print(f"Error evaluating risk criteria for {stock_code}: {e}")
            return 0.0
    
    def _calculate_composite_score(self, technical_score: float, seasonal_score: float,
                                 institutional_score: float, risk_score: float,
                                 template: ScreeningTemplate) -> float:
        """Calculate composite score from individual component scores."""
        
        # Get weights from template or use defaults
        weights = self.default_weights.copy()
        
        # Adjust weights based on which criteria are enabled
        active_criteria = []
        if template.technical_criteria and template.technical_criteria.enabled:
            active_criteria.append('technical')
        if template.seasonal_criteria and template.seasonal_criteria.enabled:
            active_criteria.append('seasonal')
        if template.institutional_criteria and template.institutional_criteria.enabled:
            active_criteria.append('institutional')
        if template.risk_criteria and template.risk_criteria.enabled:
            active_criteria.append('risk')
        
        if not active_criteria:
            return 0.0
        
        # Redistribute weights among active criteria
        total_weight = sum(weights[criteria] for criteria in active_criteria)
        for criteria in active_criteria:
            weights[criteria] = weights[criteria] / total_weight
        
        # Calculate weighted composite score
        composite = 0.0
        if 'technical' in active_criteria:
            composite += technical_score * weights['technical']
        if 'seasonal' in active_criteria:
            composite += seasonal_score * weights['seasonal']
        if 'institutional' in active_criteria:
            composite += institutional_score * weights['institutional']
        if 'risk' in active_criteria:
            composite += risk_score * weights['risk']
        
        return min(100.0, max(0.0, composite))
    
    async def _get_stock_data(self, stock_code: str) -> Optional[Dict]:
        """Get basic stock data for evaluation."""
        try:
            # This would integrate with the data source manager
            # For now, return mock data structure
            return {
                'stock_code': stock_code,
                'stock_name': f"Stock {stock_code}",
                'current_price': 10.0,
                'price_change_pct': 2.5,
                'volume': 1000000,
                'volume_ratio': 1.2,
                'market_cap': 1000000000,
                'sector': 'Technology',
                'industry': 'Software'
            }
        except Exception as e:
            print(f"Error getting stock data for {stock_code}: {e}")
            return None
    
    async def _calculate_technical_indicators(self, stock_code: str, stock_data: Dict) -> Dict:
        """Calculate technical indicators for the stock."""
        # This would integrate with technical analysis modules
        # For now, return mock indicators
        return {
            'rsi': 65.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'ma5': 9.8,
            'ma10': 9.5,
            'ma20': 9.2,
            'ma50': 8.8,
            'ma200': 8.0,
            'bollinger_upper': 10.5,
            'bollinger_lower': 9.5,
            'momentum_20': 5.2
        }
    
    def _evaluate_ma_criteria(self, indicators: Dict, criteria) -> float:
        """Evaluate moving average criteria."""
        score = 0.0
        current_price = 10.0  # Would get from stock data
        
        ma_checks = [
            ('ma5_position', indicators.get('ma5', 0)),
            ('ma10_position', indicators.get('ma10', 0)),
            ('ma20_position', indicators.get('ma20', 0)),
            ('ma50_position', indicators.get('ma50', 0)),
            ('ma200_position', indicators.get('ma200', 0))
        ]
        
        for attr_name, ma_value in ma_checks:
            position = getattr(criteria, attr_name, None)
            if position:
                if position == 'above' and current_price > ma_value:
                    score += 20.0
                elif position == 'below' and current_price < ma_value:
                    score += 20.0
                elif position in ['above', 'below']:
                    return 0.0  # Failed criteria
        
        return score
    
    def _evaluate_macd_signal(self, indicators: Dict, signal_type: str) -> float:
        """Evaluate MACD signal criteria."""
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        
        if signal_type == 'bullish' and macd > macd_signal:
            return 25.0
        elif signal_type == 'bearish' and macd < macd_signal:
            return 25.0
        elif signal_type == 'neutral':
            return 15.0
        
        return 0.0
    
    async def _get_default_stock_universe(self) -> List[str]:
        """Get default stock universe for screening."""
        # This would integrate with data source to get all available stocks
        # For now, return a sample universe
        return [f"{i:06d}" for i in range(1, 1001)]  # Sample 1000 stocks
    
    def _create_criteria_summary(self, template: ScreeningTemplate) -> Dict[str, Any]:
        """Create summary of criteria used in screening."""
        return {
            'template_name': template.name,
            'has_technical': template.technical_criteria is not None and template.technical_criteria.enabled,
            'has_seasonal': template.seasonal_criteria is not None and template.seasonal_criteria.enabled,
            'has_institutional': template.institutional_criteria is not None and template.institutional_criteria.enabled,
            'has_risk': template.risk_criteria is not None and template.risk_criteria.enabled,
            'logical_operator': template.logical_operator.value
        }
    
    def _calculate_score_distribution(self, scores: List[StockScore]) -> Dict[str, int]:
        """Calculate score distribution for results."""
        if not scores:
            return {}
        
        distribution = {
            'excellent': 0,  # 80-100
            'good': 0,       # 60-79
            'fair': 0,       # 40-59
            'poor': 0        # 0-39
        }
        
        for score in scores:
            composite = score.composite_score
            if composite >= 80:
                distribution['excellent'] += 1
            elif composite >= 60:
                distribution['good'] += 1
            elif composite >= 40:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution