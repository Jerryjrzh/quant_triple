"""
Screening Results Management

This module handles screening results, ranking, sorting, and analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class RankingMethod(str, Enum):
    """Ranking methods for screening results."""
    COMPOSITE_SCORE = "composite_score"
    TECHNICAL_SCORE = "technical_score"
    SEASONAL_SCORE = "seasonal_score"
    INSTITUTIONAL_SCORE = "institutional_score"
    RISK_SCORE = "risk_score"
    PRICE_CHANGE = "price_change"
    VOLUME_RATIO = "volume_ratio"
    MARKET_CAP = "market_cap"


@dataclass
class StockScore:
    """Individual stock scoring breakdown."""
    stock_code: str
    stock_name: str
    composite_score: float
    technical_score: float = 0.0
    seasonal_score: float = 0.0
    institutional_score: float = 0.0
    risk_score: float = 0.0
    
    # Key metrics for display
    current_price: float = 0.0
    price_change_pct: float = 0.0
    volume_ratio: float = 0.0
    market_cap: float = 0.0
    
    # Additional metadata
    sector: str = ""
    industry: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'composite_score': self.composite_score,
            'technical_score': self.technical_score,
            'seasonal_score': self.seasonal_score,
            'institutional_score': self.institutional_score,
            'risk_score': self.risk_score,
            'current_price': self.current_price,
            'price_change_pct': self.price_change_pct,
            'volume_ratio': self.volume_ratio,
            'market_cap': self.market_cap,
            'sector': self.sector,
            'industry': self.industry,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ScreeningResult:
    """Complete screening result with metadata."""
    screening_id: str
    template_name: str
    execution_time: datetime
    total_stocks_screened: int
    stocks_passed: int
    execution_duration_ms: int
    
    # Results
    stock_scores: List[StockScore] = field(default_factory=list)
    
    # Criteria used
    criteria_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    avg_composite_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'screening_id': self.screening_id,
            'template_name': self.template_name,
            'execution_time': self.execution_time.isoformat(),
            'total_stocks_screened': self.total_stocks_screened,
            'stocks_passed': self.stocks_passed,
            'execution_duration_ms': self.execution_duration_ms,
            'stock_scores': [score.to_dict() for score in self.stock_scores],
            'criteria_summary': self.criteria_summary,
            'avg_composite_score': self.avg_composite_score,
            'score_distribution': self.score_distribution
        }
    
    def get_top_stocks(self, n: int = 10) -> List[StockScore]:
        """Get top N stocks by composite score."""
        return sorted(self.stock_scores, key=lambda x: x.composite_score, reverse=True)[:n]
    
    def filter_by_sector(self, sectors: List[str]) -> List[StockScore]:
        """Filter results by sector."""
        return [score for score in self.stock_scores if score.sector in sectors]
    
    def filter_by_score_range(self, min_score: float, max_score: float) -> List[StockScore]:
        """Filter results by composite score range."""
        return [score for score in self.stock_scores 
                if min_score <= score.composite_score <= max_score]


class ScreeningResultAnalyzer:
    """Analyzer for screening results with ranking and sorting capabilities."""
    
    def __init__(self):
        self.ranking_weights = {
            RankingMethod.TECHNICAL_SCORE: 0.25,
            RankingMethod.SEASONAL_SCORE: 0.25,
            RankingMethod.INSTITUTIONAL_SCORE: 0.25,
            RankingMethod.RISK_SCORE: 0.25
        }
    
    def rank_results(self, results: ScreeningResult, 
                    method: RankingMethod = RankingMethod.COMPOSITE_SCORE,
                    order: SortOrder = SortOrder.DESC) -> List[StockScore]:
        """Rank screening results by specified method."""
        
        if method == RankingMethod.COMPOSITE_SCORE:
            key_func = lambda x: x.composite_score
        elif method == RankingMethod.TECHNICAL_SCORE:
            key_func = lambda x: x.technical_score
        elif method == RankingMethod.SEASONAL_SCORE:
            key_func = lambda x: x.seasonal_score
        elif method == RankingMethod.INSTITUTIONAL_SCORE:
            key_func = lambda x: x.institutional_score
        elif method == RankingMethod.RISK_SCORE:
            key_func = lambda x: x.risk_score
        elif method == RankingMethod.PRICE_CHANGE:
            key_func = lambda x: x.price_change_pct
        elif method == RankingMethod.VOLUME_RATIO:
            key_func = lambda x: x.volume_ratio
        elif method == RankingMethod.MARKET_CAP:
            key_func = lambda x: x.market_cap
        else:
            key_func = lambda x: x.composite_score
        
        reverse = (order == SortOrder.DESC)
        return sorted(results.stock_scores, key=key_func, reverse=reverse)
    
    def calculate_composite_score(self, stock_scores: Dict[str, float], 
                                weights: Dict[str, float] = None) -> float:
        """Calculate composite score from individual component scores."""
        if weights is None:
            weights = self.ranking_weights
        
        composite = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type.value in stock_scores:
                composite += stock_scores[score_type.value] * weight
                total_weight += weight
        
        return composite / total_weight if total_weight > 0 else 0.0
    
    def analyze_score_distribution(self, results: ScreeningResult) -> Dict[str, Any]:
        """Analyze the distribution of scores in the results."""
        if not results.stock_scores:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'quartiles': {
                    'q1': 0.0,
                    'q2': 0.0,
                    'q3': 0.0
                },
                'score_ranges': {
                    'excellent': 0,
                    'good': 0,
                    'fair': 0,
                    'poor': 0
                }
            }
        
        scores = [score.composite_score for score in results.stock_scores]
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'quartiles': {
                'q1': np.percentile(scores, 25),
                'q2': np.percentile(scores, 50),
                'q3': np.percentile(scores, 75)
            },
            'score_ranges': {
                'excellent': len([s for s in scores if s >= 80]),
                'good': len([s for s in scores if 60 <= s < 80]),
                'fair': len([s for s in scores if 40 <= s < 60]),
                'poor': len([s for s in scores if s < 40])
            }
        }
    
    def compare_results(self, results1: ScreeningResult, 
                       results2: ScreeningResult) -> Dict[str, Any]:
        """Compare two screening results."""
        
        # Find common stocks
        codes1 = {score.stock_code for score in results1.stock_scores}
        codes2 = {score.stock_code for score in results2.stock_scores}
        common_codes = codes1.intersection(codes2)
        
        # Create lookup dictionaries
        scores1_dict = {score.stock_code: score for score in results1.stock_scores}
        scores2_dict = {score.stock_code: score for score in results2.stock_scores}
        
        # Calculate score changes for common stocks
        score_changes = []
        for code in common_codes:
            change = scores2_dict[code].composite_score - scores1_dict[code].composite_score
            score_changes.append({
                'stock_code': code,
                'stock_name': scores1_dict[code].stock_name,
                'score_change': change,
                'old_score': scores1_dict[code].composite_score,
                'new_score': scores2_dict[code].composite_score
            })
        
        # Sort by absolute score change
        score_changes.sort(key=lambda x: abs(x['score_change']), reverse=True)
        
        return {
            'comparison_summary': {
                'results1_count': len(results1.stock_scores),
                'results2_count': len(results2.stock_scores),
                'common_stocks': len(common_codes),
                'new_stocks': len(codes2 - codes1),
                'removed_stocks': len(codes1 - codes2)
            },
            'score_changes': score_changes[:20],  # Top 20 changes
            'avg_score_change': np.mean([sc['score_change'] for sc in score_changes]) if score_changes else 0,
            'correlation': self._calculate_score_correlation(results1, results2, common_codes)
        }
    
    def _calculate_score_correlation(self, results1: ScreeningResult, 
                                   results2: ScreeningResult, 
                                   common_codes: set) -> float:
        """Calculate correlation between scores of common stocks."""
        if len(common_codes) < 2:
            return 0.0
        
        scores1_dict = {score.stock_code: score.composite_score for score in results1.stock_scores}
        scores2_dict = {score.stock_code: score.composite_score for score in results2.stock_scores}
        
        scores1_common = [scores1_dict[code] for code in common_codes]
        scores2_common = [scores2_dict[code] for code in common_codes]
        
        return np.corrcoef(scores1_common, scores2_common)[0, 1]
    
    def generate_sector_analysis(self, results: ScreeningResult) -> Dict[str, Any]:
        """Generate sector-wise analysis of screening results."""
        if not results.stock_scores:
            return {}
        
        sector_data = {}
        
        for score in results.stock_scores:
            sector = score.sector or "Unknown"
            if sector not in sector_data:
                sector_data[sector] = {
                    'count': 0,
                    'scores': [],
                    'market_cap_total': 0.0
                }
            
            sector_data[sector]['count'] += 1
            sector_data[sector]['scores'].append(score.composite_score)
            sector_data[sector]['market_cap_total'] += score.market_cap
        
        # Calculate sector statistics
        sector_analysis = {}
        for sector, data in sector_data.items():
            scores = data['scores']
            sector_analysis[sector] = {
                'stock_count': data['count'],
                'avg_score': np.mean(scores),
                'median_score': np.median(scores),
                'score_std': np.std(scores),
                'total_market_cap': data['market_cap_total'],
                'avg_market_cap': data['market_cap_total'] / data['count'],
                'percentage_of_results': (data['count'] / len(results.stock_scores)) * 100
            }
        
        return sector_analysis
    
    def create_performance_summary(self, results: ScreeningResult) -> Dict[str, Any]:
        """Create a comprehensive performance summary."""
        
        score_dist = self.analyze_score_distribution(results)
        sector_analysis = self.generate_sector_analysis(results)
        
        return {
            'execution_summary': {
                'screening_id': results.screening_id,
                'template_name': results.template_name,
                'execution_time': results.execution_time.isoformat(),
                'duration_ms': results.execution_duration_ms,
                'total_screened': results.total_stocks_screened,
                'stocks_passed': results.stocks_passed,
                'pass_rate': (results.stocks_passed / results.total_stocks_screened * 100) if results.total_stocks_screened > 0 else 0
            },
            'score_distribution': score_dist,
            'sector_breakdown': sector_analysis,
            'top_performers': [score.to_dict() for score in results.get_top_stocks(10)],
            'criteria_effectiveness': self._analyze_criteria_effectiveness(results)
        }
    
    def _analyze_criteria_effectiveness(self, results: ScreeningResult) -> Dict[str, Any]:
        """Analyze the effectiveness of different screening criteria."""
        if not results.stock_scores:
            return {}
        
        # Calculate average scores for each criteria type
        technical_scores = [score.technical_score for score in results.stock_scores if score.technical_score > 0]
        seasonal_scores = [score.seasonal_score for score in results.stock_scores if score.seasonal_score > 0]
        institutional_scores = [score.institutional_score for score in results.stock_scores if score.institutional_score > 0]
        risk_scores = [score.risk_score for score in results.stock_scores if score.risk_score > 0]
        
        return {
            'technical_criteria': {
                'avg_score': np.mean(technical_scores) if technical_scores else 0,
                'stocks_with_score': len(technical_scores),
                'effectiveness': np.mean(technical_scores) / 100 if technical_scores else 0
            },
            'seasonal_criteria': {
                'avg_score': np.mean(seasonal_scores) if seasonal_scores else 0,
                'stocks_with_score': len(seasonal_scores),
                'effectiveness': np.mean(seasonal_scores) / 100 if seasonal_scores else 0
            },
            'institutional_criteria': {
                'avg_score': np.mean(institutional_scores) if institutional_scores else 0,
                'stocks_with_score': len(institutional_scores),
                'effectiveness': np.mean(institutional_scores) / 100 if institutional_scores else 0
            },
            'risk_criteria': {
                'avg_score': np.mean(risk_scores) if risk_scores else 0,
                'stocks_with_score': len(risk_scores),
                'effectiveness': np.mean(risk_scores) / 100 if risk_scores else 0
            }
        }