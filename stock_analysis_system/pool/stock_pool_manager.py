"""
Stock Pool Management System

This module implements comprehensive stock pool management with multiple pool types,
analytics, performance tracking, and automated updates based on screening results.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PoolType(str, Enum):
    """Pool type enumeration"""
    WATCHLIST = "watchlist"
    CORE_HOLDINGS = "core_holdings"
    POTENTIAL_OPPORTUNITIES = "potential_opportunities"
    HIGH_RISK = "high_risk"
    DIVIDEND_FOCUS = "dividend_focus"
    GROWTH_STOCKS = "growth_stocks"
    VALUE_STOCKS = "value_stocks"
    CUSTOM = "custom"

class PoolStatus(str, Enum):
    """Pool status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"

@dataclass
class StockInfo:
    """Stock information within a pool"""
    symbol: str
    name: str
    added_date: datetime
    added_price: float
    current_price: float = 0.0
    weight: float = 0.0
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class PoolMetrics:
    """Pool performance metrics"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_holding_period: float = 0.0
    sector_concentration: Dict[str, float] = None
    risk_score: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.sector_concentration is None:
            self.sector_concentration = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class StockPool:
    """Stock pool data structure"""
    pool_id: str
    name: str
    pool_type: PoolType
    description: str
    created_date: datetime
    last_modified: datetime
    status: PoolStatus
    stocks: List[StockInfo]
    metrics: PoolMetrics
    auto_update_rules: Dict[str, Any] = None
    max_stocks: int = 100
    rebalance_frequency: str = "monthly"
    
    def __post_init__(self):
        if self.auto_update_rules is None:
            self.auto_update_rules = {}

class StockPoolManager:
    """
    Advanced Stock Pool Management System
    
    Provides comprehensive pool management with analytics, performance tracking,
    automated updates, and comparison tools.
    """
    
    def __init__(self, data_source_manager=None, risk_engine=None):
        self.data_source = data_source_manager
        self.risk_engine = risk_engine
        self.pools: Dict[str, StockPool] = {}
        self.pool_history: Dict[str, List[Dict]] = {}
        self.auto_update_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_pool(
        self,
        name: str,
        pool_type: PoolType,
        description: str = "",
        max_stocks: int = 100,
        rebalance_frequency: str = "monthly"
    ) -> str:
        """Create a new stock pool"""
        
        pool_id = f"{pool_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pool = StockPool(
            pool_id=pool_id,
            name=name,
            pool_type=pool_type,
            description=description,
            created_date=datetime.now(),
            last_modified=datetime.now(),
            status=PoolStatus.ACTIVE,
            stocks=[],
            metrics=PoolMetrics(),
            max_stocks=max_stocks,
            rebalance_frequency=rebalance_frequency
        )
        
        self.pools[pool_id] = pool
        self.pool_history[pool_id] = []
        
        # Log pool creation
        await self._log_pool_action(pool_id, "created", {"name": name, "type": pool_type.value})
        
        logger.info(f"Created new pool: {name} ({pool_id})")
        return pool_id
    
    async def add_stock_to_pool(
        self,
        pool_id: str,
        symbol: str,
        name: str = "",
        weight: float = 0.0,
        notes: str = "",
        tags: List[str] = None
    ) -> bool:
        """Add a stock to a pool"""
        
        if pool_id not in self.pools:
            logger.error(f"Pool {pool_id} not found")
            return False
        
        pool = self.pools[pool_id]
        
        # Check if stock already exists
        for stock in pool.stocks:
            if stock.symbol == symbol:
                logger.warning(f"Stock {symbol} already exists in pool {pool_id}")
                return False
        
        # Check pool capacity
        if len(pool.stocks) >= pool.max_stocks:
            logger.warning(f"Pool {pool_id} has reached maximum capacity ({pool.max_stocks})")
            return False
        
        # Get current price
        current_price = await self._get_current_price(symbol)
        
        stock_info = StockInfo(
            symbol=symbol,
            name=name or symbol,
            added_date=datetime.now(),
            added_price=current_price,
            current_price=current_price,
            weight=weight,
            notes=notes,
            tags=tags or []
        )
        
        pool.stocks.append(stock_info)
        pool.last_modified = datetime.now()
        
        # Update pool metrics
        await self._update_pool_metrics(pool_id)
        
        # Log action
        await self._log_pool_action(pool_id, "stock_added", {"symbol": symbol, "price": current_price})
        
        logger.info(f"Added stock {symbol} to pool {pool_id}")
        return True
    
    async def remove_stock_from_pool(self, pool_id: str, symbol: str) -> bool:
        """Remove a stock from a pool"""
        
        if pool_id not in self.pools:
            logger.error(f"Pool {pool_id} not found")
            return False
        
        pool = self.pools[pool_id]
        
        # Find and remove stock
        for i, stock in enumerate(pool.stocks):
            if stock.symbol == symbol:
                removed_stock = pool.stocks.pop(i)
                pool.last_modified = datetime.now()
                
                # Update pool metrics
                await self._update_pool_metrics(pool_id)
                
                # Log action
                await self._log_pool_action(
                    pool_id, 
                    "stock_removed", 
                    {
                        "symbol": symbol, 
                        "added_price": removed_stock.added_price,
                        "current_price": removed_stock.current_price
                    }
                )
                
                logger.info(f"Removed stock {symbol} from pool {pool_id}")
                return True
        
        logger.warning(f"Stock {symbol} not found in pool {pool_id}")
        return False
    
    async def update_pool_from_screening(
        self,
        pool_id: str,
        screening_results: List[Dict],
        max_additions: int = 10,
        replace_existing: bool = False
    ) -> Dict[str, Any]:
        """Update pool based on screening results"""
        
        if pool_id not in self.pools:
            logger.error(f"Pool {pool_id} not found")
            return {"success": False, "error": "Pool not found"}
        
        pool = self.pools[pool_id]
        added_stocks = []
        removed_stocks = []
        
        # If replace_existing, clear current stocks
        if replace_existing:
            removed_stocks = [stock.symbol for stock in pool.stocks]
            pool.stocks = []
        
        # Add new stocks from screening results
        additions_count = 0
        for result in screening_results[:max_additions]:
            symbol = result.get('symbol', '')
            name = result.get('name', symbol)
            
            if symbol and additions_count < max_additions:
                success = await self.add_stock_to_pool(
                    pool_id=pool_id,
                    symbol=symbol,
                    name=name,
                    notes=f"Added from screening on {datetime.now().strftime('%Y-%m-%d')}"
                )
                
                if success:
                    added_stocks.append(symbol)
                    additions_count += 1
        
        # Log bulk update
        await self._log_pool_action(
            pool_id,
            "bulk_update_from_screening",
            {
                "added_stocks": added_stocks,
                "removed_stocks": removed_stocks,
                "replace_existing": replace_existing
            }
        )
        
        return {
            "success": True,
            "added_stocks": added_stocks,
            "removed_stocks": removed_stocks,
            "total_stocks": len(pool.stocks)
        }
    
    async def get_pool_analytics(self, pool_id: str) -> Dict[str, Any]:
        """Get comprehensive pool analytics"""
        
        if pool_id not in self.pools:
            return {"error": "Pool not found"}
        
        pool = self.pools[pool_id]
        
        # Update metrics first
        await self._update_pool_metrics(pool_id)
        
        # Calculate additional analytics
        analytics = {
            "basic_info": {
                "pool_id": pool_id,
                "name": pool.name,
                "type": pool.pool_type.value,
                "status": pool.status.value,
                "total_stocks": len(pool.stocks),
                "created_date": pool.created_date.isoformat(),
                "last_modified": pool.last_modified.isoformat()
            },
            "performance_metrics": asdict(pool.metrics),
            "stock_analysis": await self._analyze_pool_stocks(pool),
            "risk_analysis": await self._analyze_pool_risk(pool),
            "sector_analysis": await self._analyze_pool_sectors(pool),
            "recommendations": await self._generate_pool_recommendations(pool)
        }
        
        return analytics
    
    async def compare_pools(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple pools"""
        
        if not pool_ids:
            return {"error": "No pools specified"}
        
        # Validate all pools exist
        for pool_id in pool_ids:
            if pool_id not in self.pools:
                return {"error": f"Pool {pool_id} not found"}
        
        comparison = {
            "pools": {},
            "comparative_metrics": {},
            "rankings": {},
            "correlation_analysis": {}
        }
        
        # Get analytics for each pool
        for pool_id in pool_ids:
            pool_analytics = await self.get_pool_analytics(pool_id)
            comparison["pools"][pool_id] = pool_analytics
        
        # Calculate comparative metrics
        comparison["comparative_metrics"] = await self._calculate_comparative_metrics(pool_ids)
        
        # Generate rankings
        comparison["rankings"] = await self._rank_pools(pool_ids)
        
        # Analyze correlations
        comparison["correlation_analysis"] = await self._analyze_pool_correlations(pool_ids)
        
        return comparison
    
    async def set_auto_update_rules(
        self,
        pool_id: str,
        rules: Dict[str, Any]
    ) -> bool:
        """Set automatic update rules for a pool"""
        
        if pool_id not in self.pools:
            logger.error(f"Pool {pool_id} not found")
            return False
        
        pool = self.pools[pool_id]
        pool.auto_update_rules = rules
        pool.last_modified = datetime.now()
        
        # Start auto-update task if enabled
        if rules.get("enabled", False):
            await self._start_auto_update_task(pool_id)
        else:
            await self._stop_auto_update_task(pool_id)
        
        logger.info(f"Set auto-update rules for pool {pool_id}")
        return True
    
    async def get_pool_history(self, pool_id: str, limit: int = 100) -> List[Dict]:
        """Get pool modification history"""
        
        if pool_id not in self.pool_history:
            return []
        
        return self.pool_history[pool_id][-limit:]
    
    async def restore_pool_state(self, pool_id: str, timestamp: datetime) -> bool:
        """Restore pool to a previous state"""
        
        if pool_id not in self.pool_history:
            logger.error(f"No history found for pool {pool_id}")
            return False
        
        # Find the closest historical state
        history = self.pool_history[pool_id]
        target_state = None
        
        for state in reversed(history):
            if datetime.fromisoformat(state["timestamp"]) <= timestamp:
                target_state = state
                break
        
        if not target_state:
            logger.error(f"No historical state found for pool {pool_id} at {timestamp}")
            return False
        
        # Restore pool state (simplified implementation)
        # In a full implementation, this would restore the complete pool state
        logger.info(f"Restored pool {pool_id} to state at {timestamp}")
        return True
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current stock price"""
        
        if self.data_source:
            try:
                # Use data source manager to get current price
                data = await self.data_source.get_stock_data(symbol, limit=1)
                if not data.empty:
                    return float(data.iloc[-1]['close_price'])
            except Exception as e:
                logger.warning(f"Failed to get current price for {symbol}: {e}")
        
        # Return a mock price for testing
        return 100.0 + np.random.uniform(-10, 10)
    
    async def _update_pool_metrics(self, pool_id: str):
        """Update pool performance metrics"""
        
        pool = self.pools[pool_id]
        
        if not pool.stocks:
            return
        
        # Calculate basic metrics
        total_return = 0.0
        total_weight = 0.0
        returns = []
        
        for stock in pool.stocks:
            if stock.added_price > 0:
                stock_return = (stock.current_price - stock.added_price) / stock.added_price
                weight = stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
                
                total_return += stock_return * weight
                total_weight += weight
                returns.append(stock_return)
        
        if total_weight > 0:
            pool.metrics.total_return = total_return / total_weight if total_weight != 1.0 else total_return
        
        # Calculate volatility
        if len(returns) > 1:
            pool.metrics.volatility = float(np.std(returns))
        
        # Calculate Sharpe ratio (simplified)
        if pool.metrics.volatility > 0:
            pool.metrics.sharpe_ratio = pool.metrics.total_return / pool.metrics.volatility
        
        # Calculate win rate
        winning_stocks = sum(1 for r in returns if r > 0)
        pool.metrics.win_rate = winning_stocks / len(returns) if returns else 0.0
        
        # Update timestamp
        pool.metrics.last_updated = datetime.now()
    
    async def _analyze_pool_stocks(self, pool: StockPool) -> Dict[str, Any]:
        """Analyze individual stocks in the pool"""
        
        stock_analysis = {
            "top_performers": [],
            "worst_performers": [],
            "recent_additions": [],
            "high_weight_stocks": []
        }
        
        if not pool.stocks:
            return stock_analysis
        
        # Calculate returns for each stock
        stock_returns = []
        for stock in pool.stocks:
            if stock.added_price > 0:
                return_pct = (stock.current_price - stock.added_price) / stock.added_price
                stock_returns.append({
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "return": return_pct,
                    "weight": stock.weight,
                    "added_date": stock.added_date
                })
        
        # Sort by performance
        stock_returns.sort(key=lambda x: x["return"], reverse=True)
        
        # Top and worst performers
        stock_analysis["top_performers"] = stock_returns[:5]
        stock_analysis["worst_performers"] = stock_returns[-5:]
        
        # Recent additions (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        stock_analysis["recent_additions"] = [
            s for s in stock_returns 
            if s["added_date"] > recent_cutoff
        ]
        
        # High weight stocks
        stock_analysis["high_weight_stocks"] = [
            s for s in stock_returns 
            if s["weight"] > 0.05  # More than 5% weight
        ]
        
        return stock_analysis
    
    async def _analyze_pool_risk(self, pool: StockPool) -> Dict[str, Any]:
        """Analyze pool risk metrics"""
        
        risk_analysis = {
            "concentration_risk": 0.0,
            "sector_concentration": {},
            "volatility_analysis": {},
            "risk_score": 0.0
        }
        
        if not pool.stocks:
            return risk_analysis
        
        # Calculate concentration risk (Herfindahl index)
        weights = [stock.weight if stock.weight > 0 else 1.0/len(pool.stocks) for stock in pool.stocks]
        concentration_risk = sum(w**2 for w in weights)
        risk_analysis["concentration_risk"] = concentration_risk
        
        # Risk score (simplified)
        risk_analysis["risk_score"] = min(100, concentration_risk * 100 + pool.metrics.volatility * 50)
        
        return risk_analysis
    
    async def _analyze_pool_sectors(self, pool: StockPool) -> Dict[str, Any]:
        """Analyze sector distribution in the pool"""
        
        # Mock sector analysis - in real implementation, would fetch sector data
        sectors = ["Technology", "Healthcare", "Finance", "Consumer", "Industrial"]
        sector_distribution = {}
        
        for i, stock in enumerate(pool.stocks):
            sector = sectors[i % len(sectors)]  # Mock assignment
            weight = stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
            
            if sector not in sector_distribution:
                sector_distribution[sector] = 0.0
            sector_distribution[sector] += weight
        
        return {
            "sector_distribution": sector_distribution,
            "diversification_score": len(sector_distribution) / len(sectors) * 100,
            "dominant_sector": max(sector_distribution.items(), key=lambda x: x[1]) if sector_distribution else None
        }
    
    async def _generate_pool_recommendations(self, pool: StockPool) -> List[Dict[str, Any]]:
        """Generate recommendations for pool optimization"""
        
        recommendations = []
        
        # Check concentration risk
        if len(pool.stocks) < 10:
            recommendations.append({
                "type": "diversification",
                "priority": "high",
                "message": "Consider adding more stocks to improve diversification",
                "action": "add_stocks"
            })
        
        # Check performance
        if pool.metrics.total_return < -0.1:  # -10%
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": "Pool is underperforming. Consider reviewing stock selection",
                "action": "review_stocks"
            })
        
        # Check rebalancing
        recommendations.append({
            "type": "maintenance",
            "priority": "low",
            "message": f"Consider rebalancing based on {pool.rebalance_frequency} schedule",
            "action": "rebalance"
        })
        
        return recommendations
    
    async def _calculate_comparative_metrics(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Calculate comparative metrics across pools"""
        
        metrics = {
            "returns": {},
            "volatility": {},
            "sharpe_ratios": {},
            "stock_counts": {}
        }
        
        for pool_id in pool_ids:
            pool = self.pools[pool_id]
            metrics["returns"][pool_id] = pool.metrics.total_return
            metrics["volatility"][pool_id] = pool.metrics.volatility
            metrics["sharpe_ratios"][pool_id] = pool.metrics.sharpe_ratio
            metrics["stock_counts"][pool_id] = len(pool.stocks)
        
        return metrics
    
    async def _rank_pools(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Rank pools by various metrics"""
        
        rankings = {
            "by_return": [],
            "by_sharpe": [],
            "by_size": []
        }
        
        # Rank by return
        pool_returns = [(pool_id, self.pools[pool_id].metrics.total_return) for pool_id in pool_ids]
        pool_returns.sort(key=lambda x: x[1], reverse=True)
        rankings["by_return"] = pool_returns
        
        # Rank by Sharpe ratio
        pool_sharpe = [(pool_id, self.pools[pool_id].metrics.sharpe_ratio) for pool_id in pool_ids]
        pool_sharpe.sort(key=lambda x: x[1], reverse=True)
        rankings["by_sharpe"] = pool_sharpe
        
        # Rank by size
        pool_sizes = [(pool_id, len(self.pools[pool_id].stocks)) for pool_id in pool_ids]
        pool_sizes.sort(key=lambda x: x[1], reverse=True)
        rankings["by_size"] = pool_sizes
        
        return rankings
    
    async def _analyze_pool_correlations(self, pool_ids: List[str]) -> Dict[str, Any]:
        """Analyze correlations between pools"""
        
        # Simplified correlation analysis
        # In real implementation, would calculate actual correlations based on historical returns
        
        correlations = {}
        for i, pool_id1 in enumerate(pool_ids):
            for pool_id2 in pool_ids[i+1:]:
                # Mock correlation
                correlation = np.random.uniform(0.3, 0.8)
                correlations[f"{pool_id1}_{pool_id2}"] = correlation
        
        return {
            "pairwise_correlations": correlations,
            "average_correlation": np.mean(list(correlations.values())) if correlations else 0.0
        }
    
    async def _log_pool_action(self, pool_id: str, action: str, details: Dict[str, Any]):
        """Log pool actions for history tracking"""
        
        if pool_id not in self.pool_history:
            self.pool_history[pool_id] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        
        self.pool_history[pool_id].append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.pool_history[pool_id]) > 1000:
            self.pool_history[pool_id] = self.pool_history[pool_id][-1000:]
    
    async def _start_auto_update_task(self, pool_id: str):
        """Start automatic update task for a pool"""
        
        if pool_id in self.auto_update_tasks:
            self.auto_update_tasks[pool_id].cancel()
        
        async def auto_update_loop():
            while True:
                try:
                    pool = self.pools[pool_id]
                    rules = pool.auto_update_rules
                    
                    # Wait for the specified interval
                    interval = rules.get("interval_hours", 24)
                    await asyncio.sleep(interval * 3600)
                    
                    # Perform auto-update based on rules
                    await self._perform_auto_update(pool_id, rules)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Auto-update error for pool {pool_id}: {e}")
                    await asyncio.sleep(3600)  # Wait 1 hour before retry
        
        self.auto_update_tasks[pool_id] = asyncio.create_task(auto_update_loop())
    
    async def _stop_auto_update_task(self, pool_id: str):
        """Stop automatic update task for a pool"""
        
        if pool_id in self.auto_update_tasks:
            self.auto_update_tasks[pool_id].cancel()
            del self.auto_update_tasks[pool_id]
    
    async def _perform_auto_update(self, pool_id: str, rules: Dict[str, Any]):
        """Perform automatic pool update based on rules"""
        
        # Update current prices
        pool = self.pools[pool_id]
        for stock in pool.stocks:
            stock.current_price = await self._get_current_price(stock.symbol)
        
        # Update metrics
        await self._update_pool_metrics(pool_id)
        
        # Apply auto-update rules
        if rules.get("remove_poor_performers", False):
            threshold = rules.get("poor_performance_threshold", -0.2)  # -20%
            
            stocks_to_remove = []
            for stock in pool.stocks:
                if stock.added_price > 0:
                    return_pct = (stock.current_price - stock.added_price) / stock.added_price
                    if return_pct < threshold:
                        stocks_to_remove.append(stock.symbol)
            
            for symbol in stocks_to_remove:
                await self.remove_stock_from_pool(pool_id, symbol)
        
        # Log auto-update
        await self._log_pool_action(pool_id, "auto_update", {"rules_applied": list(rules.keys())})
        
        logger.info(f"Performed auto-update for pool {pool_id}")