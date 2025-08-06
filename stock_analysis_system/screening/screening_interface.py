"""
Advanced Screening Interface

This module provides the main interface for the stock screening system,
including real-time screening, template management, and result handling.
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import asdict

from .screening_criteria import (
    ScreeningTemplate, ScreeningCriteriaBuilder, PredefinedTemplates,
    TechnicalCriteria, SeasonalCriteria, InstitutionalCriteria, RiskCriteria
)
from .screening_results import ScreeningResult, StockScore, ScreeningResultAnalyzer
from .screening_engine import ScreeningEngine


class ScreeningInterface:
    """
    Advanced interface for stock screening with real-time updates,
    template management, and comprehensive result analysis.
    """
    
    def __init__(self, screening_engine: ScreeningEngine):
        self.engine = screening_engine
        self.analyzer = ScreeningResultAnalyzer()
        
        # Template storage
        self.templates: Dict[str, ScreeningTemplate] = {}
        self.load_predefined_templates()
        
        # Real-time screening state
        self.active_screenings: Dict[str, Dict] = {}
        self.screening_callbacks: Dict[str, List[Callable]] = {}
        
        # Result cache
        self.result_cache: Dict[str, ScreeningResult] = {}
        self.cache_ttl_minutes = 30
        
        # Performance tracking
        self.screening_history: List[Dict] = []
    
    def load_predefined_templates(self):
        """Load predefined screening templates."""
        templates = [
            PredefinedTemplates.growth_momentum_template(),
            PredefinedTemplates.spring_festival_opportunity_template(),
            PredefinedTemplates.low_risk_value_template(),
            PredefinedTemplates.institutional_following_template()
        ]
        
        for template in templates:
            self.templates[template.name] = template
    
    async def create_custom_template(self, name: str, description: str,
                                   technical_params: Dict = None,
                                   seasonal_params: Dict = None,
                                   institutional_params: Dict = None,
                                   risk_params: Dict = None,
                                   tags: List[str] = None) -> str:
        """
        Create a custom screening template.
        
        Args:
            name: Template name
            description: Template description
            technical_params: Technical criteria parameters
            seasonal_params: Seasonal criteria parameters
            institutional_params: Institutional criteria parameters
            risk_params: Risk criteria parameters
            tags: Template tags
            
        Returns:
            Template ID
        """
        builder = ScreeningCriteriaBuilder()
        
        if technical_params:
            builder.with_technical_criteria(**technical_params)
        
        if seasonal_params:
            builder.with_seasonal_criteria(**seasonal_params)
        
        if institutional_params:
            builder.with_institutional_criteria(**institutional_params)
        
        if risk_params:
            builder.with_risk_criteria(**risk_params)
        
        template = builder.build_template(name, description, tags or [])
        self.templates[name] = template
        
        return name
    
    async def update_template(self, template_name: str, **updates) -> bool:
        """Update an existing template."""
        if template_name not in self.templates:
            return False
        
        template = self.templates[template_name]
        template.updated_at = datetime.now()
        
        # Update template fields
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        return True
    
    async def delete_template(self, template_name: str) -> bool:
        """Delete a screening template."""
        if template_name in self.templates:
            del self.templates[template_name]
            return True
        return False
    
    def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        return [
            {
                'name': name,
                'description': template.description,
                'created_at': template.created_at.isoformat(),
                'updated_at': template.updated_at.isoformat(),
                'tags': template.tags,
                'has_technical': template.technical_criteria is not None,
                'has_seasonal': template.seasonal_criteria is not None,
                'has_institutional': template.institutional_criteria is not None,
                'has_risk': template.risk_criteria is not None
            }
            for name, template in self.templates.items()
        ]
    
    def get_template_details(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed template information."""
        if template_name not in self.templates:
            return None
        
        return self.templates[template_name].to_dict()
    
    async def run_screening(self, template_name: str, 
                          stock_universe: List[str] = None,
                          max_results: int = 100,
                          use_cache: bool = True) -> ScreeningResult:
        """
        Run screening using a template.
        
        Args:
            template_name: Name of the template to use
            stock_universe: List of stock codes to screen (None for all)
            max_results: Maximum number of results to return
            use_cache: Whether to use cached results if available
            
        Returns:
            ScreeningResult object
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Check cache first
        cache_key = f"{template_name}_{hash(str(stock_universe))}"
        if use_cache and cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result
        
        # Run screening
        start_time = datetime.now()
        screening_id = str(uuid.uuid4())
        
        try:
            result = await self.engine.execute_screening(
                template=template,
                stock_universe=stock_universe,
                max_results=max_results,
                screening_id=screening_id
            )
            
            # Cache result
            if use_cache:
                self.result_cache[cache_key] = result
            
            # Track performance
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.screening_history.append({
                'screening_id': screening_id,
                'template_name': template_name,
                'execution_time': start_time.isoformat(),
                'duration_ms': execution_time,
                'stocks_found': result.stocks_passed,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Track failed screening
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.screening_history.append({
                'screening_id': screening_id,
                'template_name': template_name,
                'execution_time': start_time.isoformat(),
                'duration_ms': execution_time,
                'stocks_found': 0,
                'success': False,
                'error': str(e)
            })
            raise
    
    async def start_real_time_screening(self, template_name: str,
                                      update_interval_seconds: int = 300,
                                      callback: Callable = None) -> str:
        """
        Start real-time screening with periodic updates.
        
        Args:
            template_name: Template to use for screening
            update_interval_seconds: Update interval in seconds
            callback: Callback function for results
            
        Returns:
            Real-time screening session ID
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        session_id = str(uuid.uuid4())
        
        # Initialize session
        self.active_screenings[session_id] = {
            'template_name': template_name,
            'update_interval': update_interval_seconds,
            'started_at': datetime.now(),
            'last_update': None,
            'update_count': 0,
            'active': True
        }
        
        if callback:
            if session_id not in self.screening_callbacks:
                self.screening_callbacks[session_id] = []
            self.screening_callbacks[session_id].append(callback)
        
        # Start background task
        asyncio.create_task(self._real_time_screening_loop(session_id))
        
        return session_id
    
    async def stop_real_time_screening(self, session_id: str) -> bool:
        """Stop real-time screening session."""
        if session_id in self.active_screenings:
            self.active_screenings[session_id]['active'] = False
            return True
        return False
    
    async def _real_time_screening_loop(self, session_id: str):
        """Background loop for real-time screening updates."""
        session = self.active_screenings.get(session_id)
        if not session:
            return
        
        while session.get('active', False):
            try:
                # Run screening
                result = await self.run_screening(
                    template_name=session['template_name'],
                    use_cache=False  # Always get fresh data for real-time
                )
                
                # Update session
                session['last_update'] = datetime.now()
                session['update_count'] += 1
                
                # Call callbacks
                if session_id in self.screening_callbacks:
                    for callback in self.screening_callbacks[session_id]:
                        try:
                            await callback(session_id, result)
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                # Wait for next update
                await asyncio.sleep(session['update_interval'])
                
            except Exception as e:
                print(f"Real-time screening error: {e}")
                await asyncio.sleep(60)  # Wait before retry
        
        # Cleanup
        if session_id in self.active_screenings:
            del self.active_screenings[session_id]
        if session_id in self.screening_callbacks:
            del self.screening_callbacks[session_id]
    
    def get_real_time_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active real-time screening sessions."""
        return [
            {
                'session_id': session_id,
                'template_name': session['template_name'],
                'started_at': session['started_at'].isoformat(),
                'last_update': session['last_update'].isoformat() if session['last_update'] else None,
                'update_count': session['update_count'],
                'update_interval': session['update_interval'],
                'active': session['active']
            }
            for session_id, session in self.active_screenings.items()
        ]
    
    async def analyze_screening_result(self, result: ScreeningResult) -> Dict[str, Any]:
        """Perform comprehensive analysis of screening results."""
        return self.analyzer.create_performance_summary(result)
    
    async def compare_screening_results(self, result1: ScreeningResult, 
                                      result2: ScreeningResult) -> Dict[str, Any]:
        """Compare two screening results."""
        return self.analyzer.compare_results(result1, result2)
    
    async def get_screening_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get screening execution history."""
        return sorted(
            self.screening_history[-limit:],
            key=lambda x: x['execution_time'],
            reverse=True
        )
    
    async def export_template(self, template_name: str) -> str:
        """Export template to JSON string."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        return json.dumps(self.templates[template_name].to_dict(), indent=2)
    
    async def import_template(self, template_json: str) -> str:
        """Import template from JSON string."""
        try:
            template_data = json.loads(template_json)
            
            # Reconstruct template object
            template = ScreeningTemplate(
                name=template_data['name'],
                description=template_data['description'],
                created_at=datetime.fromisoformat(template_data['created_at']),
                updated_at=datetime.fromisoformat(template_data['updated_at']),
                tags=template_data.get('tags', [])
            )
            
            # Reconstruct criteria objects
            if template_data.get('technical_criteria'):
                tech_data = template_data['technical_criteria'].copy()
                tech_data.pop('type', None)  # Remove type field
                template.technical_criteria = TechnicalCriteria(**tech_data)
            
            if template_data.get('seasonal_criteria'):
                seasonal_data = template_data['seasonal_criteria'].copy()
                seasonal_data.pop('type', None)  # Remove type field
                template.seasonal_criteria = SeasonalCriteria(**seasonal_data)
            
            if template_data.get('institutional_criteria'):
                inst_data = template_data['institutional_criteria'].copy()
                inst_data.pop('type', None)  # Remove type field
                template.institutional_criteria = InstitutionalCriteria(**inst_data)
            
            if template_data.get('risk_criteria'):
                risk_data = template_data['risk_criteria'].copy()
                risk_data.pop('type', None)  # Remove type field
                template.risk_criteria = RiskCriteria(**risk_data)
            
            # Store template
            self.templates[template.name] = template
            
            return template.name
            
        except Exception as e:
            raise ValueError(f"Failed to import template: {e}")
    
    def _is_cache_valid(self, result: ScreeningResult) -> bool:
        """Check if cached result is still valid."""
        cache_age = datetime.now() - result.execution_time
        return cache_age.total_seconds() < (self.cache_ttl_minutes * 60)
    
    async def clear_cache(self):
        """Clear the result cache."""
        self.result_cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_entries = sum(1 for result in self.result_cache.values() 
                          if self._is_cache_valid(result))
        
        return {
            'total_entries': len(self.result_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.result_cache) - valid_entries,
            'cache_ttl_minutes': self.cache_ttl_minutes
        }
    
    async def cleanup_expired_cache(self):
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, result in self.result_cache.items()
            if not self._is_cache_valid(result)
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
        
        return len(expired_keys)