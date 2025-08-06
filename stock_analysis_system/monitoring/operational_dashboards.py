"""
Operational Dashboards System

This module implements comprehensive operational dashboards for the stock analysis system.
It provides system health monitoring, business metrics tracking, incident management,
and operational runbooks integration.
"""

import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import pandas as pd


@dataclass
class SystemHealthStatus:
    """System health status data structure"""
    component: str
    status: str  # healthy, degraded, unhealthy, unknown
    last_check: datetime
    response_time: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_percent: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessKPI:
    """Business KPI data structure"""
    name: str
    current_value: float
    target_value: Optional[float]
    unit: str
    trend: str  # up, down, stable
    change_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"


@dataclass
class Incident:
    """Incident data structure"""
    id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    status: str  # open, investigating, resolved, closed
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    resolution_time: Optional[timedelta] = None
    affected_components: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Runbook:
    """Operational runbook data structure"""
    id: str
    title: str
    description: str
    category: str
    steps: List[Dict[str, str]]
    triggers: List[str]
    estimated_time: Optional[int] = None  # minutes
    last_updated: datetime = field(default_factory=datetime.now)
    success_rate: Optional[float] = None


class OperationalDashboards:
    """
    Comprehensive operational dashboards manager.
    
    Features:
    - System health and status monitoring
    - Business metrics and KPI tracking
    - Incident management and response workflows
    - Operational runbooks and documentation
    - Real-time dashboard generation
    - Alert integration and visualization
    - Performance trend analysis
    """
    
    def __init__(self, 
                 dashboard_refresh_interval: int = 30,
                 history_retention_days: int = 30):
        """
        Initialize operational dashboards.
        
        Args:
            dashboard_refresh_interval: Dashboard refresh interval in seconds
            history_retention_days: How long to retain historical data
        """
        self.refresh_interval = dashboard_refresh_interval
        self.history_retention = timedelta(days=history_retention_days)
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.system_health: Dict[str, SystemHealthStatus] = {}
        self.business_kpis: Dict[str, BusinessKPI] = {}
        self.incidents: Dict[str, Incident] = {}
        self.runbooks: Dict[str, Runbook] = {}
        
        # Historical data
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.kpi_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.incident_history: List[Incident] = []
        
        # Dashboard generation
        self.dashboard_cache: Dict[str, Dict[str, Any]] = {}
        self.last_dashboard_update: Dict[str, datetime] = {}
        
        # Thread management
        self._update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        
        self.lock = threading.Lock()
        
        # Initialize default components and KPIs
        self._initialize_default_components()
        self._initialize_default_kpis()
        self._initialize_default_runbooks()
    
    def _initialize_default_components(self):
        """Initialize default system components to monitor"""
        default_components = [
            "api_server", "database", "redis_cache", "celery_workers",
            "prometheus", "grafana", "jaeger", "elasticsearch",
            "data_sources", "ml_models", "file_storage"
        ]
        
        for component in default_components:
            self.system_health[component] = SystemHealthStatus(
                component=component,
                status="unknown",
                last_check=datetime.now()
            )
    
    def _initialize_default_kpis(self):
        """Initialize default business KPIs"""
        default_kpis = [
            ("active_users", "Users", "count", 0, 1000),
            ("api_requests_per_minute", "API Requests/min", "count", 0, 500),
            ("stocks_analyzed_per_hour", "Stocks Analyzed/hour", "count", 0, 10000),
            ("system_uptime", "System Uptime", "percent", 99.9, 99.9),
            ("error_rate", "Error Rate", "percent", 0, 1.0),
            ("response_time_p95", "Response Time P95", "seconds", 0, 2.0),
            ("data_freshness", "Data Freshness", "minutes", 0, 15),
            ("cache_hit_rate", "Cache Hit Rate", "percent", 85, 90),
            ("ml_model_accuracy", "ML Model Accuracy", "percent", 85, 90),
            ("user_satisfaction", "User Satisfaction", "score", 4.5, 4.8)
        ]
        
        for name, display_name, unit, current, target in default_kpis:
            self.business_kpis[name] = BusinessKPI(
                name=display_name,
                current_value=current,
                target_value=target,
                unit=unit,
                trend="stable",
                category="system" if name in ["system_uptime", "error_rate", "response_time_p95"] else "business"
            )
    
    def _initialize_default_runbooks(self):
        """Initialize default operational runbooks"""
        runbooks_data = [
            {
                "id": "high_error_rate",
                "title": "High Error Rate Response",
                "description": "Steps to investigate and resolve high error rates",
                "category": "incident_response",
                "triggers": ["error_rate > 5%", "api_errors_spike"],
                "steps": [
                    {"step": "1", "action": "Check error logs in ELK stack", "command": "kubectl logs -f deployment/api-server"},
                    {"step": "2", "action": "Verify database connectivity", "command": "pg_isready -h db-host"},
                    {"step": "3", "action": "Check external API status", "command": "curl -I https://api.external.com/health"},
                    {"step": "4", "action": "Scale up API servers if needed", "command": "kubectl scale deployment api-server --replicas=5"},
                    {"step": "5", "action": "Monitor error rate recovery", "command": "Check Grafana dashboard"}
                ],
                "estimated_time": 15
            },
            {
                "id": "database_performance",
                "title": "Database Performance Issues",
                "description": "Troubleshoot database performance problems",
                "category": "performance",
                "triggers": ["db_query_time > 2s", "db_connections_high"],
                "steps": [
                    {"step": "1", "action": "Check active queries", "command": "SELECT * FROM pg_stat_activity WHERE state = 'active'"},
                    {"step": "2", "action": "Identify slow queries", "command": "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10"},
                    {"step": "3", "action": "Check database locks", "command": "SELECT * FROM pg_locks WHERE NOT granted"},
                    {"step": "4", "action": "Analyze table statistics", "command": "ANALYZE;"},
                    {"step": "5", "action": "Consider connection pooling adjustment", "command": "Review pgbouncer configuration"}
                ],
                "estimated_time": 20
            },
            {
                "id": "memory_leak",
                "title": "Memory Leak Investigation",
                "description": "Investigate and resolve memory leaks",
                "category": "performance",
                "triggers": ["memory_usage > 90%", "oom_killer_invoked"],
                "steps": [
                    {"step": "1", "action": "Check memory usage by process", "command": "ps aux --sort=-%mem | head -20"},
                    {"step": "2", "action": "Generate heap dump", "command": "jmap -dump:format=b,file=heap.hprof <pid>"},
                    {"step": "3", "action": "Check for memory leaks in logs", "command": "grep -i 'memory\\|leak\\|oom' /var/log/app.log"},
                    {"step": "4", "action": "Restart affected services", "command": "systemctl restart app-service"},
                    {"step": "5", "action": "Monitor memory usage recovery", "command": "watch -n 5 free -h"}
                ],
                "estimated_time": 30
            },
            {
                "id": "data_pipeline_failure",
                "title": "Data Pipeline Failure Recovery",
                "description": "Recover from data pipeline failures",
                "category": "data",
                "triggers": ["etl_job_failed", "data_freshness > 60min"],
                "steps": [
                    {"step": "1", "action": "Check Celery worker status", "command": "celery -A app inspect active"},
                    {"step": "2", "action": "Review failed tasks", "command": "celery -A app inspect reserved"},
                    {"step": "3", "action": "Check data source connectivity", "command": "python check_data_sources.py"},
                    {"step": "4", "action": "Restart failed tasks", "command": "celery -A app control revoke <task_id>"},
                    {"step": "5", "action": "Monitor pipeline recovery", "command": "Check data freshness metrics"}
                ],
                "estimated_time": 25
            }
        ]
        
        for runbook_data in runbooks_data:
            runbook = Runbook(**runbook_data)
            self.runbooks[runbook.id] = runbook
    
    def start_dashboard_updates(self):
        """Start automatic dashboard updates"""
        if self._update_thread and self._update_thread.is_alive():
            self.logger.warning("Dashboard updates already running")
            return
        
        self._stop_updates.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self._update_thread.start()
        
        self.logger.info("Dashboard updates started")
    
    def stop_dashboard_updates(self):
        """Stop automatic dashboard updates"""
        if self._update_thread:
            self._stop_updates.set()
            self._update_thread.join(timeout=5)
        
        self.logger.info("Dashboard updates stopped")
    
    def _update_loop(self):
        """Main dashboard update loop"""
        while not self._stop_updates.is_set():
            try:
                # Update system health
                self._update_system_health()
                
                # Update business KPIs
                self._update_business_kpis()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Refresh dashboard cache
                self._refresh_dashboard_cache()
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
            
            # Wait for next update
            self._stop_updates.wait(self.refresh_interval)
    
    def _update_system_health(self):
        """Update system health status"""
        # This would typically integrate with actual health checks
        # For now, we'll simulate health status updates
        pass
    
    def _update_business_kpis(self):
        """Update business KPIs"""
        # This would typically pull from metrics systems
        # For now, we'll keep existing values
        pass
    
    def _cleanup_old_data(self):
        """Clean up old historical data"""
        cutoff_time = datetime.now() - self.history_retention
        
        with self.lock:
            # Clean health history
            for component, history in self.health_history.items():
                while history and history[0].last_check < cutoff_time:
                    history.popleft()
            
            # Clean KPI history
            for kpi_name, history in self.kpi_history.items():
                while history and history[0].timestamp < cutoff_time:
                    history.popleft()
            
            # Clean incident history
            self.incident_history = [
                incident for incident in self.incident_history
                if incident.created_at > cutoff_time
            ]
    
    def _refresh_dashboard_cache(self):
        """Refresh dashboard cache"""
        dashboard_types = ["system_health", "business_kpis", "incidents", "performance"]
        
        for dashboard_type in dashboard_types:
            try:
                if (dashboard_type not in self.last_dashboard_update or
                    datetime.now() - self.last_dashboard_update[dashboard_type] > 
                    timedelta(seconds=self.refresh_interval)):
                    
                    self.dashboard_cache[dashboard_type] = self._generate_dashboard_data(dashboard_type)
                    self.last_dashboard_update[dashboard_type] = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error refreshing {dashboard_type} dashboard: {e}")
    
    def update_system_health(self, component: str, status: str, 
                           response_time: Optional[float] = None,
                           error_rate: Optional[float] = None,
                           uptime_percent: Optional[float] = None,
                           details: Optional[Dict[str, Any]] = None):
        """
        Update system health status for a component.
        
        Args:
            component: Component name
            status: Health status (healthy, degraded, unhealthy, unknown)
            response_time: Response time in seconds
            error_rate: Error rate percentage
            uptime_percent: Uptime percentage
            details: Additional details
        """
        health_status = SystemHealthStatus(
            component=component,
            status=status,
            last_check=datetime.now(),
            response_time=response_time,
            error_rate=error_rate,
            uptime_percent=uptime_percent,
            details=details or {}
        )
        
        with self.lock:
            self.system_health[component] = health_status
            self.health_history[component].append(health_status)
    
    def update_business_kpi(self, name: str, current_value: float,
                          target_value: Optional[float] = None,
                          trend: Optional[str] = None):
        """
        Update business KPI value.
        
        Args:
            name: KPI name
            current_value: Current value
            target_value: Target value
            trend: Trend direction (up, down, stable)
        """
        with self.lock:
            if name in self.business_kpis:
                kpi = self.business_kpis[name]
                
                # Calculate trend if not provided
                if trend is None and self.kpi_history[name]:
                    last_value = self.kpi_history[name][-1].current_value
                    if current_value > last_value * 1.05:
                        trend = "up"
                    elif current_value < last_value * 0.95:
                        trend = "down"
                    else:
                        trend = "stable"
                
                # Calculate change percentage
                change_percent = None
                if self.kpi_history[name]:
                    last_value = self.kpi_history[name][-1].current_value
                    if last_value > 0:
                        change_percent = ((current_value - last_value) / last_value) * 100
                
                # Update KPI
                kpi.current_value = current_value
                if target_value is not None:
                    kpi.target_value = target_value
                if trend:
                    kpi.trend = trend
                kpi.change_percent = change_percent
                kpi.timestamp = datetime.now()
                
                # Add to history
                self.kpi_history[name].append(kpi)
    
    def create_incident(self, title: str, description: str, severity: str,
                       affected_components: Optional[List[str]] = None,
                       assigned_to: Optional[str] = None) -> str:
        """
        Create a new incident.
        
        Args:
            title: Incident title
            description: Incident description
            severity: Severity level (critical, high, medium, low)
            affected_components: List of affected components
            assigned_to: Person assigned to the incident
            
        Returns:
            Incident ID
        """
        incident_id = f"INC-{int(time.time())}"
        
        incident = Incident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status="open",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            assigned_to=assigned_to,
            affected_components=affected_components or [],
            timeline=[{
                "timestamp": datetime.now().isoformat(),
                "action": "incident_created",
                "description": f"Incident created: {title}",
                "user": "system"
            }]
        )
        
        with self.lock:
            self.incidents[incident_id] = incident
            self.incident_history.append(incident)
        
        self.logger.warning(f"Incident created: {incident_id} - {title}")
        return incident_id
    
    def update_incident(self, incident_id: str, status: Optional[str] = None,
                       assigned_to: Optional[str] = None,
                       update_description: Optional[str] = None,
                       user: str = "system"):
        """
        Update an existing incident.
        
        Args:
            incident_id: Incident ID
            status: New status
            assigned_to: New assignee
            update_description: Update description
            user: User making the update
        """
        with self.lock:
            if incident_id not in self.incidents:
                raise ValueError(f"Incident {incident_id} not found")
            
            incident = self.incidents[incident_id]
            
            # Update fields
            if status:
                incident.status = status
            if assigned_to:
                incident.assigned_to = assigned_to
            
            incident.updated_at = datetime.now()
            
            # Calculate resolution time if resolved
            if status in ["resolved", "closed"] and not incident.resolution_time:
                incident.resolution_time = datetime.now() - incident.created_at
            
            # Add timeline entry
            timeline_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "incident_updated",
                "description": update_description or f"Incident updated",
                "user": user
            }
            
            if status:
                timeline_entry["status_change"] = status
            if assigned_to:
                timeline_entry["assigned_to"] = assigned_to
            
            incident.timeline.append(timeline_entry)
        
        self.logger.info(f"Incident updated: {incident_id}")
    
    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get runbook by ID"""
        return self.runbooks.get(runbook_id)
    
    def get_runbooks_by_trigger(self, trigger: str) -> List[Runbook]:
        """Get runbooks that match a trigger"""
        matching_runbooks = []
        
        for runbook in self.runbooks.values():
            if any(trigger.lower() in t.lower() for t in runbook.triggers):
                matching_runbooks.append(runbook)
        
        return matching_runbooks
    
    def _generate_dashboard_data(self, dashboard_type: str) -> Dict[str, Any]:
        """Generate dashboard data for specified type"""
        if dashboard_type == "system_health":
            return self._generate_system_health_dashboard()
        elif dashboard_type == "business_kpis":
            return self._generate_business_kpis_dashboard()
        elif dashboard_type == "incidents":
            return self._generate_incidents_dashboard()
        elif dashboard_type == "performance":
            return self._generate_performance_dashboard()
        else:
            return {}
    
    def _generate_system_health_dashboard(self) -> Dict[str, Any]:
        """Generate system health dashboard data"""
        with self.lock:
            health_summary = {
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "unknown": 0
            }
            
            components_data = []
            
            for component, health in self.system_health.items():
                health_summary[health.status] += 1
                
                components_data.append({
                    "component": component,
                    "status": health.status,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "error_rate": health.error_rate,
                    "uptime_percent": health.uptime_percent,
                    "details": health.details
                })
            
            return {
                "summary": health_summary,
                "components": components_data,
                "total_components": len(self.system_health),
                "overall_health": self._calculate_overall_health(),
                "last_updated": datetime.now().isoformat()
            }
    
    def _generate_business_kpis_dashboard(self) -> Dict[str, Any]:
        """Generate business KPIs dashboard data"""
        with self.lock:
            kpis_data = []
            categories = defaultdict(list)
            
            for kpi_name, kpi in self.business_kpis.items():
                kpi_data = {
                    "name": kpi.name,
                    "current_value": kpi.current_value,
                    "target_value": kpi.target_value,
                    "unit": kpi.unit,
                    "trend": kpi.trend,
                    "change_percent": kpi.change_percent,
                    "timestamp": kpi.timestamp.isoformat(),
                    "category": kpi.category,
                    "status": self._get_kpi_status(kpi)
                }
                
                kpis_data.append(kpi_data)
                categories[kpi.category].append(kpi_data)
            
            return {
                "kpis": kpis_data,
                "categories": dict(categories),
                "total_kpis": len(self.business_kpis),
                "last_updated": datetime.now().isoformat()
            }
    
    def _generate_incidents_dashboard(self) -> Dict[str, Any]:
        """Generate incidents dashboard data"""
        with self.lock:
            incidents_data = []
            status_summary = defaultdict(int)
            severity_summary = defaultdict(int)
            
            for incident in self.incidents.values():
                incident_data = {
                    "id": incident.id,
                    "title": incident.title,
                    "description": incident.description,
                    "severity": incident.severity,
                    "status": incident.status,
                    "created_at": incident.created_at.isoformat(),
                    "updated_at": incident.updated_at.isoformat(),
                    "assigned_to": incident.assigned_to,
                    "affected_components": incident.affected_components,
                    "resolution_time": incident.resolution_time.total_seconds() if incident.resolution_time else None,
                    "timeline_count": len(incident.timeline)
                }
                
                incidents_data.append(incident_data)
                status_summary[incident.status] += 1
                severity_summary[incident.severity] += 1
            
            return {
                "incidents": incidents_data,
                "status_summary": dict(status_summary),
                "severity_summary": dict(severity_summary),
                "total_incidents": len(self.incidents),
                "open_incidents": status_summary.get("open", 0) + status_summary.get("investigating", 0),
                "last_updated": datetime.now().isoformat()
            }
    
    def _generate_performance_dashboard(self) -> Dict[str, Any]:
        """Generate performance dashboard data"""
        # This would integrate with the performance monitoring system
        return {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 34.1,
            "network_io": {"in": 1024000, "out": 2048000},
            "api_response_time": 0.234,
            "database_query_time": 0.045,
            "cache_hit_rate": 89.5,
            "error_rate": 0.12,
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        if not self.system_health:
            return "unknown"
        
        status_counts = defaultdict(int)
        for health in self.system_health.values():
            status_counts[health.status] += 1
        
        total = len(self.system_health)
        
        if status_counts["unhealthy"] > 0:
            return "unhealthy"
        elif status_counts["degraded"] > total * 0.2:  # More than 20% degraded
            return "degraded"
        elif status_counts["healthy"] > total * 0.8:  # More than 80% healthy
            return "healthy"
        else:
            return "degraded"
    
    def _get_kpi_status(self, kpi: BusinessKPI) -> str:
        """Get KPI status based on target"""
        if not kpi.target_value:
            return "unknown"
        
        ratio = kpi.current_value / kpi.target_value
        
        if ratio >= 0.95:  # Within 5% of target
            return "good"
        elif ratio >= 0.8:  # Within 20% of target
            return "warning"
        else:
            return "critical"
    
    def generate_html_dashboard(self, dashboard_type: str = "overview") -> str:
        """
        Generate HTML dashboard.
        
        Args:
            dashboard_type: Type of dashboard (overview, system_health, business_kpis, incidents)
            
        Returns:
            HTML string for the dashboard
        """
        if dashboard_type == "overview":
            return self._generate_overview_html()
        elif dashboard_type == "system_health":
            return self._generate_system_health_html()
        elif dashboard_type == "business_kpis":
            return self._generate_business_kpis_html()
        elif dashboard_type == "incidents":
            return self._generate_incidents_html()
        else:
            return "<html><body><h1>Dashboard type not found</h1></body></html>"
    
    def _generate_overview_html(self) -> str:
        """Generate overview dashboard HTML"""
        # Get dashboard data
        health_data = self._generate_system_health_dashboard()
        kpis_data = self._generate_business_kpis_dashboard()
        incidents_data = self._generate_incidents_dashboard()
        
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Health', 'KPI Status', 'Incident Status', 'Performance Trends'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # System health pie chart
        health_labels = list(health_data["summary"].keys())
        health_values = list(health_data["summary"].values())
        
        fig.add_trace(
            go.Pie(labels=health_labels, values=health_values, name="Health"),
            row=1, col=1
        )
        
        # KPI status bar chart
        kpi_statuses = defaultdict(int)
        for kpi in kpis_data["kpis"]:
            kpi_statuses[kpi["status"]] += 1
        
        fig.add_trace(
            go.Bar(x=list(kpi_statuses.keys()), y=list(kpi_statuses.values()), name="KPIs"),
            row=1, col=2
        )
        
        # Incident status pie chart
        incident_labels = list(incidents_data["status_summary"].keys())
        incident_values = list(incidents_data["status_summary"].values())
        
        if incident_labels:
            fig.add_trace(
                go.Pie(labels=incident_labels, values=incident_values, name="Incidents"),
                row=2, col=1
            )
        
        # Performance trend (mock data)
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                             end=datetime.now(), freq='H')
        cpu_values = [45 + i * 0.5 + (i % 3) * 2 for i in range(len(times))]
        
        fig.add_trace(
            go.Scatter(x=times, y=cpu_values, mode='lines', name="CPU Usage %"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Stock Analysis System - Operational Overview",
            height=800,
            showlegend=True
        )
        
        return plot(fig, output_type='div', include_plotlyjs=True)
    
    def _generate_system_health_html(self) -> str:
        """Generate system health dashboard HTML"""
        health_data = self._generate_system_health_dashboard()
        
        # Create health status visualization
        components = [comp["component"] for comp in health_data["components"]]
        statuses = [comp["status"] for comp in health_data["components"]]
        response_times = [comp["response_time"] or 0 for comp in health_data["components"]]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Component Health Status', 'Response Times'),
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        
        # Color mapping for status
        color_map = {
            "healthy": "green",
            "degraded": "yellow", 
            "unhealthy": "red",
            "unknown": "gray"
        }
        colors = [color_map.get(status, "gray") for status in statuses]
        
        fig.add_trace(
            go.Bar(x=components, y=[1]*len(components), 
                  marker_color=colors, name="Health Status"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=components, y=response_times, name="Response Time (s)"),
            row=2, col=1
        )
        
        fig.update_layout(
            title="System Health Dashboard",
            height=600,
            showlegend=True
        )
        
        return plot(fig, output_type='div', include_plotlyjs=True)
    
    def _generate_business_kpis_html(self) -> str:
        """Generate business KPIs dashboard HTML"""
        kpis_data = self._generate_business_kpis_dashboard()
        
        # Create KPI visualization
        kpi_names = [kpi["name"] for kpi in kpis_data["kpis"]]
        current_values = [kpi["current_value"] for kpi in kpis_data["kpis"]]
        target_values = [kpi["target_value"] or 0 for kpi in kpis_data["kpis"]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=kpi_names,
            y=current_values,
            name="Current Value",
            marker_color="lightblue"
        ))
        
        fig.add_trace(go.Bar(
            x=kpi_names,
            y=target_values,
            name="Target Value",
            marker_color="darkblue"
        ))
        
        fig.update_layout(
            title="Business KPIs Dashboard",
            xaxis_title="KPIs",
            yaxis_title="Value",
            barmode='group',
            height=500
        )
        
        return plot(fig, output_type='div', include_plotlyjs=True)
    
    def _generate_incidents_html(self) -> str:
        """Generate incidents dashboard HTML"""
        incidents_data = self._generate_incidents_dashboard()
        
        if not incidents_data["incidents"]:
            return "<html><body><h2>No incidents to display</h2></body></html>"
        
        # Create incidents timeline
        incidents = incidents_data["incidents"]
        
        fig = go.Figure()
        
        for i, incident in enumerate(incidents):
            fig.add_trace(go.Scatter(
                x=[incident["created_at"], incident["updated_at"]],
                y=[i, i],
                mode='lines+markers',
                name=f"{incident['id']}: {incident['title'][:30]}...",
                line=dict(width=5),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Incidents Timeline",
            xaxis_title="Time",
            yaxis_title="Incident",
            height=max(400, len(incidents) * 30),
            showlegend=True
        )
        
        return plot(fig, output_type='div', include_plotlyjs=True)
    
    def export_dashboard_data(self, file_path: str, dashboard_type: str = "all"):
        """Export dashboard data to JSON file"""
        try:
            if dashboard_type == "all":
                data = {
                    "system_health": self._generate_system_health_dashboard(),
                    "business_kpis": self._generate_business_kpis_dashboard(),
                    "incidents": self._generate_incidents_dashboard(),
                    "performance": self._generate_performance_dashboard(),
                    "export_timestamp": datetime.now().isoformat()
                }
            else:
                data = {
                    dashboard_type: self._generate_dashboard_data(dashboard_type),
                    "export_timestamp": datetime.now().isoformat()
                }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Dashboard data exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get overall dashboard summary"""
        return {
            "system_health": {
                "overall_status": self._calculate_overall_health(),
                "total_components": len(self.system_health),
                "healthy_components": sum(1 for h in self.system_health.values() if h.status == "healthy")
            },
            "business_kpis": {
                "total_kpis": len(self.business_kpis),
                "kpis_on_target": sum(1 for kpi in self.business_kpis.values() 
                                    if self._get_kpi_status(kpi) == "good")
            },
            "incidents": {
                "total_incidents": len(self.incidents),
                "open_incidents": sum(1 for i in self.incidents.values() 
                                    if i.status in ["open", "investigating"]),
                "critical_incidents": sum(1 for i in self.incidents.values() 
                                        if i.severity == "critical" and i.status != "closed")
            },
            "runbooks": {
                "total_runbooks": len(self.runbooks),
                "categories": list(set(r.category for r in self.runbooks.values()))
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of operational dashboards"""
        return {
            "status": "healthy" if self._update_thread and self._update_thread.is_alive() else "unhealthy",
            "dashboard_updates_active": self._update_thread and self._update_thread.is_alive(),
            "refresh_interval": self.refresh_interval,
            "cached_dashboards": list(self.dashboard_cache.keys()),
            "total_components_monitored": len(self.system_health),
            "total_kpis_tracked": len(self.business_kpis),
            "active_incidents": len([i for i in self.incidents.values() if i.status in ["open", "investigating"]]),
            "last_update": max(self.last_dashboard_update.values()) if self.last_dashboard_update else None
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_dashboard_updates()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_dashboard_updates()


# Example usage and testing
if __name__ == "__main__":
    import random
    import time
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create operational dashboards
    dashboards = OperationalDashboards(
        dashboard_refresh_interval=10,  # 10 seconds for testing
        history_retention_days=1
    )
    
    try:
        with dashboards:
            print("Operational dashboards started...")
            
            # Simulate system health updates
            components = ["api_server", "database", "redis_cache", "celery_workers"]
            statuses = ["healthy", "degraded", "unhealthy"]
            
            for i in range(20):
                # Update system health
                for component in components:
                    status = random.choice(statuses) if random.random() > 0.8 else "healthy"
                    dashboards.update_system_health(
                        component=component,
                        status=status,
                        response_time=random.uniform(0.1, 2.0),
                        error_rate=random.uniform(0, 10),
                        uptime_percent=random.uniform(95, 100)
                    )
                
                # Update business KPIs
                kpi_names = ["active_users", "api_requests_per_minute", "stocks_analyzed_per_hour"]
                for kpi_name in kpi_names:
                    dashboards.update_business_kpi(
                        name=kpi_name,
                        current_value=random.uniform(50, 200),
                        target_value=150
                    )
                
                # Occasionally create incidents
                if random.random() < 0.1:  # 10% chance
                    incident_id = dashboards.create_incident(
                        title=f"Test incident {i}",
                        description=f"This is a test incident created at iteration {i}",
                        severity=random.choice(["low", "medium", "high", "critical"]),
                        affected_components=[random.choice(components)]
                    )
                    
                    # Sometimes update the incident
                    if random.random() < 0.5:
                        dashboards.update_incident(
                            incident_id=incident_id,
                            status="investigating",
                            update_description="Investigation started"
                        )
                
                time.sleep(0.5)
            
            # Wait for dashboard updates
            time.sleep(5)
            
            # Print dashboard summary
            summary = dashboards.get_dashboard_summary()
            print(f"\nDashboard Summary:")
            print(json.dumps(summary, indent=2, default=str))
            
            # Generate HTML dashboards
            overview_html = dashboards.generate_html_dashboard("overview")
            with open("overview_dashboard.html", "w") as f:
                f.write(overview_html)
            
            health_html = dashboards.generate_html_dashboard("system_health")
            with open("health_dashboard.html", "w") as f:
                f.write(health_html)
            
            # Export dashboard data
            dashboards.export_dashboard_data("dashboard_data.json", "all")
            
            # Print health status
            health = dashboards.get_health_status()
            print(f"\nHealth Status:")
            print(json.dumps(health, indent=2, default=str))
            
            print("\nOperational dashboards test completed!")
            print("Generated files:")
            print("- overview_dashboard.html")
            print("- health_dashboard.html") 
            print("- dashboard_data.json")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        raise