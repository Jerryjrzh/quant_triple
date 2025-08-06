"""
Grafana Dashboard Management System

This module provides comprehensive Grafana dashboard management for the stock analysis system.
It includes dashboard creation, configuration, and automated deployment capabilities.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import yaml


@dataclass
class DashboardPanel:
    """Grafana dashboard panel configuration"""
    id: int
    title: str
    type: str
    targets: List[Dict[str, Any]]
    gridPos: Dict[str, int]
    options: Dict[str, Any] = field(default_factory=dict)
    fieldConfig: Dict[str, Any] = field(default_factory=dict)
    datasource: Optional[str] = None


@dataclass
class DashboardConfig:
    """Grafana dashboard configuration"""
    uid: str
    title: str
    description: str
    tags: List[str]
    panels: List[DashboardPanel]
    time_range: Dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    refresh: str = "30s"
    timezone: str = "browser"


@dataclass
class GrafanaConfig:
    """Grafana server configuration"""
    url: str
    api_key: str
    org_id: int = 1
    timeout: int = 30
    verify_ssl: bool = True


class GrafanaDashboardManager:
    """
    Comprehensive Grafana dashboard manager for the stock analysis system.
    
    Features:
    - Dashboard creation and management
    - Panel configuration and templating
    - Automated dashboard deployment
    - Dashboard versioning and backup
    - Alert rule management
    - Data source configuration
    """
    
    def __init__(self, config: GrafanaConfig):
        """
        Initialize Grafana dashboard manager.
        
        Args:
            config: Grafana server configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Dashboard templates
        self.dashboard_templates = {}
        self._load_dashboard_templates()
    
    def _load_dashboard_templates(self):
        """Load dashboard templates from configuration"""
        self.dashboard_templates = {
            'system_overview': self._create_system_overview_template(),
            'business_metrics': self._create_business_metrics_template(),
            'stock_analysis': self._create_stock_analysis_template(),
            'ml_models': self._create_ml_models_template(),
            'api_performance': self._create_api_performance_template(),
            'error_monitoring': self._create_error_monitoring_template()
        }
    
    def _create_system_overview_template(self) -> DashboardConfig:
        """Create system overview dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="CPU Usage",
                type="stat",
                targets=[{
                    "expr": "avg(system_cpu_usage_percent)",
                    "legendFormat": "CPU Usage %",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                options={
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto"
                },
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="Memory Usage",
                type="stat",
                targets=[{
                    "expr": "system_memory_usage_bytes{type=\"used\"} / system_memory_usage_bytes{type=\"total\"} * 100",
                    "legendFormat": "Memory Usage %",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0},
                options={
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto"
                },
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 80},
                                {"color": "red", "value": 95}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="Network I/O",
                type="timeseries",
                targets=[
                    {
                        "expr": "rate(system_network_io_bytes_total{direction=\"sent\"}[5m])",
                        "legendFormat": "Bytes Sent/sec",
                        "refId": "A"
                    },
                    {
                        "expr": "rate(system_network_io_bytes_total{direction=\"recv\"}[5m])",
                        "legendFormat": "Bytes Received/sec",
                        "refId": "B"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "binBps"
                    }
                }
            ),
            DashboardPanel(
                id=4,
                title="System Load Average",
                type="timeseries",
                targets=[
                    {
                        "expr": "system_load_average{period=\"1min\"}",
                        "legendFormat": "1 minute",
                        "refId": "A"
                    },
                    {
                        "expr": "system_load_average{period=\"5min\"}",
                        "legendFormat": "5 minutes",
                        "refId": "B"
                    },
                    {
                        "expr": "system_load_average{period=\"15min\"}",
                        "legendFormat": "15 minutes",
                        "refId": "C"
                    }
                ],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                }
            )
        ]
        
        return DashboardConfig(
            uid="system-overview",
            title="System Overview",
            description="System resource monitoring dashboard",
            tags=["system", "monitoring", "infrastructure"],
            panels=panels
        )
    
    def _create_business_metrics_template(self) -> DashboardConfig:
        """Create business metrics dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="API Requests Rate",
                type="stat",
                targets=[{
                    "expr": "rate(api_requests_total[5m])",
                    "legendFormat": "Requests/sec",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "reqps"
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="Active Users",
                type="stat",
                targets=[{
                    "expr": "active_users_count",
                    "legendFormat": "Active Users",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "short"
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="API Response Times",
                type="timeseries",
                targets=[{
                    "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "s"
                    }
                }
            ),
            DashboardPanel(
                id=4,
                title="Error Rate by Component",
                type="timeseries",
                targets=[{
                    "expr": "error_rate_percent",
                    "legendFormat": "{{component}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percent"
                    }
                }
            )
        ]
        
        return DashboardConfig(
            uid="business-metrics",
            title="Business Metrics",
            description="Business logic and application metrics",
            tags=["business", "application", "performance"],
            panels=panels
        )
    
    def _create_stock_analysis_template(self) -> DashboardConfig:
        """Create stock analysis dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="Stocks Analyzed",
                type="stat",
                targets=[{
                    "expr": "increase(stocks_analyzed_total[1h])",
                    "legendFormat": "Stocks/hour",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "short"
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="Spring Festival Patterns",
                type="stat",
                targets=[{
                    "expr": "sum(spring_festival_patterns_detected)",
                    "legendFormat": "Patterns Detected",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "short"
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="Analysis Types",
                type="piechart",
                targets=[{
                    "expr": "stocks_analyzed_total",
                    "legendFormat": "{{analysis_type}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=4,
                title="Risk Calculations",
                type="timeseries",
                targets=[{
                    "expr": "rate(risk_calculations_total[5m])",
                    "legendFormat": "{{calculation_type}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "ops"
                    }
                }
            )
        ]
        
        return DashboardConfig(
            uid="stock-analysis",
            title="Stock Analysis Metrics",
            description="Stock analysis and pattern detection metrics",
            tags=["stock", "analysis", "patterns"],
            panels=panels
        )
    
    def _create_ml_models_template(self) -> DashboardConfig:
        """Create ML models dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="Model Predictions",
                type="stat",
                targets=[{
                    "expr": "rate(ml_model_predictions_total[5m])",
                    "legendFormat": "Predictions/sec",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "ops"
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="Model Accuracy",
                type="stat",
                targets=[{
                    "expr": "avg(ml_model_accuracy_score)",
                    "legendFormat": "Average Accuracy",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "percentunit",
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.85}
                            ]
                        }
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="Predictions by Model",
                type="timeseries",
                targets=[{
                    "expr": "rate(ml_model_predictions_total[5m])",
                    "legendFormat": "{{model_name}} {{model_version}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "ops"
                    }
                }
            ),
            DashboardPanel(
                id=4,
                title="Model Accuracy Over Time",
                type="timeseries",
                targets=[{
                    "expr": "ml_model_accuracy_score",
                    "legendFormat": "{{model_name}} {{model_version}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percentunit",
                        "min": 0,
                        "max": 1
                    }
                }
            )
        ]
        
        return DashboardConfig(
            uid="ml-models",
            title="ML Models Performance",
            description="Machine learning models performance and accuracy",
            tags=["ml", "models", "performance"],
            panels=panels
        )
    
    def _create_api_performance_template(self) -> DashboardConfig:
        """Create API performance dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="Request Rate",
                type="timeseries",
                targets=[{
                    "expr": "rate(api_requests_total[5m])",
                    "legendFormat": "{{method}} {{endpoint}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "reqps"
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="Response Time Percentiles",
                type="timeseries",
                targets=[
                    {
                        "expr": "histogram_quantile(0.50, rate(api_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "50th percentile",
                        "refId": "A"
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile",
                        "refId": "B"
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(api_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "99th percentile",
                        "refId": "C"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "s"
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="Status Code Distribution",
                type="piechart",
                targets=[{
                    "expr": "increase(api_requests_total[1h])",
                    "legendFormat": "{{status}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8}
            )
        ]
        
        return DashboardConfig(
            uid="api-performance",
            title="API Performance",
            description="API performance and response time metrics",
            tags=["api", "performance", "response-time"],
            panels=panels
        )
    
    def _create_error_monitoring_template(self) -> DashboardConfig:
        """Create error monitoring dashboard template"""
        panels = [
            DashboardPanel(
                id=1,
                title="Overall Error Rate",
                type="stat",
                targets=[{
                    "expr": "avg(error_rate_percent)",
                    "legendFormat": "Error Rate %",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "percent",
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 1},
                                {"color": "red", "value": 5}
                            ]
                        }
                    }
                }
            ),
            DashboardPanel(
                id=2,
                title="4xx Errors",
                type="stat",
                targets=[{
                    "expr": "rate(api_requests_total{status=~\"4..\"}[5m])",
                    "legendFormat": "4xx/sec",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "reqps"
                    }
                }
            ),
            DashboardPanel(
                id=3,
                title="5xx Errors",
                type="stat",
                targets=[{
                    "expr": "rate(api_requests_total{status=~\"5..\"}[5m])",
                    "legendFormat": "5xx/sec",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 6, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "unit": "reqps",
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 0.1},
                                {"color": "red", "value": 1}
                            ]
                        }
                    }
                }
            ),
            DashboardPanel(
                id=4,
                title="Error Rate by Component",
                type="timeseries",
                targets=[{
                    "expr": "error_rate_percent",
                    "legendFormat": "{{component}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percent"
                    }
                }
            )
        ]
        
        return DashboardConfig(
            uid="error-monitoring",
            title="Error Monitoring",
            description="Error tracking and monitoring dashboard",
            tags=["errors", "monitoring", "alerts"],
            panels=panels
        )
    
    def create_dashboard(self, template_name: str, 
                        custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a dashboard from template.
        
        Args:
            template_name: Name of the dashboard template
            custom_config: Custom configuration overrides
            
        Returns:
            Dashboard configuration dictionary
        """
        if template_name not in self.dashboard_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        config = self.dashboard_templates[template_name]
        
        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Convert to Grafana dashboard format
        dashboard = {
            "dashboard": {
                "uid": config.uid,
                "title": config.title,
                "description": config.description,
                "tags": config.tags,
                "timezone": config.timezone,
                "refresh": config.refresh,
                "time": config.time_range,
                "panels": [self._panel_to_dict(panel) for panel in config.panels],
                "schemaVersion": 30,
                "version": 1,
                "editable": True,
                "gnetId": None,
                "graphTooltip": 0,
                "id": None,
                "links": [],
                "style": "dark",
                "templating": {"list": []},
                "timepicker": {},
                "annotations": {"list": []}
            },
            "overwrite": True
        }
        
        return dashboard
    
    def _panel_to_dict(self, panel: DashboardPanel) -> Dict[str, Any]:
        """Convert panel object to dictionary"""
        panel_dict = {
            "id": panel.id,
            "title": panel.title,
            "type": panel.type,
            "targets": panel.targets,
            "gridPos": panel.gridPos,
            "options": panel.options,
            "fieldConfig": panel.fieldConfig
        }
        
        if panel.datasource:
            panel_dict["datasource"] = panel.datasource
        
        return panel_dict
    
    def deploy_dashboard(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy dashboard to Grafana.
        
        Args:
            dashboard: Dashboard configuration
            
        Returns:
            Deployment response
        """
        try:
            url = f"{self.config.url}/api/dashboards/db"
            response = self.session.post(
                url,
                json=dashboard,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            
            result = response.json()
            self.logger.info(f"Deployed dashboard: {dashboard['dashboard']['title']}")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error deploying dashboard: {e}")
            raise
    
    def deploy_all_dashboards(self) -> Dict[str, Any]:
        """Deploy all dashboard templates"""
        results = {}
        
        for template_name in self.dashboard_templates.keys():
            try:
                dashboard = self.create_dashboard(template_name)
                result = self.deploy_dashboard(dashboard)
                results[template_name] = {
                    "status": "success",
                    "uid": result.get("uid"),
                    "url": result.get("url")
                }
            except Exception as e:
                results[template_name] = {
                    "status": "error",
                    "error": str(e)
                }
                self.logger.error(f"Failed to deploy {template_name}: {e}")
        
        return results
    
    def get_dashboard(self, uid: str) -> Dict[str, Any]:
        """Get dashboard by UID"""
        try:
            url = f"{self.config.url}/api/dashboards/uid/{uid}"
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting dashboard {uid}: {e}")
            raise
    
    def delete_dashboard(self, uid: str) -> bool:
        """Delete dashboard by UID"""
        try:
            url = f"{self.config.url}/api/dashboards/uid/{uid}"
            response = self.session.delete(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            
            self.logger.info(f"Deleted dashboard: {uid}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error deleting dashboard {uid}: {e}")
            return False
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards"""
        try:
            url = f"{self.config.url}/api/search?type=dash-db"
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error listing dashboards: {e}")
            raise
    
    def backup_dashboards(self, backup_path: str) -> bool:
        """Backup all dashboards to file"""
        try:
            dashboards = self.list_dashboards()
            backup_data = []
            
            for dashboard_info in dashboards:
                dashboard = self.get_dashboard(dashboard_info['uid'])
                backup_data.append(dashboard)
            
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Backed up {len(backup_data)} dashboards to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up dashboards: {e}")
            return False
    
    def restore_dashboards(self, backup_path: str) -> Dict[str, Any]:
        """Restore dashboards from backup file"""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            results = {}
            for dashboard_data in backup_data:
                dashboard = {"dashboard": dashboard_data["dashboard"], "overwrite": True}
                try:
                    result = self.deploy_dashboard(dashboard)
                    uid = dashboard_data["dashboard"]["uid"]
                    results[uid] = {"status": "success", "result": result}
                except Exception as e:
                    results[uid] = {"status": "error", "error": str(e)}
            
            self.logger.info(f"Restored {len(backup_data)} dashboards from {backup_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error restoring dashboards: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Grafana connection"""
        try:
            url = f"{self.config.url}/api/health"
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "grafana_version": response.json().get("version", "unknown"),
                "url": self.config.url,
                "org_id": self.config.org_id
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "url": self.config.url
            }


# Example usage
if __name__ == "__main__":
    import os
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Grafana configuration
    config = GrafanaConfig(
        url=os.getenv("GRAFANA_URL", "http://localhost:3000"),
        api_key=os.getenv("GRAFANA_API_KEY", "your-api-key"),
        org_id=1
    )
    
    # Create dashboard manager
    manager = GrafanaDashboardManager(config)
    
    # Check health
    health = manager.get_health_status()
    print(f"Grafana health: {health}")
    
    if health["status"] == "healthy":
        # Deploy all dashboards
        results = manager.deploy_all_dashboards()
        print(f"Deployment results: {results}")
        
        # List dashboards
        dashboards = manager.list_dashboards()
        print(f"Found {len(dashboards)} dashboards")
        
        # Backup dashboards
        manager.backup_dashboards("dashboards_backup.json")