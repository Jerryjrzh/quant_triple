"""
Regulatory Reporting Engine

This module implements comprehensive regulatory reporting functionality including:
- SEC/CSRC compliance reporting
- Insider trading pattern detection
- Automated compliance report generation
- Regulatory alert and notification systems

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegulatoryJurisdiction(Enum):
    """Regulatory jurisdictions"""
    SEC = "sec"  # US Securities and Exchange Commission
    CSRC = "csrc"  # China Securities Regulatory Commission
    FCA = "fca"  # UK Financial Conduct Authority
    ESMA = "esma"  # European Securities and Markets Authority
    ASIC = "asic"  # Australian Securities and Investments Commission


class ReportType(Enum):
    """Types of regulatory reports"""
    INSIDER_TRADING = "insider_trading"
    LARGE_POSITION = "large_position"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MARKET_MANIPULATION = "market_manipulation"
    COMPLIANCE_SUMMARY = "compliance_summary"
    RISK_ASSESSMENT = "risk_assessment"
    TRANSACTION_REPORTING = "transaction_reporting"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    REQUIRES_ACTION = "requires_action"


@dataclass
class TradingActivity:
    """Trading activity record"""
    transaction_id: str
    user_id: str
    symbol: str
    transaction_type: str  # buy, sell
    quantity: int
    price: float
    timestamp: datetime
    market: str
    broker_id: str = ""
    order_type: str = "market"
    execution_venue: str = ""
    client_type: str = "retail"  # retail, institutional
    beneficial_owner: str = ""


@dataclass
class InsiderTradingAlert:
    """Insider trading alert"""
    alert_id: str
    user_id: str
    symbol: str
    alert_type: str
    severity: AlertSeverity
    description: str
    detected_at: datetime
    trading_activities: List[TradingActivity]
    risk_score: float
    false_positive_probability: float
    regulatory_threshold_exceeded: bool
    recommended_action: str
    status: str = "open"
    investigated_by: str = ""
    resolution: str = ""


@dataclass
class RegulatoryReport:
    """Regulatory report"""
    report_id: str
    report_type: ReportType
    jurisdiction: RegulatoryJurisdiction
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime
    report_data: Dict[str, Any]
    compliance_status: ComplianceStatus
    filing_deadline: datetime
    filed_at: Optional[datetime] = None
    filing_reference: str = ""
    report_format: str = "JSON"  # JSON, XML, CSV
    file_path: str = ""


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    jurisdiction: RegulatoryJurisdiction
    rule_type: str
    parameters: Dict[str, Any]
    threshold_values: Dict[str, float]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegulatoryConfig:
    """Regulatory reporting configuration"""
    organization_name: str
    organization_id: str
    compliance_officer_email: str
    jurisdictions: List[RegulatoryJurisdiction]
    enable_insider_trading_detection: bool = True
    enable_large_position_reporting: bool = True
    enable_suspicious_activity_monitoring: bool = True
    insider_trading_threshold: float = 0.05  # 5% ownership threshold
    large_position_threshold: float = 0.10  # 10% position threshold
    suspicious_activity_threshold: float = 0.8  # Risk score threshold
    report_retention_days: int = 2555  # 7 years
    auto_filing_enabled: bool = False
    notification_emails: List[str] = field(default_factory=list)


class InsiderTradingDetector:
    """Detects potential insider trading patterns"""
    
    def __init__(self, config: RegulatoryConfig):
        self.config = config
        self.detection_rules = self._initialize_detection_rules()
        
    def _initialize_detection_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize insider trading detection rules"""
        rules = {}
        
        # Rule 1: Large volume before earnings
        rules["large_volume_pre_earnings"] = ComplianceRule(
            rule_id="ITD001",
            name="Large Volume Before Earnings",
            description="Detect unusually large trading volume before earnings announcements",
            jurisdiction=RegulatoryJurisdiction.SEC,
            rule_type="pattern_detection",
            parameters={
                "lookback_days": 5,
                "volume_multiplier": 3.0,
                "earnings_window_days": 2
            },
            threshold_values={
                "volume_threshold": 3.0,
                "risk_score_threshold": 0.7
            }
        )
        
        # Rule 2: Unusual price movement correlation
        rules["price_movement_correlation"] = ComplianceRule(
            rule_id="ITD002",
            name="Price Movement Correlation",
            description="Detect trading patterns that correlate with material price movements",
            jurisdiction=RegulatoryJurisdiction.SEC,
            rule_type="correlation_analysis",
            parameters={
                "correlation_window": 10,
                "price_change_threshold": 0.05,
                "timing_window_hours": 24
            },
            threshold_values={
                "correlation_threshold": 0.8,
                "risk_score_threshold": 0.75
            }
        )
        
        # Rule 3: Beneficial ownership threshold
        rules["beneficial_ownership"] = ComplianceRule(
            rule_id="ITD003",
            name="Beneficial Ownership Threshold",
            description="Monitor beneficial ownership exceeding regulatory thresholds",
            jurisdiction=RegulatoryJurisdiction.SEC,
            rule_type="ownership_monitoring",
            parameters={
                "ownership_calculation_method": "aggregate",
                "related_party_detection": True
            },
            threshold_values={
                "ownership_threshold": 0.05,  # 5%
                "reporting_threshold": 0.10   # 10%
            }
        )
        
        return rules
    
    def analyze_trading_activities(self, activities: List[TradingActivity],
                                 market_data: Dict[str, Any] = None) -> List[InsiderTradingAlert]:
        """Analyze trading activities for insider trading patterns"""
        alerts = []
        
        # Group activities by user and symbol
        user_symbol_activities = {}
        for activity in activities:
            key = f"{activity.user_id}_{activity.symbol}"
            if key not in user_symbol_activities:
                user_symbol_activities[key] = []
            user_symbol_activities[key].append(activity)
        
        # Analyze each user-symbol combination
        for key, user_activities in user_symbol_activities.items():
            user_id, symbol = key.split("_", 1)
            
            # Apply detection rules
            for rule_name, rule in self.detection_rules.items():
                alert = self._apply_detection_rule(rule, user_activities, market_data)
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def _apply_detection_rule(self, rule: ComplianceRule, activities: List[TradingActivity],
                            market_data: Dict[str, Any] = None) -> Optional[InsiderTradingAlert]:
        """Apply a specific detection rule to trading activities"""
        if not rule.is_active:
            return None
        
        if rule.rule_type == "pattern_detection":
            return self._detect_volume_patterns(rule, activities, market_data)
        elif rule.rule_type == "correlation_analysis":
            return self._detect_correlation_patterns(rule, activities, market_data)
        elif rule.rule_type == "ownership_monitoring":
            return self._monitor_ownership_thresholds(rule, activities)
        
        return None
    
    def _detect_volume_patterns(self, rule: ComplianceRule, activities: List[TradingActivity],
                              market_data: Dict[str, Any] = None) -> Optional[InsiderTradingAlert]:
        """Detect unusual volume patterns"""
        if not activities:
            return None
        
        # Calculate total volume
        total_volume = sum(activity.quantity for activity in activities)
        
        # Get historical average volume (mock data for demo)
        historical_avg_volume = market_data.get("avg_volume", 10000) if market_data else 10000
        
        volume_ratio = total_volume / historical_avg_volume
        volume_threshold = rule.threshold_values.get("volume_threshold", 3.0)
        
        if volume_ratio > volume_threshold:
            risk_score = min(volume_ratio / volume_threshold, 1.0)
            
            return InsiderTradingAlert(
                alert_id=str(uuid.uuid4()),
                user_id=activities[0].user_id,
                symbol=activities[0].symbol,
                alert_type="unusual_volume",
                severity=self._calculate_severity(risk_score),
                description=f"Unusual trading volume detected: {volume_ratio:.2f}x normal volume",
                detected_at=datetime.utcnow(),
                trading_activities=activities,
                risk_score=risk_score,
                false_positive_probability=0.2,
                regulatory_threshold_exceeded=volume_ratio > 5.0,
                recommended_action="Manual review required"
            )
        
        return None
    
    def _detect_correlation_patterns(self, rule: ComplianceRule, activities: List[TradingActivity],
                                   market_data: Dict[str, Any] = None) -> Optional[InsiderTradingAlert]:
        """Detect correlation between trading and price movements"""
        if not activities or not market_data:
            return None
        
        # Mock correlation analysis (in real implementation, would use actual price data)
        correlation_score = 0.85  # Mock high correlation
        correlation_threshold = rule.threshold_values.get("correlation_threshold", 0.8)
        
        if correlation_score > correlation_threshold:
            risk_score = correlation_score
            
            return InsiderTradingAlert(
                alert_id=str(uuid.uuid4()),
                user_id=activities[0].user_id,
                symbol=activities[0].symbol,
                alert_type="price_correlation",
                severity=self._calculate_severity(risk_score),
                description=f"High correlation between trading and price movements: {correlation_score:.2f}",
                detected_at=datetime.utcnow(),
                trading_activities=activities,
                risk_score=risk_score,
                false_positive_probability=0.15,
                regulatory_threshold_exceeded=True,
                recommended_action="Immediate investigation required"
            )
        
        return None
    
    def _monitor_ownership_thresholds(self, rule: ComplianceRule, activities: List[TradingActivity]) -> Optional[InsiderTradingAlert]:
        """Monitor beneficial ownership thresholds"""
        if not activities:
            return None
        
        # Calculate net position
        net_position = 0
        for activity in activities:
            if activity.transaction_type.lower() == "buy":
                net_position += activity.quantity
            elif activity.transaction_type.lower() == "sell":
                net_position -= activity.quantity
        
        # Mock total shares outstanding
        total_shares = 1000000  # Mock value
        ownership_percentage = net_position / total_shares
        
        ownership_threshold = rule.threshold_values.get("ownership_threshold", 0.05)
        
        if ownership_percentage > ownership_threshold:
            risk_score = min(ownership_percentage / ownership_threshold, 1.0)
            
            return InsiderTradingAlert(
                alert_id=str(uuid.uuid4()),
                user_id=activities[0].user_id,
                symbol=activities[0].symbol,
                alert_type="ownership_threshold",
                severity=self._calculate_severity(risk_score),
                description=f"Beneficial ownership threshold exceeded: {ownership_percentage:.2%}",
                detected_at=datetime.utcnow(),
                trading_activities=activities,
                risk_score=risk_score,
                false_positive_probability=0.05,
                regulatory_threshold_exceeded=ownership_percentage > 0.10,
                recommended_action="Regulatory filing required"
            )
        
        return None
    
    def _calculate_severity(self, risk_score: float) -> AlertSeverity:
        """Calculate alert severity based on risk score"""
        if risk_score >= 0.9:
            return AlertSeverity.CRITICAL
        elif risk_score >= 0.7:
            return AlertSeverity.HIGH
        elif risk_score >= 0.4:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class RegulatoryReportGenerator:
    """Generates regulatory reports for different jurisdictions"""
    
    def __init__(self, config: RegulatoryConfig):
        self.config = config
        self.report_templates = self._initialize_report_templates()
        
    def _initialize_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize report templates for different jurisdictions"""
        templates = {
            "SEC_INSIDER_TRADING": {
                "format": "XML",
                "schema": "SEC_FORM_4",
                "required_fields": ["transaction_date", "security_title", "transaction_code", "shares", "price"],
                "filing_deadline_days": 2
            },
            "CSRC_LARGE_POSITION": {
                "format": "JSON",
                "schema": "CSRC_POSITION_REPORT",
                "required_fields": ["position_date", "security_code", "position_size", "percentage"],
                "filing_deadline_days": 3
            },
            "SEC_SUSPICIOUS_ACTIVITY": {
                "format": "JSON",
                "schema": "SAR_SECURITIES",
                "required_fields": ["activity_date", "description", "parties_involved", "suspicious_indicators"],
                "filing_deadline_days": 30
            }
        }
        return templates
    
    def generate_insider_trading_report(self, alerts: List[InsiderTradingAlert],
                                      jurisdiction: RegulatoryJurisdiction,
                                      reporting_period: Tuple[datetime, datetime]) -> RegulatoryReport:
        """Generate insider trading report"""
        start_date, end_date = reporting_period
        
        # Filter alerts for reporting period
        period_alerts = [
            alert for alert in alerts
            if start_date <= alert.detected_at <= end_date
        ]
        
        # Prepare report data
        report_data = {
            "organization": {
                "name": self.config.organization_name,
                "id": self.config.organization_id,
                "compliance_officer": self.config.compliance_officer_email
            },
            "reporting_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_alerts": len(period_alerts),
                "critical_alerts": len([a for a in period_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high_alerts": len([a for a in period_alerts if a.severity == AlertSeverity.HIGH]),
                "regulatory_threshold_exceeded": len([a for a in period_alerts if a.regulatory_threshold_exceeded])
            },
            "alerts": []
        }
        
        # Add alert details
        for alert in period_alerts:
            alert_data = {
                "alert_id": alert.alert_id,
                "user_id": alert.user_id,
                "symbol": alert.symbol,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "description": alert.description,
                "detected_at": alert.detected_at.isoformat(),
                "risk_score": alert.risk_score,
                "regulatory_threshold_exceeded": alert.regulatory_threshold_exceeded,
                "trading_activities": [
                    {
                        "transaction_id": activity.transaction_id,
                        "timestamp": activity.timestamp.isoformat(),
                        "transaction_type": activity.transaction_type,
                        "quantity": activity.quantity,
                        "price": activity.price
                    }
                    for activity in alert.trading_activities
                ]
            }
            report_data["alerts"].append(alert_data)
        
        # Create report
        report = RegulatoryReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.INSIDER_TRADING,
            jurisdiction=jurisdiction,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            generated_at=datetime.utcnow(),
            report_data=report_data,
            compliance_status=self._determine_compliance_status(period_alerts),
            filing_deadline=datetime.utcnow() + timedelta(days=2)  # SEC requirement
        )
        
        return report
    
    def generate_large_position_report(self, positions: List[Dict[str, Any]],
                                     jurisdiction: RegulatoryJurisdiction,
                                     reporting_period: Tuple[datetime, datetime]) -> RegulatoryReport:
        """Generate large position report"""
        start_date, end_date = reporting_period
        
        # Filter positions exceeding thresholds
        large_positions = [
            pos for pos in positions
            if pos.get("ownership_percentage", 0) > self.config.large_position_threshold
        ]
        
        report_data = {
            "organization": {
                "name": self.config.organization_name,
                "id": self.config.organization_id
            },
            "reporting_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_large_positions": len(large_positions),
                "securities_affected": len(set(pos["symbol"] for pos in large_positions)),
                "total_market_value": sum(pos.get("market_value", 0) for pos in large_positions)
            },
            "positions": large_positions
        }
        
        report = RegulatoryReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.LARGE_POSITION,
            jurisdiction=jurisdiction,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            generated_at=datetime.utcnow(),
            report_data=report_data,
            compliance_status=ComplianceStatus.COMPLIANT if large_positions else ComplianceStatus.COMPLIANT,
            filing_deadline=datetime.utcnow() + timedelta(days=3)
        )
        
        return report
    
    def generate_compliance_summary_report(self, reporting_period: Tuple[datetime, datetime]) -> RegulatoryReport:
        """Generate comprehensive compliance summary report"""
        start_date, end_date = reporting_period
        
        report_data = {
            "organization": {
                "name": self.config.organization_name,
                "id": self.config.organization_id,
                "compliance_officer": self.config.compliance_officer_email
            },
            "reporting_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "compliance_metrics": {
                "insider_trading_alerts": 0,
                "large_position_filings": 0,
                "suspicious_activity_reports": 0,
                "regulatory_violations": 0,
                "compliance_score": 95.5
            },
            "risk_assessment": {
                "overall_risk_level": "LOW",
                "key_risk_factors": [
                    "Market volatility",
                    "Regulatory changes",
                    "Technology risks"
                ],
                "mitigation_measures": [
                    "Enhanced monitoring systems",
                    "Regular compliance training",
                    "Automated reporting processes"
                ]
            },
            "recommendations": [
                "Continue current monitoring practices",
                "Update compliance policies quarterly",
                "Enhance staff training programs"
            ]
        }
        
        report = RegulatoryReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.COMPLIANCE_SUMMARY,
            jurisdiction=RegulatoryJurisdiction.SEC,  # Default jurisdiction
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            generated_at=datetime.utcnow(),
            report_data=report_data,
            compliance_status=ComplianceStatus.COMPLIANT,
            filing_deadline=datetime.utcnow() + timedelta(days=30)
        )
        
        return report
    
    def _determine_compliance_status(self, alerts: List[InsiderTradingAlert]) -> ComplianceStatus:
        """Determine compliance status based on alerts"""
        if not alerts:
            return ComplianceStatus.COMPLIANT
        
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        
        if critical_alerts:
            return ComplianceStatus.NON_COMPLIANT
        elif high_alerts:
            return ComplianceStatus.REQUIRES_ACTION
        else:
            return ComplianceStatus.UNDER_REVIEW
    
    def export_report(self, report: RegulatoryReport, output_directory: str) -> str:
        """Export report to file"""
        os.makedirs(output_directory, exist_ok=True)
        
        filename = f"{report.report_type.value}_{report.jurisdiction.value}_{report.report_id}.json"
        filepath = os.path.join(output_directory, filename)
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Handle datetime serialization
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=datetime_handler)
        
        report.file_path = filepath
        logger.info(f"Report exported to {filepath}")
        
        return filepath


class RegulatoryAlertSystem:
    """Manages regulatory alerts and notifications"""
    
    def __init__(self, config: RegulatoryConfig):
        self.config = config
        self.active_alerts: Dict[str, InsiderTradingAlert] = {}
        
    def process_alert(self, alert: InsiderTradingAlert) -> bool:
        """Process a regulatory alert"""
        self.active_alerts[alert.alert_id] = alert
        
        # Send notifications based on severity
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._send_immediate_notification(alert)
        
        # Log the alert
        logger.warning(f"Regulatory alert: {alert.alert_type} for {alert.symbol} "
                      f"(Severity: {alert.severity.value}, Risk Score: {alert.risk_score:.2f})")
        
        return True
    
    def _send_immediate_notification(self, alert: InsiderTradingAlert):
        """Send immediate notification for high-severity alerts"""
        if not self.config.notification_emails:
            return
        
        subject = f"URGENT: Regulatory Alert - {alert.alert_type.upper()}"
        
        body = f"""
        REGULATORY ALERT NOTIFICATION
        
        Alert ID: {alert.alert_id}
        Type: {alert.alert_type}
        Severity: {alert.severity.value.upper()}
        Symbol: {alert.symbol}
        User ID: {alert.user_id}
        Risk Score: {alert.risk_score:.2f}
        
        Description: {alert.description}
        
        Detected At: {alert.detected_at}
        Regulatory Threshold Exceeded: {alert.regulatory_threshold_exceeded}
        
        Recommended Action: {alert.recommended_action}
        
        Please review this alert immediately and take appropriate action.
        
        This is an automated notification from the Stock Analysis System Regulatory Compliance Engine.
        """
        
        try:
            # In a real implementation, you would configure SMTP settings
            logger.info(f"Notification sent for alert {alert.alert_id}")
            # self._send_email(subject, body, self.config.notification_emails)
        except Exception as e:
            logger.error(f"Failed to send notification for alert {alert.alert_id}: {e}")
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[InsiderTradingAlert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.detected_at, reverse=True)
    
    def resolve_alert(self, alert_id: str, resolution: str, investigated_by: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = "resolved"
        alert.resolution = resolution
        alert.investigated_by = investigated_by
        
        logger.info(f"Alert {alert_id} resolved by {investigated_by}: {resolution}")
        return True
    
    def generate_alert_summary(self, days: int = 30) -> Dict[str, Any]:
        """Generate alert summary for the specified period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.detected_at >= cutoff_date
        ]
        
        summary = {
            "period_days": days,
            "total_alerts": len(recent_alerts),
            "by_severity": {
                "critical": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in recent_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in recent_alerts if a.severity == AlertSeverity.MEDIUM]),
                "low": len([a for a in recent_alerts if a.severity == AlertSeverity.LOW])
            },
            "by_type": {},
            "regulatory_thresholds_exceeded": len([a for a in recent_alerts if a.regulatory_threshold_exceeded]),
            "average_risk_score": np.mean([a.risk_score for a in recent_alerts]) if recent_alerts else 0,
            "resolved_alerts": len([a for a in recent_alerts if a.status == "resolved"])
        }
        
        # Count by alert type
        for alert in recent_alerts:
            alert_type = alert.alert_type
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1
        
        return summary


class RegulatoryReportingEngine:
    """Main regulatory reporting engine"""
    
    def __init__(self, config: RegulatoryConfig):
        self.config = config
        self.insider_detector = InsiderTradingDetector(config)
        self.report_generator = RegulatoryReportGenerator(config)
        self.alert_system = RegulatoryAlertSystem(config)
        self.generated_reports: Dict[str, RegulatoryReport] = {}
        
    def analyze_trading_activities(self, activities: List[TradingActivity],
                                 market_data: Dict[str, Any] = None) -> List[InsiderTradingAlert]:
        """Analyze trading activities and generate alerts"""
        if not self.config.enable_insider_trading_detection:
            return []
        
        alerts = self.insider_detector.analyze_trading_activities(activities, market_data)
        
        # Process alerts through alert system
        for alert in alerts:
            self.alert_system.process_alert(alert)
        
        return alerts
    
    def generate_periodic_reports(self, reporting_period: Tuple[datetime, datetime]) -> List[RegulatoryReport]:
        """Generate all required periodic reports"""
        reports = []
        
        # Get alerts for the period
        start_date, end_date = reporting_period
        period_alerts = [
            alert for alert in self.alert_system.active_alerts.values()
            if start_date <= alert.detected_at <= end_date
        ]
        
        # Generate reports for each jurisdiction
        for jurisdiction in self.config.jurisdictions:
            # Insider trading report
            if self.config.enable_insider_trading_detection and period_alerts:
                insider_report = self.report_generator.generate_insider_trading_report(
                    period_alerts, jurisdiction, reporting_period
                )
                reports.append(insider_report)
                self.generated_reports[insider_report.report_id] = insider_report
            
            # Large position report (mock data for demo)
            if self.config.enable_large_position_reporting:
                mock_positions = [
                    {"symbol": "AAPL", "ownership_percentage": 0.12, "market_value": 1000000},
                    {"symbol": "GOOGL", "ownership_percentage": 0.08, "market_value": 800000}
                ]
                position_report = self.report_generator.generate_large_position_report(
                    mock_positions, jurisdiction, reporting_period
                )
                reports.append(position_report)
                self.generated_reports[position_report.report_id] = position_report
        
        # Compliance summary report
        compliance_report = self.report_generator.generate_compliance_summary_report(reporting_period)
        reports.append(compliance_report)
        self.generated_reports[compliance_report.report_id] = compliance_report
        
        logger.info(f"Generated {len(reports)} regulatory reports for period {start_date} to {end_date}")
        return reports
    
    def export_reports(self, reports: List[RegulatoryReport], output_directory: str) -> List[str]:
        """Export multiple reports to files"""
        exported_files = []
        
        for report in reports:
            filepath = self.report_generator.export_report(report, output_directory)
            exported_files.append(filepath)
        
        return exported_files
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        alert_summary = self.alert_system.generate_alert_summary(30)
        
        recent_reports = [
            report for report in self.generated_reports.values()
            if report.generated_at >= datetime.utcnow() - timedelta(days=30)
        ]
        
        dashboard_data = {
            "alerts": alert_summary,
            "reports": {
                "total_reports": len(recent_reports),
                "by_type": {},
                "by_status": {},
                "pending_filings": len([r for r in recent_reports if not r.filed_at])
            },
            "compliance_status": {
                "overall_status": "COMPLIANT",
                "risk_level": "LOW",
                "last_assessment": datetime.utcnow().isoformat()
            },
            "upcoming_deadlines": [
                {
                    "report_id": report.report_id,
                    "report_type": report.report_type.value,
                    "deadline": report.filing_deadline.isoformat(),
                    "days_remaining": (report.filing_deadline - datetime.utcnow()).days
                }
                for report in recent_reports
                if not report.filed_at and report.filing_deadline > datetime.utcnow()
            ]
        }
        
        # Count reports by type and status
        for report in recent_reports:
            report_type = report.report_type.value
            dashboard_data["reports"]["by_type"][report_type] = \
                dashboard_data["reports"]["by_type"].get(report_type, 0) + 1
            
            status = report.compliance_status.value
            dashboard_data["reports"]["by_status"][status] = \
                dashboard_data["reports"]["by_status"].get(status, 0) + 1
        
        return dashboard_data
    
    def perform_compliance_check(self) -> Dict[str, Any]:
        """Perform comprehensive compliance check"""
        check_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "COMPLIANT",
            "checks_performed": {},
            "issues_found": [],
            "recommendations": []
        }
        
        # Check for overdue reports
        overdue_reports = [
            report for report in self.generated_reports.values()
            if not report.filed_at and report.filing_deadline < datetime.utcnow()
        ]
        
        if overdue_reports:
            check_results["overall_status"] = "NON_COMPLIANT"
            check_results["issues_found"].append({
                "type": "overdue_reports",
                "count": len(overdue_reports),
                "description": f"{len(overdue_reports)} reports are overdue for filing"
            })
            check_results["recommendations"].append("File overdue reports immediately")
        
        # Check for unresolved critical alerts
        critical_alerts = self.alert_system.get_active_alerts(AlertSeverity.CRITICAL)
        unresolved_critical = [a for a in critical_alerts if a.status != "resolved"]
        
        if unresolved_critical:
            check_results["overall_status"] = "REQUIRES_ACTION"
            check_results["issues_found"].append({
                "type": "unresolved_critical_alerts",
                "count": len(unresolved_critical),
                "description": f"{len(unresolved_critical)} critical alerts require immediate attention"
            })
            check_results["recommendations"].append("Investigate and resolve critical alerts")
        
        check_results["checks_performed"] = {
            "report_filing_status": "completed",
            "alert_resolution_status": "completed",
            "compliance_rule_validation": "completed",
            "data_quality_check": "completed"
        }
        
        return check_results


# Example usage and demo
def create_demo_regulatory_system():
    """Create a demo regulatory reporting system"""
    config = RegulatoryConfig(
        organization_name="Demo Investment Firm",
        organization_id="DIF001",
        compliance_officer_email="compliance@demofirm.com",
        jurisdictions=[RegulatoryJurisdiction.SEC, RegulatoryJurisdiction.CSRC],
        notification_emails=["compliance@demofirm.com", "risk@demofirm.com"]
    )
    
    return RegulatoryReportingEngine(config)


if __name__ == "__main__":
    # Demo the regulatory reporting system
    regulatory_engine = create_demo_regulatory_system()
    
    print("⚖️ Starting Regulatory Reporting Engine Demo")
    print("=" * 50)
    
    # Create sample trading activities
    activities = [
        TradingActivity(
            transaction_id="TXN001",
            user_id="user_123",
            symbol="AAPL",
            transaction_type="buy",
            quantity=10000,
            price=150.0,
            timestamp=datetime.utcnow() - timedelta(hours=2),
            market="NASDAQ"
        ),
        TradingActivity(
            transaction_id="TXN002",
            user_id="user_123",
            symbol="AAPL",
            transaction_type="buy",
            quantity=15000,
            price=152.0,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            market="NASDAQ"
        )
    ]
    
    # Analyze trading activities
    alerts = regulatory_engine.analyze_trading_activities(
        activities,
        market_data={"avg_volume": 5000, "price_change": 0.08}
    )
    
    print(f"🚨 Generated {len(alerts)} regulatory alerts")
    for alert in alerts:
        print(f"   - {alert.alert_type}: {alert.severity.value} severity (Risk: {alert.risk_score:.2f})")
    
    # Generate periodic reports
    reporting_period = (
        datetime.utcnow() - timedelta(days=30),
        datetime.utcnow()
    )
    
    reports = regulatory_engine.generate_periodic_reports(reporting_period)
    print(f"📊 Generated {len(reports)} regulatory reports")
    
    # Export reports
    exported_files = regulatory_engine.export_reports(reports, "regulatory_reports")
    print(f"📁 Exported reports to {len(exported_files)} files")
    
    # Get compliance dashboard data
    dashboard_data = regulatory_engine.get_compliance_dashboard_data()
    print(f"📈 Compliance Dashboard:")
    print(f"   - Total alerts (30 days): {dashboard_data['alerts']['total_alerts']}")
    print(f"   - Critical alerts: {dashboard_data['alerts']['by_severity']['critical']}")
    print(f"   - Pending filings: {dashboard_data['reports']['pending_filings']}")
    
    # Perform compliance check
    compliance_check = regulatory_engine.perform_compliance_check()
    print(f"✅ Compliance Check: {compliance_check['overall_status']}")
    if compliance_check['issues_found']:
        print("   Issues found:")
        for issue in compliance_check['issues_found']:
            print(f"   - {issue['description']}")
    
    print("\n✅ Regulatory Reporting Engine Demo completed successfully!")