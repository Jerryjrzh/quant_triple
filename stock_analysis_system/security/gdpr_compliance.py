"""
GDPR Compliance System

This module implements comprehensive GDPR compliance functionality including:
- Data subject request handling (access, deletion, portability)
- Consent management and tracking
- Audit logging for GDPR compliance
- Data retention and anonymization

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path
import zipfile
import csv
import pandas as pd
from cryptography.fernet import Fernet


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent"""
    ESSENTIAL = "essential"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    PERSONALIZATION = "personalization"
    THIRD_PARTY = "third_party"


class RequestType(Enum):
    """Types of data subject requests"""
    ACCESS = "access"
    DELETION = "deletion"
    PORTABILITY = "portability"
    RECTIFICATION = "rectification"
    RESTRICTION = "restriction"
    OBJECTION = "objection"


class RequestStatus(Enum):
    """Status of data subject requests"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ConsentRecord:
    """Consent record for GDPR compliance"""
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: datetime
    ip_address: str
    user_agent: str
    consent_version: str
    withdrawal_timestamp: Optional[datetime] = None
    legal_basis: str = "consent"
    purpose: str = ""
    data_categories: List[str] = field(default_factory=list)
    retention_period: Optional[int] = None  # days
    third_parties: List[str] = field(default_factory=list)


@dataclass
class DataSubjectRequest:
    """Data subject request under GDPR"""
    request_id: str
    user_id: str
    request_type: RequestType
    status: RequestStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    requester_email: str = ""
    requester_ip: str = ""
    description: str = ""
    response_data: Optional[Dict[str, Any]] = None
    rejection_reason: str = ""
    assigned_to: str = ""
    priority: str = "normal"  # low, normal, high
    deadline: Optional[datetime] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for GDPR compliance"""
    log_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    legal_basis: str
    data_categories: List[str] = field(default_factory=list)
    retention_applied: bool = False


@dataclass
class DataCategory:
    """Data category definition"""
    category_id: str
    name: str
    description: str
    legal_basis: str
    retention_period: int  # days
    is_sensitive: bool = False
    requires_consent: bool = True
    can_be_anonymized: bool = True
    third_party_sharing: bool = False


@dataclass
class GDPRConfig:
    """GDPR compliance configuration"""
    organization_name: str
    dpo_contact: str  # Data Protection Officer
    default_retention_days: int = 2555  # 7 years
    request_response_days: int = 30
    enable_audit_logging: bool = True
    enable_consent_tracking: bool = True
    enable_data_minimization: bool = True
    anonymization_delay_days: int = 30
    backup_retention_days: int = 90
    encryption_key_rotation_days: int = 90


class ConsentManager:
    """Manages user consent for GDPR compliance"""
    
    def __init__(self, config: GDPRConfig):
        self.config = config
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.consent_versions = {
            "v1.0": "Initial consent version",
            "v1.1": "Updated privacy policy with analytics",
            "v2.0": "Enhanced data processing consent"
        }
        self.current_version = "v2.0"
        
    def record_consent(self, user_id: str, consent_type: ConsentType, 
                      granted: bool, ip_address: str, user_agent: str,
                      purpose: str = "", data_categories: List[str] = None,
                      retention_period: int = None, third_parties: List[str] = None) -> ConsentRecord:
        """Record user consent"""
        consent = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            consent_version=self.current_version,
            purpose=purpose,
            data_categories=data_categories or [],
            retention_period=retention_period or self.config.default_retention_days,
            third_parties=third_parties or []
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent)
        
        logger.info(f"Consent recorded for user {user_id}: {consent_type.value} = {granted}")
        return consent
    
    def withdraw_consent(self, user_id: str, consent_type: ConsentType,
                        ip_address: str, user_agent: str) -> bool:
        """Withdraw user consent"""
        if user_id not in self.consent_records:
            return False
        
        # Find the latest consent record for this type
        for consent in reversed(self.consent_records[user_id]):
            if consent.consent_type == consent_type and consent.granted and not consent.withdrawal_timestamp:
                consent.withdrawal_timestamp = datetime.utcnow()
                
                # Record new withdrawal consent
                self.record_consent(
                    user_id=user_id,
                    consent_type=consent_type,
                    granted=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    purpose="Consent withdrawal"
                )
                
                logger.info(f"Consent withdrawn for user {user_id}: {consent_type.value}")
                return True
        
        return False
    
    def get_current_consent(self, user_id: str, consent_type: ConsentType) -> Optional[ConsentRecord]:
        """Get current consent status for user and type"""
        if user_id not in self.consent_records:
            return None
        
        # Find the latest consent record for this type
        for consent in reversed(self.consent_records[user_id]):
            if consent.consent_type == consent_type:
                return consent
        
        return None
    
    def has_valid_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has valid consent for specific type"""
        consent = self.get_current_consent(user_id, consent_type)
        return consent is not None and consent.granted and not consent.withdrawal_timestamp
    
    def get_all_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for user"""
        return self.consent_records.get(user_id, [])
    
    def get_consent_summary(self, user_id: str) -> Dict[str, bool]:
        """Get summary of current consent status"""
        summary = {}
        for consent_type in ConsentType:
            summary[consent_type.value] = self.has_valid_consent(user_id, consent_type)
        return summary
    
    def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records"""
        cleaned_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.default_retention_days)
        
        for user_id in list(self.consent_records.keys()):
            original_count = len(self.consent_records[user_id])
            self.consent_records[user_id] = [
                consent for consent in self.consent_records[user_id]
                if consent.timestamp > cutoff_date
            ]
            cleaned_count += original_count - len(self.consent_records[user_id])
            
            # Remove empty user records
            if not self.consent_records[user_id]:
                del self.consent_records[user_id]
        
        logger.info(f"Cleaned up {cleaned_count} expired consent records")
        return cleaned_count


class DataSubjectRequestManager:
    """Manages data subject requests under GDPR"""
    
    def __init__(self, config: GDPRConfig):
        self.config = config
        self.requests: Dict[str, DataSubjectRequest] = {}
        self.data_exporters = {
            RequestType.ACCESS: self._export_user_data,
            RequestType.PORTABILITY: self._export_portable_data
        }
        
    def create_request(self, user_id: str, request_type: RequestType,
                      requester_email: str, requester_ip: str,
                      description: str = "") -> DataSubjectRequest:
        """Create a new data subject request"""
        request_id = str(uuid.uuid4())
        deadline = datetime.utcnow() + timedelta(days=self.config.request_response_days)
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            requester_email=requester_email,
            requester_ip=requester_ip,
            description=description,
            deadline=deadline
        )
        
        self.requests[request_id] = request
        
        logger.info(f"Data subject request created: {request_id} ({request_type.value}) for user {user_id}")
        return request
    
    def process_request(self, request_id: str, assigned_to: str = "") -> bool:
        """Start processing a data subject request"""
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        request.status = RequestStatus.IN_PROGRESS
        request.updated_at = datetime.utcnow()
        request.assigned_to = assigned_to
        
        logger.info(f"Processing data subject request: {request_id}")
        
        # Auto-process certain request types
        if request.request_type in [RequestType.ACCESS, RequestType.PORTABILITY]:
            return self._auto_process_data_request(request)
        
        return True
    
    def complete_request(self, request_id: str, response_data: Dict[str, Any] = None) -> bool:
        """Complete a data subject request"""
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        request.updated_at = datetime.utcnow()
        request.response_data = response_data
        
        logger.info(f"Data subject request completed: {request_id}")
        return True
    
    def reject_request(self, request_id: str, reason: str) -> bool:
        """Reject a data subject request"""
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        request.status = RequestStatus.REJECTED
        request.updated_at = datetime.utcnow()
        request.rejection_reason = reason
        
        logger.info(f"Data subject request rejected: {request_id} - {reason}")
        return True
    
    def get_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get a data subject request"""
        return self.requests.get(request_id)
    
    def get_user_requests(self, user_id: str) -> List[DataSubjectRequest]:
        """Get all requests for a user"""
        return [req for req in self.requests.values() if req.user_id == user_id]
    
    def get_pending_requests(self) -> List[DataSubjectRequest]:
        """Get all pending requests"""
        return [req for req in self.requests.values() if req.status == RequestStatus.PENDING]
    
    def get_overdue_requests(self) -> List[DataSubjectRequest]:
        """Get overdue requests"""
        now = datetime.utcnow()
        return [req for req in self.requests.values() 
                if req.deadline and req.deadline < now and req.status != RequestStatus.COMPLETED]
    
    def _auto_process_data_request(self, request: DataSubjectRequest) -> bool:
        """Auto-process data access/portability requests"""
        try:
            if request.request_type in self.data_exporters:
                export_func = self.data_exporters[request.request_type]
                response_data = export_func(request.user_id)
                return self.complete_request(request.request_id, response_data)
            return False
        except Exception as e:
            logger.error(f"Error auto-processing request {request.request_id}: {e}")
            return False
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for access request"""
        # In a real implementation, this would query all relevant databases
        return {
            "user_profile": {
                "user_id": user_id,
                "export_date": datetime.utcnow().isoformat(),
                "data_categories": ["profile", "preferences", "activity_logs"]
            },
            "trading_data": {
                "portfolios": [],
                "transactions": [],
                "watchlists": []
            },
            "system_data": {
                "login_history": [],
                "consent_records": [],
                "preferences": {}
            }
        }
    
    def _export_portable_data(self, user_id: str) -> Dict[str, Any]:
        """Export portable user data"""
        # Similar to access request but in machine-readable format
        data = self._export_user_data(user_id)
        data["format"] = "JSON"
        data["schema_version"] = "1.0"
        return data


class AuditLogger:
    """GDPR-compliant audit logging system"""
    
    def __init__(self, config: GDPRConfig):
        self.config = config
        self.audit_logs: List[AuditLogEntry] = []
        self.encryption_key = self._get_encryption_key()
        
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for audit logs"""
        key_file = "audit_encryption.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def log_action(self, user_id: str, action: str, resource_type: str,
                   resource_id: str, details: Dict[str, Any],
                   ip_address: str, user_agent: str, legal_basis: str,
                   data_categories: List[str] = None) -> AuditLogEntry:
        """Log an action for GDPR compliance"""
        if not self.config.enable_audit_logging:
            return None
        
        log_entry = AuditLogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            legal_basis=legal_basis,
            data_categories=data_categories or []
        )
        
        self.audit_logs.append(log_entry)
        
        # Keep only recent logs in memory (persist to database in production)
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]
        
        return log_entry
    
    def get_user_audit_trail(self, user_id: str, start_date: datetime = None,
                           end_date: datetime = None) -> List[AuditLogEntry]:
        """Get audit trail for a specific user"""
        logs = [log for log in self.audit_logs if log.user_id == user_id]
        
        if start_date:
            logs = [log for log in logs if log.timestamp >= start_date]
        
        if end_date:
            logs = [log for log in logs if log.timestamp <= end_date]
        
        return sorted(logs, key=lambda x: x.timestamp, reverse=True)
    
    def search_audit_logs(self, action: str = None, resource_type: str = None,
                         start_date: datetime = None, end_date: datetime = None,
                         limit: int = 1000) -> List[AuditLogEntry]:
        """Search audit logs with filters"""
        logs = self.audit_logs
        
        if action:
            logs = [log for log in logs if log.action == action]
        
        if resource_type:
            logs = [log for log in logs if log.resource_type == resource_type]
        
        if start_date:
            logs = [log for log in logs if log.timestamp >= start_date]
        
        if end_date:
            logs = [log for log in logs if log.timestamp <= end_date]
        
        return sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def export_audit_logs(self, output_path: str, start_date: datetime = None,
                         end_date: datetime = None) -> str:
        """Export audit logs to file"""
        logs = self.search_audit_logs(start_date=start_date, end_date=end_date)
        
        # Convert to DataFrame for easy export
        log_data = []
        for log in logs:
            log_dict = asdict(log)
            log_dict['details'] = json.dumps(log_dict['details'])
            log_dict['data_categories'] = ','.join(log_dict['data_categories'])
            log_data.append(log_dict)
        
        df = pd.DataFrame(log_data)
        
        # Export to CSV
        csv_path = f"{output_path}/audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Audit logs exported to {csv_path}")
        return csv_path


class DataRetentionManager:
    """Manages data retention and deletion for GDPR compliance"""
    
    def __init__(self, config: GDPRConfig):
        self.config = config
        self.data_categories = self._initialize_data_categories()
        self.retention_policies: Dict[str, int] = {}
        
    def _initialize_data_categories(self) -> Dict[str, DataCategory]:
        """Initialize data categories with retention policies"""
        categories = {
            "user_profile": DataCategory(
                category_id="user_profile",
                name="User Profile Data",
                description="Basic user information and preferences",
                legal_basis="contract",
                retention_period=2555,  # 7 years
                requires_consent=False
            ),
            "trading_data": DataCategory(
                category_id="trading_data",
                name="Trading and Investment Data",
                description="Portfolio, transactions, and trading history",
                legal_basis="contract",
                retention_period=2555,  # 7 years for financial records
                requires_consent=False
            ),
            "analytics_data": DataCategory(
                category_id="analytics_data",
                name="Analytics and Usage Data",
                description="System usage and behavior analytics",
                legal_basis="consent",
                retention_period=730,  # 2 years
                requires_consent=True,
                can_be_anonymized=True
            ),
            "marketing_data": DataCategory(
                category_id="marketing_data",
                name="Marketing and Communication Data",
                description="Marketing preferences and communication history",
                legal_basis="consent",
                retention_period=1095,  # 3 years
                requires_consent=True
            ),
            "support_data": DataCategory(
                category_id="support_data",
                name="Customer Support Data",
                description="Support tickets and communication",
                legal_basis="legitimate_interest",
                retention_period=1825,  # 5 years
                requires_consent=False
            )
        }
        return categories
    
    def set_retention_policy(self, category_id: str, retention_days: int):
        """Set retention policy for a data category"""
        if category_id in self.data_categories:
            self.data_categories[category_id].retention_period = retention_days
            self.retention_policies[category_id] = retention_days
            logger.info(f"Retention policy updated for {category_id}: {retention_days} days")
    
    def get_retention_policy(self, category_id: str) -> int:
        """Get retention policy for a data category"""
        if category_id in self.data_categories:
            return self.data_categories[category_id].retention_period
        return self.config.default_retention_days
    
    def identify_expired_data(self) -> Dict[str, List[str]]:
        """Identify data that has exceeded retention period"""
        expired_data = {}
        cutoff_date = datetime.utcnow()
        
        for category_id, category in self.data_categories.items():
            retention_cutoff = cutoff_date - timedelta(days=category.retention_period)
            
            # In a real implementation, this would query the database
            # For demo purposes, we'll return mock expired data
            expired_data[category_id] = [
                f"record_{i}" for i in range(1, 6)  # Mock expired records
            ]
        
        return expired_data
    
    def anonymize_data(self, category_id: str, record_ids: List[str]) -> int:
        """Anonymize data instead of deleting it"""
        if category_id not in self.data_categories:
            return 0
        
        category = self.data_categories[category_id]
        if not category.can_be_anonymized:
            logger.warning(f"Category {category_id} cannot be anonymized")
            return 0
        
        # In a real implementation, this would anonymize the actual data
        anonymized_count = len(record_ids)
        logger.info(f"Anonymized {anonymized_count} records in category {category_id}")
        
        return anonymized_count
    
    def delete_expired_data(self, dry_run: bool = True) -> Dict[str, int]:
        """Delete or anonymize expired data"""
        expired_data = self.identify_expired_data()
        deletion_summary = {}
        
        for category_id, record_ids in expired_data.items():
            if not record_ids:
                continue
            
            category = self.data_categories[category_id]
            
            if category.can_be_anonymized:
                if not dry_run:
                    count = self.anonymize_data(category_id, record_ids)
                    deletion_summary[f"{category_id}_anonymized"] = count
                else:
                    deletion_summary[f"{category_id}_to_anonymize"] = len(record_ids)
            else:
                if not dry_run:
                    # In a real implementation, this would delete the actual data
                    count = len(record_ids)
                    deletion_summary[f"{category_id}_deleted"] = count
                    logger.info(f"Deleted {count} records in category {category_id}")
                else:
                    deletion_summary[f"{category_id}_to_delete"] = len(record_ids)
        
        return deletion_summary


class GDPRComplianceManager:
    """Main GDPR compliance manager"""
    
    def __init__(self, config: GDPRConfig):
        self.config = config
        self.consent_manager = ConsentManager(config)
        self.request_manager = DataSubjectRequestManager(config)
        self.audit_logger = AuditLogger(config)
        self.retention_manager = DataRetentionManager(config)
        
    def record_consent(self, user_id: str, consent_type: ConsentType,
                      granted: bool, ip_address: str, user_agent: str,
                      **kwargs) -> ConsentRecord:
        """Record user consent with audit logging"""
        consent = self.consent_manager.record_consent(
            user_id, consent_type, granted, ip_address, user_agent, **kwargs
        )
        
        # Log the consent action
        self.audit_logger.log_action(
            user_id=user_id,
            action="consent_recorded",
            resource_type="consent",
            resource_id=f"{user_id}_{consent_type.value}",
            details={
                "consent_type": consent_type.value,
                "granted": granted,
                "version": consent.consent_version
            },
            ip_address=ip_address,
            user_agent=user_agent,
            legal_basis="consent",
            data_categories=["consent_records"]
        )
        
        return consent
    
    def handle_data_subject_request(self, user_id: str, request_type: RequestType,
                                  requester_email: str, requester_ip: str,
                                  description: str = "") -> DataSubjectRequest:
        """Handle a data subject request"""
        request = self.request_manager.create_request(
            user_id, request_type, requester_email, requester_ip, description
        )
        
        # Log the request
        self.audit_logger.log_action(
            user_id=user_id,
            action="data_subject_request_created",
            resource_type="gdpr_request",
            resource_id=request.request_id,
            details={
                "request_type": request_type.value,
                "description": description
            },
            ip_address=requester_ip,
            user_agent="",
            legal_basis="legal_obligation",
            data_categories=["gdpr_requests"]
        )
        
        # Auto-process if possible
        if request_type in [RequestType.ACCESS, RequestType.PORTABILITY]:
            self.request_manager.process_request(request.request_id)
        
        return request
    
    def generate_compliance_report(self, start_date: datetime = None,
                                 end_date: datetime = None) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Consent statistics
        total_consents = sum(len(records) for records in self.consent_manager.consent_records.values())
        
        # Request statistics
        all_requests = list(self.request_manager.requests.values())
        period_requests = [
            req for req in all_requests
            if start_date <= req.created_at <= end_date
        ]
        
        # Audit log statistics
        audit_logs = self.audit_logger.search_audit_logs(
            start_date=start_date, end_date=end_date
        )
        
        # Data retention statistics
        expired_data = self.retention_manager.identify_expired_data()
        
        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.utcnow().isoformat()
            },
            "consent_management": {
                "total_consent_records": total_consents,
                "active_users_with_consent": len(self.consent_manager.consent_records),
                "consent_types": {ct.value: 0 for ct in ConsentType}
            },
            "data_subject_requests": {
                "total_requests": len(period_requests),
                "by_type": {rt.value: 0 for rt in RequestType},
                "by_status": {rs.value: 0 for rs in RequestStatus},
                "average_response_time_days": 0,
                "overdue_requests": len(self.request_manager.get_overdue_requests())
            },
            "audit_logging": {
                "total_audit_entries": len(audit_logs),
                "unique_users": len(set(log.user_id for log in audit_logs)),
                "top_actions": {}
            },
            "data_retention": {
                "categories_with_expired_data": len([k for k, v in expired_data.items() if v]),
                "total_expired_records": sum(len(v) for v in expired_data.values()),
                "retention_policies": len(self.retention_manager.data_categories)
            }
        }
        
        # Calculate detailed statistics
        for req in period_requests:
            report["data_subject_requests"]["by_type"][req.request_type.value] += 1
            report["data_subject_requests"]["by_status"][req.status.value] += 1
        
        # Calculate average response time
        completed_requests = [req for req in period_requests if req.completed_at]
        if completed_requests:
            total_response_time = sum(
                (req.completed_at - req.created_at).days
                for req in completed_requests
            )
            report["data_subject_requests"]["average_response_time_days"] = \
                total_response_time / len(completed_requests)
        
        return report
    
    def perform_data_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """Perform automated data cleanup for GDPR compliance"""
        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "actions_taken": {}
        }
        
        # Clean up expired consent records
        if not dry_run:
            expired_consents = self.consent_manager.cleanup_expired_consents()
            cleanup_results["actions_taken"]["expired_consents_cleaned"] = expired_consents
        
        # Clean up expired data
        data_cleanup = self.retention_manager.delete_expired_data(dry_run=dry_run)
        cleanup_results["actions_taken"]["data_retention"] = data_cleanup
        
        # Mark overdue requests as expired
        overdue_requests = self.request_manager.get_overdue_requests()
        if not dry_run:
            for request in overdue_requests:
                request.status = RequestStatus.EXPIRED
                request.updated_at = datetime.utcnow()
        
        cleanup_results["actions_taken"]["expired_requests"] = len(overdue_requests)
        
        return cleanup_results
    
    def export_user_data(self, user_id: str, output_path: str) -> str:
        """Export all user data for GDPR compliance"""
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "data_categories": {}
        }
        
        # Export consent records
        consents = self.consent_manager.get_all_consents(user_id)
        export_data["data_categories"]["consents"] = [asdict(c) for c in consents]
        
        # Export data subject requests
        requests = self.request_manager.get_user_requests(user_id)
        export_data["data_categories"]["gdpr_requests"] = [asdict(r) for r in requests]
        
        # Export audit trail
        audit_logs = self.audit_logger.get_user_audit_trail(user_id)
        export_data["data_categories"]["audit_logs"] = [asdict(log) for log in audit_logs]
        
        # Create export file
        export_filename = f"user_data_export_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        export_filepath = os.path.join(output_path, export_filename)
        
        os.makedirs(output_path, exist_ok=True)
        with open(export_filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"User data exported to {export_filepath}")
        return export_filepath


# Example usage and demo
def create_demo_gdpr_system():
    """Create a demo GDPR compliance system"""
    config = GDPRConfig(
        organization_name="Stock Analysis System Ltd",
        dpo_contact="dpo@stockanalysis.com",
        default_retention_days=2555,
        request_response_days=30,
        enable_audit_logging=True,
        enable_consent_tracking=True
    )
    
    return GDPRComplianceManager(config)


if __name__ == "__main__":
    # Demo the GDPR compliance system
    gdpr_manager = create_demo_gdpr_system()
    
    print("🛡️ Starting GDPR Compliance System Demo")
    print("=" * 50)
    
    user_id = "user_12345"
    ip_address = "192.168.1.100"
    user_agent = "Mozilla/5.0 (Demo Browser)"
    
    # Record consent
    consent = gdpr_manager.record_consent(
        user_id=user_id,
        consent_type=ConsentType.ANALYTICS,
        granted=True,
        ip_address=ip_address,
        user_agent=user_agent,
        purpose="Website analytics and improvement"
    )
    print(f"✅ Consent recorded: {consent.consent_type.value} = {consent.granted}")
    
    # Create data subject request
    request = gdpr_manager.handle_data_subject_request(
        user_id=user_id,
        request_type=RequestType.ACCESS,
        requester_email="user@example.com",
        requester_ip=ip_address,
        description="Request for all my personal data"
    )
    print(f"📋 Data subject request created: {request.request_id} ({request.request_type.value})")
    
    # Generate compliance report
    report = gdpr_manager.generate_compliance_report()
    print(f"📊 Compliance report generated:")
    print(f"   - Total consent records: {report['consent_management']['total_consent_records']}")
    print(f"   - Data subject requests: {report['data_subject_requests']['total_requests']}")
    print(f"   - Audit log entries: {report['audit_logging']['total_audit_entries']}")
    
    # Perform data cleanup (dry run)
    cleanup_results = gdpr_manager.perform_data_cleanup(dry_run=True)
    print(f"🧹 Data cleanup analysis:")
    for action, count in cleanup_results["actions_taken"].items():
        print(f"   - {action}: {count}")
    
    # Export user data
    export_path = gdpr_manager.export_user_data(user_id, "exports")
    print(f"📦 User data exported to: {export_path}")
    
    print("\n✅ GDPR Compliance System Demo completed successfully!")