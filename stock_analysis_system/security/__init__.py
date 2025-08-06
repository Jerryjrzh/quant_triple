"""
Security and Compliance Module for Stock Analysis System

This module provides comprehensive security and compliance functionality including:
- JWT-based authentication with refresh tokens
- OAuth2/OIDC integration for third-party authentication
- Role-based access control (RBAC)
- User session management and security monitoring
- GDPR compliance system
- Regulatory reporting engine

Author: Stock Analysis System Team
Date: 2024-01-20
"""

from .authentication import AuthenticationManager, JWTManager, OAuth2Manager
from .authorization import RBACManager, PermissionManager
from .session import SessionManager, SecurityMonitor
from .gdpr_compliance import GDPRComplianceManager
from .regulatory_reporting import RegulatoryReportingEngine

__all__ = [
    'AuthenticationManager',
    'JWTManager', 
    'OAuth2Manager',
    'RBACManager',
    'PermissionManager',
    'SessionManager',
    'SecurityMonitor',
    'GDPRComplianceManager',
    'RegulatoryReportingEngine'
]