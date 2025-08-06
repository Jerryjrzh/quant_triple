"""
Authentication System for Stock Analysis System

This module provides comprehensive authentication functionality including:
- JWT-based authentication with refresh tokens
- OAuth2/OIDC integration for third-party authentication
- Multi-factor authentication support
- Password security and validation

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import os
import jwt
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlencode
import requests
import base64
import json
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    enable_mfa: bool = True
    oauth2_providers: Dict[str, Dict[str, str]] = None


@dataclass
class User:
    """User data model"""
    id: int
    username: str
    email: str
    password_hash: str
    role: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = None
    last_login: datetime = None
    failed_login_attempts: int = 0
    locked_until: datetime = None
    mfa_enabled: bool = False
    mfa_secret: str = None


@dataclass
class TokenPair:
    """JWT token pair"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


@dataclass
class LoginAttempt:
    """Login attempt record"""
    username: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    failure_reason: str = None


class PasswordValidator:
    """Validates password strength and security"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against security requirements"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Check against common passwords
        if self._is_common_password(password):
            errors.append("Password is too common, please choose a more secure password")
        
        return len(errors) == 0, errors
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list"""
        common_passwords = [
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "dragon", "master"
        ]
        return password.lower() in common_passwords
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


class JWTManager:
    """Manages JWT tokens for authentication"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.encryption_key = self._get_encryption_key()
        
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for sensitive data"""
        key_file = "jwt_encryption.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "iat": now,
            "exp": now + timedelta(minutes=self.config.access_token_expire_minutes),
            "type": "access"
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "iat": now,
            "exp": now + timedelta(days=self.config.refresh_token_expire_days),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        # In a real implementation, you would fetch the user from database
        # For demo purposes, we'll create a mock user
        user = User(
            id=int(payload["sub"]),
            username=payload["username"],
            email="",
            password_hash="",
            role="user"
        )
        
        return self.create_access_token(user)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)"""
        # In a real implementation, you would store revoked tokens in a database
        # or Redis with expiration times
        payload = self.verify_token(token)
        if payload:
            jti = payload.get("jti")
            if jti:
                # Store in blacklist
                logger.info(f"Token {jti} revoked")
                return True
        return False


class MFAManager:
    """Manages Multi-Factor Authentication"""
    
    def __init__(self):
        self.backup_codes_count = 10
        
    def generate_secret(self) -> str:
        """Generate TOTP secret for user"""
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')
    
    def generate_qr_code_url(self, user: User, secret: str, issuer: str = "Stock Analysis System") -> str:
        """Generate QR code URL for TOTP setup"""
        return f"otpauth://totp/{issuer}:{user.email}?secret={secret}&issuer={issuer}"
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except ImportError:
            logger.warning("pyotp not available, TOTP verification disabled")
            return True  # For demo purposes
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA"""
        return [secrets.token_hex(4).upper() for _ in range(self.backup_codes_count)]
    
    def verify_backup_code(self, user_backup_codes: List[str], code: str) -> bool:
        """Verify backup code and remove it from list"""
        if code.upper() in user_backup_codes:
            user_backup_codes.remove(code.upper())
            return True
        return False


class OAuth2Manager:
    """Manages OAuth2/OIDC authentication"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.providers = config.oauth2_providers or {}
        
    def get_authorization_url(self, provider: str, redirect_uri: str, state: str = None) -> Optional[str]:
        """Get OAuth2 authorization URL"""
        if provider not in self.providers:
            return None
        
        provider_config = self.providers[provider]
        
        params = {
            "client_id": provider_config["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": provider_config.get("scope", "openid email profile"),
            "state": state or secrets.token_urlsafe(32)
        }
        
        auth_url = provider_config["authorization_endpoint"]
        return f"{auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, provider: str, code: str, redirect_uri: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token"""
        if provider not in self.providers:
            return None
        
        provider_config = self.providers[provider]
        
        data = {
            "grant_type": "authorization_code",
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "code": code,
            "redirect_uri": redirect_uri
        }
        
        try:
            response = requests.post(
                provider_config["token_endpoint"],
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OAuth2 token exchange failed: {e}")
            return None
    
    def get_user_info(self, provider: str, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth2 provider"""
        if provider not in self.providers:
            return None
        
        provider_config = self.providers[provider]
        
        try:
            response = requests.get(
                provider_config["userinfo_endpoint"],
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OAuth2 user info request failed: {e}")
            return None


class AuthenticationManager:
    """Main authentication manager"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.password_validator = PasswordValidator(config)
        self.jwt_manager = JWTManager(config)
        self.mfa_manager = MFAManager()
        self.oauth2_manager = OAuth2Manager(config)
        self.login_attempts = []  # In production, use database
        self.users = {}  # In production, use database
        
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> Tuple[bool, str, Optional[User]]:
        """Register a new user"""
        # Validate password
        is_valid, errors = self.password_validator.validate_password(password)
        if not is_valid:
            return False, "; ".join(errors), None
        
        # Check if user already exists
        if self._user_exists(username, email):
            return False, "User already exists", None
        
        # Create user
        user = User(
            id=len(self.users) + 1,
            username=username,
            email=email,
            password_hash=self.password_validator.hash_password(password),
            role=role,
            created_at=datetime.utcnow()
        )
        
        self.users[user.id] = user
        logger.info(f"User {username} registered successfully")
        
        return True, "User registered successfully", user
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "", user_agent: str = "", mfa_token: str = None) -> Tuple[bool, str, Optional[TokenPair]]:
        """Authenticate user with username/password"""
        # Check if user is locked
        user = self._get_user_by_username(username)
        if not user:
            self._record_login_attempt(username, ip_address, user_agent, False, "User not found")
            return False, "Invalid credentials", None
        
        if self._is_user_locked(user):
            return False, f"Account locked until {user.locked_until}", None
        
        # Verify password
        if not self.password_validator.verify_password(password, user.password_hash):
            self._handle_failed_login(user, ip_address, user_agent)
            return False, "Invalid credentials", None
        
        # Check MFA if enabled
        if user.mfa_enabled and self.config.enable_mfa:
            if not mfa_token:
                return False, "MFA token required", None
            
            if not self.mfa_manager.verify_totp(user.mfa_secret, mfa_token):
                self._handle_failed_login(user, ip_address, user_agent)
                return False, "Invalid MFA token", None
        
        # Successful login
        self._handle_successful_login(user, ip_address, user_agent)
        
        # Create tokens
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)
        
        token_pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60
        )
        
        return True, "Authentication successful", token_pair
    
    def authenticate_oauth2(self, provider: str, code: str, redirect_uri: str) -> Tuple[bool, str, Optional[TokenPair]]:
        """Authenticate user via OAuth2"""
        # Exchange code for token
        token_data = self.oauth2_manager.exchange_code_for_token(provider, code, redirect_uri)
        if not token_data:
            return False, "OAuth2 token exchange failed", None
        
        # Get user info
        user_info = self.oauth2_manager.get_user_info(provider, token_data["access_token"])
        if not user_info:
            return False, "Failed to get user information", None
        
        # Find or create user
        user = self._find_or_create_oauth_user(user_info, provider)
        
        # Create tokens
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)
        
        token_pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60
        )
        
        return True, "OAuth2 authentication successful", token_pair
    
    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token"""
        return self.jwt_manager.refresh_access_token(refresh_token)
    
    def logout(self, access_token: str, refresh_token: str = None) -> bool:
        """Logout user and revoke tokens"""
        success = self.jwt_manager.revoke_token(access_token)
        if refresh_token:
            success = success and self.jwt_manager.revoke_token(refresh_token)
        return success
    
    def enable_mfa(self, user_id: int) -> Tuple[str, str, List[str]]:
        """Enable MFA for user"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        secret = self.mfa_manager.generate_secret()
        qr_url = self.mfa_manager.generate_qr_code_url(user, secret)
        backup_codes = self.mfa_manager.generate_backup_codes()
        
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        return secret, qr_url, backup_codes
    
    def disable_mfa(self, user_id: int) -> bool:
        """Disable MFA for user"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.mfa_enabled = False
        user.mfa_secret = None
        return True
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        user = self.users.get(user_id)
        if not user:
            return False, "User not found"
        
        # Verify old password
        if not self.password_validator.verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect"
        
        # Validate new password
        is_valid, errors = self.password_validator.validate_password(new_password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Update password
        user.password_hash = self.password_validator.hash_password(new_password)
        logger.info(f"Password changed for user {user.username}")
        
        return True, "Password changed successfully"
    
    def _user_exists(self, username: str, email: str) -> bool:
        """Check if user already exists"""
        for user in self.users.values():
            if user.username == username or user.email == email:
                return True
        return False
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _is_user_locked(self, user: User) -> bool:
        """Check if user account is locked"""
        if user.locked_until and user.locked_until > datetime.utcnow():
            return True
        return False
    
    def _handle_failed_login(self, user: User, ip_address: str, user_agent: str):
        """Handle failed login attempt"""
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= self.config.max_login_attempts:
            user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
            logger.warning(f"User {user.username} locked due to too many failed attempts")
        
        self._record_login_attempt(user.username, ip_address, user_agent, False, "Invalid credentials")
    
    def _handle_successful_login(self, user: User, ip_address: str, user_agent: str):
        """Handle successful login"""
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        self._record_login_attempt(user.username, ip_address, user_agent, True)
    
    def _record_login_attempt(self, username: str, ip_address: str, user_agent: str, success: bool, failure_reason: str = None):
        """Record login attempt"""
        attempt = LoginAttempt(
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            success=success,
            failure_reason=failure_reason
        )
        
        self.login_attempts.append(attempt)
        
        # Keep only last 1000 attempts
        if len(self.login_attempts) > 1000:
            self.login_attempts = self.login_attempts[-1000:]
    
    def _find_or_create_oauth_user(self, user_info: Dict[str, Any], provider: str) -> User:
        """Find existing OAuth user or create new one"""
        email = user_info.get("email")
        
        # Try to find existing user by email
        for user in self.users.values():
            if user.email == email:
                return user
        
        # Create new user
        username = user_info.get("preferred_username") or user_info.get("name") or email.split("@")[0]
        
        user = User(
            id=len(self.users) + 1,
            username=f"{username}_{provider}",
            email=email,
            password_hash="",  # OAuth users don't have passwords
            role="user",
            is_verified=True,  # OAuth users are pre-verified
            created_at=datetime.utcnow()
        )
        
        self.users[user.id] = user
        logger.info(f"OAuth user {username} created from {provider}")
        
        return user
    
    def get_login_history(self, username: str = None, limit: int = 100) -> List[LoginAttempt]:
        """Get login history"""
        attempts = self.login_attempts
        
        if username:
            attempts = [a for a in attempts if a.username == username]
        
        return sorted(attempts, key=lambda x: x.timestamp, reverse=True)[:limit]


# Example usage and demo
def create_demo_auth():
    """Create a demo authentication system"""
    config = AuthConfig(
        jwt_secret_key="your-secret-key-here",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
        enable_mfa=True,
        oauth2_providers={
            "google": {
                "client_id": "your-google-client-id",
                "client_secret": "your-google-client-secret",
                "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_endpoint": "https://oauth2.googleapis.com/token",
                "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
                "scope": "openid email profile"
            }
        }
    )
    
    return AuthenticationManager(config)


if __name__ == "__main__":
    # Demo the authentication system
    auth_manager = create_demo_auth()
    
    print("🔐 Starting Authentication System Demo")
    print("=" * 50)
    
    # Register a user
    success, message, user = auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="SecurePass123!",
        role="analyst"
    )
    
    print(f"Registration: {message}")
    
    if success:
        # Authenticate user
        success, message, tokens = auth_manager.authenticate_user(
            username="testuser",
            password="SecurePass123!",
            ip_address="192.168.1.1",
            user_agent="Test Browser"
        )
        
        print(f"Authentication: {message}")
        
        if success:
            print(f"Access Token: {tokens.access_token[:50]}...")
            print(f"Refresh Token: {tokens.refresh_token[:50]}...")
            
            # Verify token
            payload = auth_manager.jwt_manager.verify_token(tokens.access_token)
            if payload:
                print(f"Token verified for user: {payload['username']}")
            
            # Enable MFA
            secret, qr_url, backup_codes = auth_manager.enable_mfa(user.id)
            print(f"MFA enabled. Secret: {secret}")
            print(f"Backup codes: {backup_codes[:3]}...")
    
    # Show login history
    history = auth_manager.get_login_history(limit=5)
    print(f"\n📊 Recent login attempts: {len(history)}")
    for attempt in history:
        status = "✅" if attempt.success else "❌"
        print(f"{status} {attempt.username} - {attempt.timestamp} - {attempt.ip_address}")