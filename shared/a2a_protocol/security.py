"""
Security and Authentication for A2A Protocol

Implements enterprise-grade authentication, authorization, and encryption
as specified in the A2A protocol with HIPAA compliance.
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from pydantic import BaseModel, Field
import base64


class AccessLevel(str, Enum):
    """Access levels for different types of data"""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    TOP_SECRET = "top_secret"


class PermissionType(str, Enum):
    """Types of permissions that can be granted"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


class AgentRole(str, Enum):
    """Roles that agents can have in the system"""
    PRIMARY_SCREENING = "primary_screening"
    CRISIS_DETECTION = "crisis_detection"
    THERAPEUTIC_INTERVENTION = "therapeutic_intervention"
    CARE_COORDINATION = "care_coordination"
    PROGRESS_ANALYTICS = "progress_analytics"
    SYSTEM_ADMIN = "system_admin"
    CLINICAL_OVERSIGHT = "clinical_oversight"


class SecurityToken(BaseModel):
    """Security token for agent authentication"""
    token_id: str
    agent_id: str
    role: AgentRole
    permissions: List[PermissionType]
    access_level: AccessLevel
    issued_at: datetime
    expires_at: datetime
    scope: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class EncryptionKey(BaseModel):
    """Encryption key for data protection"""
    key_id: str
    algorithm: str
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_level: AccessLevel


class A2ASecurity:
    """
    Handles security, authentication, and encryption for A2A protocol
    Implements HIPAA-compliant security measures
    """
    
    def __init__(self, secret_key: str, encryption_key: Optional[bytes] = None):
        self.secret_key = secret_key
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.active_tokens: Dict[str, SecurityToken] = {}
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.role_permissions: Dict[AgentRole, Set[PermissionType]] = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[AgentRole, Set[PermissionType]]:
        """Initialize default permissions for each agent role"""
        return {
            AgentRole.PRIMARY_SCREENING: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE
            },
            AgentRole.CRISIS_DETECTION: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE, PermissionType.EXECUTE
            },
            AgentRole.THERAPEUTIC_INTERVENTION: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE
            },
            AgentRole.CARE_COORDINATION: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE, PermissionType.EXECUTE
            },
            AgentRole.PROGRESS_ANALYTICS: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE
            },
            AgentRole.SYSTEM_ADMIN: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, 
                PermissionType.DELETE, PermissionType.SHARE, PermissionType.ADMIN
            },
            AgentRole.CLINICAL_OVERSIGHT: {
                PermissionType.READ, PermissionType.WRITE, PermissionType.SHARE, PermissionType.EXECUTE
            }
        }
    
    def generate_agent_token(
        self,
        agent_id: str,
        role: AgentRole,
        access_level: AccessLevel = AccessLevel.RESTRICTED,
        scope: Optional[List[str]] = None,
        expires_in_hours: int = 24
    ) -> SecurityToken:
        """
        Generate a security token for an agent
        
        Args:
            agent_id: ID of the agent
            role: Role of the agent
            access_level: Access level for the token
            scope: Optional scope restrictions
            expires_in_hours: Token expiration time in hours
            
        Returns:
            SecurityToken for the agent
        """
        token_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=expires_in_hours)
        
        permissions = self.role_permissions.get(role, set())
        
        token = SecurityToken(
            token_id=token_id,
            agent_id=agent_id,
            role=role,
            permissions=list(permissions),
            access_level=access_level,
            issued_at=now,
            expires_at=expires_at,
            scope=scope or []
        )
        
        self.active_tokens[token_id] = token
        return token
    
    def validate_token(self, token_id: str) -> Optional[SecurityToken]:
        """
        Validate a security token
        
        Args:
            token_id: ID of the token to validate
            
        Returns:
            SecurityToken if valid, None otherwise
        """
        token = self.active_tokens.get(token_id)
        
        if not token:
            return None
        
        if datetime.utcnow() > token.expires_at:
            # Token expired, remove it
            del self.active_tokens[token_id]
            return None
        
        return token
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke a security token
        
        Args:
            token_id: ID of the token to revoke
            
        Returns:
            bool: True if token was revoked successfully
        """
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
            return True
        return False
    
    def check_permission(
        self,
        token: SecurityToken,
        required_permission: PermissionType,
        required_access_level: AccessLevel
    ) -> bool:
        """
        Check if a token has the required permission and access level
        
        Args:
            token: Security token to check
            required_permission: Required permission type
            required_access_level: Required access level
            
        Returns:
            bool: True if permission is granted
        """
        # Check if token has expired
        if datetime.utcnow() > token.expires_at:
            return False
        
        # Check permission
        if required_permission not in token.permissions:
            return False
        
        # Check access level (higher number = more restrictive)
        access_levels = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.RESTRICTED: 2,
            AccessLevel.CONFIDENTIAL: 3,
            AccessLevel.TOP_SECRET: 4
        }
        
        token_level = access_levels.get(token.access_level, 0)
        required_level = access_levels.get(required_access_level, 0)
        
        return token_level >= required_level
    
    def encrypt_data(self, data: str, access_level: AccessLevel = AccessLevel.RESTRICTED) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data: Data to encrypt
            access_level: Access level for the data
            
        Returns:
            Encrypted data as base64 string
        """
        # Use different encryption keys based on access level
        key_id = f"key_{access_level.value}"
        
        if key_id not in self.encryption_keys:
            # Generate new key for this access level
            key_data = Fernet.generate_key()
            self.encryption_keys[key_id] = EncryptionKey(
                key_id=key_id,
                algorithm="fernet",
                key_data=key_data,
                created_at=datetime.utcnow(),
                access_level=access_level
            )
        
        encryption_key = self.encryption_keys[key_id]
        fernet = Fernet(encryption_key.key_data)
        
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str, access_level: AccessLevel = AccessLevel.RESTRICTED) -> Optional[str]:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Encrypted data as base64 string
            access_level: Access level of the data
            
        Returns:
            Decrypted data if successful, None otherwise
        """
        try:
            key_id = f"key_{access_level.value}"
            
            if key_id not in self.encryption_keys:
                return None
            
            encryption_key = self.encryption_keys[key_id]
            fernet = Fernet(encryption_key.key_data)
            
            decoded_data = base64.b64decode(encrypted_data)
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        
        except Exception as e:
            print(f"Decryption failed: {e}")
            return None
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash sensitive data for storage
        
        Args:
            data: Data to hash
            salt: Optional salt for hashing
            
        Returns:
            Tuple of (hashed_data, salt_used)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(data.encode()))
        hashed_data = hashlib.sha256(key + data.encode()).hexdigest()
        
        return hashed_data, salt
    
    def verify_hashed_data(self, data: str, hashed_data: str, salt: str) -> bool:
        """
        Verify hashed data
        
        Args:
            data: Original data
            hashed_data: Stored hash
            salt: Salt used for hashing
            
        Returns:
            bool: True if data matches hash
        """
        computed_hash, _ = self.hash_sensitive_data(data, salt)
        return computed_hash == hashed_data
    
    def create_audit_log(
        self,
        agent_id: str,
        action: str,
        resource: str,
        access_level: AccessLevel,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an audit log entry for compliance
        
        Args:
            agent_id: ID of the agent performing the action
            action: Action performed
            resource: Resource accessed
            access_level: Access level required
            success: Whether the action was successful
            metadata: Additional metadata
            
        Returns:
            Audit log entry
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "action": action,
            "resource": resource,
            "access_level": access_level.value,
            "success": success,
            "metadata": metadata or {},
            "session_id": secrets.token_urlsafe(16)
        }


class AuthenticationManager:
    """
    Manages authentication and authorization for the A2A system
    """
    
    def __init__(self, security: A2ASecurity):
        self.security = security
        self.agent_credentials: Dict[str, str] = {}  # agent_id -> hashed_password
    
    async def register_agent(
        self,
        agent_id: str,
        password: str,
        role: AgentRole,
        access_level: AccessLevel = AccessLevel.RESTRICTED
    ) -> bool:
        """
        Register a new agent with authentication credentials
        
        Args:
            agent_id: ID of the agent
            password: Password for the agent
            role: Role of the agent
            access_level: Access level for the agent
            
        Returns:
            bool: True if registration was successful
        """
        try:
            # Hash the password
            hashed_password, salt = self.security.hash_sensitive_data(password)
            
            # Store credentials (in production, this would be in a secure database)
            self.agent_credentials[agent_id] = f"{hashed_password}:{salt}"
            
            # Generate initial token
            token = self.security.generate_agent_token(agent_id, role, access_level)
            
            return True
        
        except Exception as e:
            print(f"Failed to register agent: {e}")
            return False
    
    async def authenticate_agent(
        self,
        agent_id: str,
        password: str
    ) -> Optional[SecurityToken]:
        """
        Authenticate an agent and return a security token
        
        Args:
            agent_id: ID of the agent
            password: Password for the agent
            
        Returns:
            SecurityToken if authentication successful, None otherwise
        """
        if agent_id not in self.agent_credentials:
            return None
        
        try:
            stored_credentials = self.agent_credentials[agent_id]
            hashed_password, salt = stored_credentials.split(":")
            
            if not self.security.verify_hashed_data(password, hashed_password, salt):
                return None
            
            # Generate new token for authenticated agent
            # In production, you'd look up the agent's role and access level
            token = self.security.generate_agent_token(
                agent_id=agent_id,
                role=AgentRole.PRIMARY_SCREENING,  # Default role
                access_level=AccessLevel.RESTRICTED
            )
            
            return token
        
        except Exception as e:
            print(f"Authentication failed: {e}")
            return None
