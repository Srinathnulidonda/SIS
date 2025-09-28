import os
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import (
    Index, UniqueConstraint, CheckConstraint, event,
    text, func, select, and_, or_, desc
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import validates, deferred, column_property
from sqlalchemy.sql import expression
from sqlalchemy.ext.declarative import declared_attr
from werkzeug.security import generate_password_hash, check_password_hash
import redis
import requests
import feedparser
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass
from functools import lru_cache, wraps
import asyncio
import aiohttp
from cryptography.fernet import Fernet
import jwt

# Initialize extensions
db = SQLAlchemy()

# Initialize Redis with error handling
try:
    redis_client = redis.Redis.from_url(
        os.getenv('REDIS_URL', 'redis://red-d3cikjqdbo4c73e72slg:mirq8x6uekGSDV0O3eb1eVjUG3GuYkVe@red-d3cikjqdbo4c73e72slg:6379'),
        decode_responses=True
    )
    redis_client.ping()
except:
    # Fallback to mock redis client if Redis is not available
    class MockRedis:
        def get(self, key):
            return None
        def set(self, key, value):
            return True
        def setex(self, key, ttl, value):
            return True
        def incr(self, key):
            return 1
        def expire(self, key, ttl):
            return True
        def lpush(self, key, value):
            return True
        def ping(self):
            return True
    
    redis_client = MockRedis()
    logging.warning("Redis not available, using mock client")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
if isinstance(ENCRYPTION_KEY, str):
    ENCRYPTION_KEY = ENCRYPTION_KEY.encode()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Enums for better type safety
class UserRole(Enum):
    SUPER_ADMIN = 'super_admin'
    ADMIN = 'admin'
    MANAGER = 'manager'
    EDITOR = 'editor'
    VIEWER = 'viewer'

class JobStatus(Enum):
    DRAFT = 'draft'
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    ARCHIVED = 'archived'

class JobSource(Enum):
    MANUAL = 'manual'
    RSS = 'rss'
    API = 'api'
    SCRAPER = 'scraper'
    IMPORT = 'import'

class AdType(Enum):
    BANNER = 'banner'
    SIDEBAR = 'sidebar'
    POPUP = 'popup'
    NATIVE = 'native'
    VIDEO = 'video'

# Base model with common fields
class BaseModel(db.Model):
    """Abstract base model with common fields and methods"""
    __abstract__ = True
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        index=True
    )
    deleted_at = db.Column(db.DateTime(timezone=True), index=True)
    version = db.Column(db.Integer, default=1, nullable=False)
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    @hybrid_property
    def is_deleted(self):
        return self.deleted_at is not None
    
    def soft_delete(self):
        """Soft delete the record"""
        self.deleted_at = datetime.utcnow()
        db.session.commit()
    
    def restore(self):
        """Restore soft deleted record"""
        self.deleted_at = None
        db.session.commit()
    
    def to_dict(self, exclude: List[str] = None) -> Dict:
        """Convert model to dictionary"""
        exclude = exclude or []
        data = {}
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                elif isinstance(value, Decimal):
                    value = float(value)
                elif isinstance(value, Enum):
                    value = value.value
                data[column.name] = value
        return data
    
    def update_from_dict(self, data: Dict, allowed_fields: List[str] = None):
        """Update model from dictionary with field validation"""
        allowed_fields = allowed_fields or []
        for key, value in data.items():
            if hasattr(self, key) and (not allowed_fields or key in allowed_fields):
                setattr(self, key, value)
        self.version += 1
        db.session.commit()
    
    @classmethod
    def bulk_insert(cls, records: List[Dict]) -> int:
        """Bulk insert records for better performance"""
        if not records:
            return 0
        
        try:
            db.session.bulk_insert_mappings(cls, records)
            db.session.commit()
            return len(records)
        except Exception as e:
            db.session.rollback()
            logger.error(f"Bulk insert failed: {str(e)}")
            raise

# Audit log model for tracking changes
class AuditLog(BaseModel):
    """Track all database changes for compliance"""
    __tablename__ = 'audit_logs'
    
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    table_name = db.Column(db.String(100), nullable=False, index=True)
    record_id = db.Column(UUID(as_uuid=True), nullable=False, index=True)
    action = db.Column(db.String(20), nullable=False)  # CREATE, UPDATE, DELETE
    old_values = db.Column(JSONB)
    new_values = db.Column(JSONB)
    ip_address = db.Column(INET)
    user_agent = db.Column(db.String(500))
    
    __table_args__ = (
        Index('idx_audit_table_record', 'table_name', 'record_id'),
        Index('idx_audit_user_action', 'user_id', 'action'),
    )

# User model with advanced features
class User(UserMixin, BaseModel):
    """Enhanced user model with security features"""
    __tablename__ = 'users'
    
    # Basic fields
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    
    # Security fields
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    email_verified_at = db.Column(db.DateTime(timezone=True))
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    
    # Profile fields
    phone = db.Column(db.String(20))
    avatar_url = db.Column(db.String(500))
    bio = db.Column(db.Text)
    user_metadata = db.Column(JSONB, default=dict)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Session management
    last_login_at = db.Column(db.DateTime(timezone=True))
    last_login_ip = db.Column(INET)
    login_count = db.Column(db.Integer, default=0)
    failed_login_count = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime(timezone=True))
    
    # API access
    api_key = db.Column(db.String(64), unique=True, index=True)
    api_key_created_at = db.Column(db.DateTime(timezone=True))
    api_rate_limit = db.Column(db.Integer, default=1000)
    
    # Relationships
    jobs = db.relationship('Job', backref='creator', lazy='dynamic', foreign_keys='Job.created_by')
    audit_logs = db.relationship('AuditLog', backref='user', lazy='dynamic')
    sessions = db.relationship('UserSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    permissions = db.relationship('UserPermission', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    __table_args__ = (
        CheckConstraint('char_length(username) >= 3', name='username_min_length'),
        CheckConstraint('char_length(password_hash) >= 60', name='password_hash_min_length'),
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_role_active', 'role', 'is_active'),
    )
    
    def validate_email_field(self, email):
        """Validate email format - simplified without external dependency"""
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            raise ValueError(f"Invalid email address: {email}")
        return email.lower()
    
    @validates('email')
    def validate_email(self, key, email):
        """SQLAlchemy validator for email"""
        return self.validate_email_field(email)
    
    @validates('username')
    def validate_username(self, key, username):
        """Validate username"""
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not username.replace('_', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, and underscores")
        return username.lower()
    
    def set_password(self, password: str):
        """Set password with strength validation"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Check password strength
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError("Password must contain uppercase, lowercase, and numbers")
        
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    
    def check_password(self, password: str) -> bool:
        """Check password with rate limiting"""
        if self.is_locked():
            return False
        
        is_valid = check_password_hash(self.password_hash, password)
        
        if not is_valid:
            self.failed_login_count += 1
            if self.failed_login_count >= 5:
                self.lock_account(minutes=30)
            db.session.commit()
        else:
            self.failed_login_count = 0
            self.login_count += 1
            self.last_login_at = datetime.utcnow()
            db.session.commit()
        
        return is_valid
    
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False
    
    def lock_account(self, minutes: int = 30):
        """Lock account for specified minutes"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
        logger.warning(f"Account locked for user {self.username}")
    
    def generate_api_key(self) -> str:
        """Generate new API key"""
        self.api_key = hashlib.sha256(f"{self.id}{datetime.utcnow()}".encode()).hexdigest()
        self.api_key_created_at = datetime.utcnow()
        db.session.commit()
        return self.api_key
    
    def generate_auth_token(self, expires_in: int = 3600) -> str:
        """Generate JWT authentication token"""
        payload = {
            'user_id': str(self.id),
            'username': self.username,
            'role': self.role.value,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, os.getenv('JWT_SECRET_KEY', 'dev-secret'), algorithm='HS256')
    
    @staticmethod
    def verify_auth_token(token: str) -> Optional['User']:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, os.getenv('JWT_SECRET_KEY', 'dev-secret'), algorithms=['HS256'])
            return User.query.get(payload['user_id'])
        except jwt.InvalidTokenError:
            return None
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        # Super admin has all permissions
        if self.role == UserRole.SUPER_ADMIN:
            return True
        
        # Check role-based permissions
        role_permissions = {
            UserRole.ADMIN: ['manage_users', 'manage_jobs', 'view_analytics', 'manage_ads'],
            UserRole.MANAGER: ['manage_jobs', 'view_analytics', 'manage_ads'],
            UserRole.EDITOR: ['create_jobs', 'edit_jobs', 'view_analytics'],
            UserRole.VIEWER: ['view_jobs', 'view_basic_analytics']
        }
        
        if self.role in role_permissions and permission in role_permissions[self.role]:
            return True
        
        # Check specific user permissions
        return self.permissions.filter_by(permission=permission, granted=True).first() is not None
    
    def get_active_sessions(self) -> List['UserSession']:
        """Get all active sessions for user"""
        return self.sessions.filter(
            UserSession.expires_at > datetime.utcnow(),
            UserSession.is_active == True
        ).all()
    
    def revoke_all_sessions(self):
        """Revoke all user sessions"""
        self.sessions.update({UserSession.is_active: False})
        db.session.commit()

# User session management
class UserSession(BaseModel):
    """Track user sessions for security"""
    __tablename__ = 'user_sessions'
    
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    ip_address = db.Column(INET, nullable=False)
    user_agent = db.Column(db.String(500))
    location = db.Column(db.String(100))
    device_type = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    
    __table_args__ = (
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_token_active', 'session_token', 'is_active'),
    )

# User permissions for fine-grained access control
class UserPermission(BaseModel):
    """Granular permission system"""
    __tablename__ = 'user_permissions'
    
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    permission = db.Column(db.String(100), nullable=False)
    granted = db.Column(db.Boolean, default=True)
    granted_by = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    expires_at = db.Column(db.DateTime(timezone=True))
    
    __table_args__ = (
        UniqueConstraint('user_id', 'permission', name='unique_user_permission'),
        Index('idx_permission_user', 'user_id', 'permission'),
    )

# Enhanced Job model  
class Job(BaseModel):
    """Job posting model with full-text search and versioning"""
    __tablename__ = 'jobs'
    
    # Basic fields
    title = db.Column(db.String(200), nullable=False, index=True)
    slug = db.Column(db.String(250), unique=True, nullable=False, index=True)
    category = db.Column(db.String(100), index=True)
    sub_category = db.Column(db.String(100))
    description = db.Column(db.Text)
    requirements = db.Column(db.Text)
    
    # Job details
    company = db.Column(db.String(200))
    location = db.Column(db.String(200))
    salary_min = db.Column(db.Numeric(10, 2))
    salary_max = db.Column(db.Numeric(10, 2))
    salary_currency = db.Column(db.String(3), default='INR')
    job_type = db.Column(db.String(50))  # full-time, part-time, contract
    experience_required = db.Column(db.String(50))
    
    # Source and status
    source = db.Column(db.Enum(JobSource), default=JobSource.MANUAL, nullable=False)
    source_url = db.Column(db.String(500))
    status = db.Column(db.Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    
    # Dates
    posted_date = db.Column(db.DateTime(timezone=True))
    expires_at = db.Column(db.DateTime(timezone=True), index=True)
    approved_at = db.Column(db.DateTime(timezone=True))
    
    # Relationships
    created_by = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    approved_by = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    
    # SEO and metadata
    meta_title = db.Column(db.String(160))
    meta_description = db.Column(db.String(320))
    tags = db.Column(JSONB, default=list)
    job_metadata = db.Column(JSONB, default=dict)  # Renamed from metadata to avoid conflicts
    
    # Analytics
    view_count = db.Column(db.Integer, default=0)
    apply_count = db.Column(db.Integer, default=0)
    share_count = db.Column(db.Integer, default=0)
    
    # Quality score
    quality_score = db.Column(db.Float, default=0.0)
    
    __table_args__ = (
        Index('idx_job_status_expires', 'status', 'expires_at'),
        Index('idx_job_category_status', 'category', 'status'),
        CheckConstraint('salary_min <= salary_max', name='salary_range_check'),
    )
    
    @validates('slug')
    def validate_slug(self, key, slug):
        """Generate and validate slug"""
        if not slug and self.title:
            slug = self.generate_slug(self.title)
        return slug
    
    @staticmethod
    def generate_slug(title: str) -> str:
        """Generate URL-friendly slug"""
        import re
        slug = re.sub(r'[^\w\s-]', '', title.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        
        # Ensure uniqueness
        base_slug = slug
        counter = 1
        while Job.query.filter_by(slug=slug).first():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        return slug
    
    def calculate_quality_score(self) -> float:
        """Calculate job quality score based on completeness"""
        score = 0.0
        
        # Check field completeness
        if self.title: score += 10
        if self.description and len(self.description) > 100: score += 20
        if self.requirements: score += 10
        if self.company: score += 10
        if self.location: score += 10
        if self.salary_min and self.salary_max: score += 15
        if self.tags and len(self.tags) > 0: score += 10
        if self.meta_description: score += 5
        if self.expires_at and self.expires_at > datetime.utcnow(): score += 10
        
        self.quality_score = min(score, 100.0)
        return self.quality_score
    
    def update_search_vector(self):
        """Update full-text search vector - placeholder for PostgreSQL implementation"""
        pass
    
    @hybrid_property
    def is_active(self):
        """Check if job is currently active"""
        return (
            self.status == JobStatus.APPROVED and
            (self.expires_at is None or self.expires_at > datetime.utcnow()) and
            self.deleted_at is None
        )
    
    @classmethod
    def search(cls, query: str, filters: Dict = None) -> List['Job']:
        """Advanced search with filters"""
        search_query = cls.query
        
        # Simple text search (can be enhanced with PostgreSQL full-text search)
        if query:
            search_query = search_query.filter(
                or_(
                    cls.title.ilike(f'%{query}%'),
                    cls.description.ilike(f'%{query}%'),
                    cls.company.ilike(f'%{query}%')
                )
            )
        
        # Apply filters
        if filters:
            if filters.get('category'):
                search_query = search_query.filter_by(category=filters['category'])
            if filters.get('location'):
                search_query = search_query.filter(cls.location.ilike(f"%{filters['location']}%"))
            if filters.get('salary_min'):
                search_query = search_query.filter(cls.salary_max >= filters['salary_min'])
            if filters.get('job_type'):
                search_query = search_query.filter_by(job_type=filters['job_type'])
        
        # Only active jobs
        search_query = search_query.filter(
            cls.status == JobStatus.APPROVED,
            or_(cls.expires_at.is_(None), cls.expires_at > datetime.utcnow()),
            cls.deleted_at.is_(None)
        )
        
        return search_query.order_by(desc(cls.quality_score), desc(cls.created_at)).all()

# Enhanced Ads model
class Ad(BaseModel):
    """Advertisement tracking with advanced analytics"""
    __tablename__ = 'ads'
    
    ad_type = db.Column(db.Enum(AdType), nullable=False, index=True)
    campaign_id = db.Column(db.String(100), index=True)
    advertiser = db.Column(db.String(200))
    
    # Metrics
    impressions = db.Column(db.BigInteger, default=0)
    clicks = db.Column(db.BigInteger, default=0)
    conversions = db.Column(db.BigInteger, default=0)
    revenue = db.Column(db.Numeric(10, 2), default=0)
    
    # Targeting
    target_audience = db.Column(JSONB, default=dict)
    placement = db.Column(db.String(100))
    
    # Performance
    ctr = db.Column(db.Float, default=0.0)  # Click-through rate
    conversion_rate = db.Column(db.Float, default=0.0)
    ecpm = db.Column(db.Numeric(10, 2), default=0)  # Effective cost per mille
    
    # Time-based aggregation
    date = db.Column(db.Date, nullable=False, index=True)
    hour = db.Column(db.Integer)
    
    __table_args__ = (
        Index('idx_ad_date_type', 'date', 'ad_type'),
        Index('idx_ad_campaign', 'campaign_id', 'date'),
        UniqueConstraint('ad_type', 'campaign_id', 'date', 'hour', name='unique_ad_hour'),
    )
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if self.impressions > 0:
            self.ctr = (self.clicks / self.impressions) * 100
            self.ecpm = (self.revenue / self.impressions) * 1000
        
        if self.clicks > 0:
            self.conversion_rate = (self.conversions / self.clicks) * 100

# Enhanced Analytics model
class Analytics(BaseModel):
    """Comprehensive analytics tracking"""
    __tablename__ = 'analytics'
    
    # User identification
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    session_id = db.Column(db.String(100), index=True)
    user_ip = db.Column(INET, index=True)
    
    # Event tracking
    event_type = db.Column(db.String(50), nullable=False, index=True)  # pageview, click, scroll, etc.
    event_category = db.Column(db.String(50))
    event_label = db.Column(db.String(200))
    event_value = db.Column(db.Numeric(10, 2))
    
    # Page tracking
    page_url = db.Column(db.String(500), index=True)
    page_title = db.Column(db.String(200))
    referrer = db.Column(db.String(500))
    
    # Device and browser
    user_agent = db.Column(db.String(500))
    device_type = db.Column(db.String(50))  # desktop, mobile, tablet
    browser = db.Column(db.String(50))
    os = db.Column(db.String(50))
    screen_resolution = db.Column(db.String(20))
    
    # Geographic
    country = db.Column(db.String(2))
    region = db.Column(db.String(100))
    city = db.Column(db.String(100))
    timezone = db.Column(db.String(50))
    
    # Performance
    page_load_time = db.Column(db.Integer)  # milliseconds
    dom_ready_time = db.Column(db.Integer)
    
    # Custom dimensions
    custom_data = db.Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_analytics_session', 'session_id', 'created_at'),
        Index('idx_analytics_user', 'user_id', 'created_at'),
        Index('idx_analytics_event', 'event_type', 'created_at'),
        Index('idx_analytics_page', 'page_url', 'created_at'),
    )

# Job fetching service with advanced features
class JobFetcherService:
    """Enterprise-grade job fetching service"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SridharBot/1.0)'
        })
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_response(url: str, cache_time: int = 3600) -> Optional[str]:
        """Get cached response from Redis"""
        cache_key = f"fetch:{hashlib.md5(url.encode()).hexdigest()}"
        cached = redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for {url}")
            return cached
        
        return None
    
    def set_cache(self, url: str, content: str, ttl: int = 3600):
        """Set cache in Redis"""
        cache_key = f"fetch:{hashlib.md5(url.encode()).hexdigest()}"
        redis_client.setex(cache_key, ttl, content)
    
    async def fetch_rss_async(self, feed_urls: Dict[str, str]) -> List[Dict]:
        """Asynchronously fetch RSS feeds"""
        jobs = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source, url in feed_urls.items():
                tasks.append(self._fetch_single_rss(session, source, url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    jobs.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"RSS fetch error: {str(result)}")
        
        return jobs
    
    async def _fetch_single_rss(self, session: aiohttp.ClientSession, source: str, url: str) -> List[Dict]:
        """Fetch single RSS feed"""
        try:
            # Check cache first
            cached = self.get_cached_response(url)
            if cached:
                feed = feedparser.parse(cached)
            else:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    content = await response.text()
                    self.set_cache(url, content)
                    feed = feedparser.parse(content)
            
            jobs = []
            for entry in feed.entries[:20]:
                job = {
                    'title': self._clean_text(entry.get('title', '')),
                    'description': self._clean_text(entry.get('summary', '')),
                    'source_url': entry.get('link', ''),
                    'source': JobSource.RSS,
                    'category': self._detect_category(entry.get('title', '')),
                    'status': JobStatus.PENDING,
                    'posted_date': self._parse_date(entry.get('published_parsed')),
                    'expires_at': datetime.utcnow() + timedelta(days=30),
                    'tags': self._extract_tags(entry.get('title', '') + ' ' + entry.get('summary', '')),
                    'job_metadata': {'source_name': source}
                }
                jobs.append(job)
            
            logger.info(f"Fetched {len(jobs)} jobs from {source}")
            return jobs
            
        except Exception as e:
            logger.error(f"Error fetching RSS from {source}: {str(e)}")
            return []
    
    def fetch_api_jobs_batch(self, apis: List[Dict]) -> List[Dict]:
        """Fetch jobs from multiple APIs with retry logic"""
        jobs = []
        
        for api_config in apis:
            try:
                jobs.extend(self._fetch_with_retry(api_config))
            except Exception as e:
                logger.error(f"API fetch failed for {api_config['name']}: {str(e)}")
                continue
        
        return jobs
    
    def _fetch_with_retry(self, api_config: Dict, max_retries: int = 3) -> List[Dict]:
        """Fetch with exponential backoff retry"""
        import time
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    api_config['url'],
                    params=api_config.get('params', {}),
                    headers=api_config.get('headers', {}),
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                return self._parse_api_response(data, api_config['parser'])
                
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {api_config['name']} after {wait_time}s")
                time.sleep(wait_time)
                
                if attempt == max_retries - 1:
                    raise
        
        return []
    
    def _parse_api_response(self, data: Dict, parser_config: Dict) -> List[Dict]:
        """Parse API response based on configuration"""
        jobs = []
        
        # Navigate to jobs array in response
        jobs_data = data
        for key in parser_config.get('path', '').split('.'):
            if key:
                jobs_data = jobs_data.get(key, [])
        
        # Ensure jobs_data is a list
        if not isinstance(jobs_data, list):
            jobs_data = [jobs_data] if jobs_data else []
        
        for item in jobs_data[:50]:  # Limit to 50 jobs per API
            job = {
                'title': self._get_nested_value(item, parser_config.get('title')),
                'description': self._get_nested_value(item, parser_config.get('description')),
                'company': self._get_nested_value(item, parser_config.get('company')),
                'location': self._get_nested_value(item, parser_config.get('location')),
                'source_url': self._get_nested_value(item, parser_config.get('url')),
                'source': JobSource.API,
                'status': JobStatus.PENDING,
                'expires_at': datetime.utcnow() + timedelta(days=30)
            }
            
            jobs.append(job)
        
        return jobs
    
    def scrape_websites_intelligent(self, configs: List[Dict]) -> List[Dict]:
        """Intelligent web scraping with anti-detection"""
        jobs = []
        
        for config in configs:
            try:
                # Add random delay to avoid detection
                import random
                import time
                time.sleep(random.uniform(1, 3))
                
                # Rotate user agents
                self.session.headers['User-Agent'] = self._get_random_user_agent()
                
                response = self.session.get(config['url'], timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'lxml')
                jobs.extend(self._parse_scraped_content(soup, config))
                
            except Exception as e:
                logger.error(f"Scraping failed for {config['name']}: {str(e)}")
                continue
        
        return jobs
    
    def _parse_scraped_content(self, soup: BeautifulSoup, config: Dict) -> List[Dict]:
        """Parse scraped content based on selectors"""
        jobs = []
        
        job_elements = soup.select(config['job_selector'])[:20]
        
        for element in job_elements:
            try:
                job = {
                    'title': self._safe_extract(element, config.get('title_selector')),
                    'description': self._safe_extract(element, config.get('description_selector')),
                    'source_url': self._safe_extract_url(element, config.get('url_selector'), config['base_url']),
                    'source': JobSource.SCRAPER,
                    'status': JobStatus.PENDING,
                    'expires_at': datetime.utcnow() + timedelta(days=30),
                    'job_metadata': {'source_name': config['name']}
                }
                
                if job['title']:  # Only add if title exists
                    jobs.append(job)
                    
            except Exception as e:
                logger.warning(f"Failed to parse job element: {str(e)}")
                continue
        
        return jobs
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        import re
        import html
        
        if not text:
            return ''
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?;:()```math```{}\'"]', '', text)
        
        return text.strip()
    
    @staticmethod
    def _detect_category(title: str) -> str:
        """Detect job category from title using keywords"""
        categories = {
            'Government': ['govt', 'government', 'psc', 'upsc', 'ssc', 'railway', 'defense'],
            'IT': ['software', 'developer', 'programmer', 'engineer', 'tech', 'data', 'cloud'],
            'Banking': ['bank', 'finance', 'accountant', 'clerk', 'po', 'cashier'],
            'Teaching': ['teacher', 'professor', 'lecturer', 'education', 'trainer'],
            'Healthcare': ['doctor', 'nurse', 'medical', 'hospital', 'pharma', 'health'],
            'Sales': ['sales', 'marketing', 'business', 'executive', 'manager'],
        }
        
        title_lower = title.lower()
        
        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        return 'General'
    
    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        """Extract relevant tags from text"""
        import re
        from collections import Counter
        
        # Common words to exclude
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'to', 'for', 'of', 'in', 'with'}
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter and count
        word_counts = Counter(word for word in words if len(word) > 3 and word not in stop_words)
        
        # Return top 10 most common as tags
        return [word for word, _ in word_counts.most_common(10)]
    
    @staticmethod
    def _parse_date(date_tuple) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_tuple:
            try:
                import time
                return datetime.fromtimestamp(time.mktime(date_tuple))
            except:
                pass
        return datetime.utcnow()
    
    @staticmethod
    def _get_nested_value(data: Dict, path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        if not path:
            return None
        
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value
    
    @staticmethod
    def _get_random_user_agent() -> str:
        """Get random user agent for rotation"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        ]
        import random
        return random.choice(user_agents)
    
    @staticmethod
    def _safe_extract(element, selector: str) -> str:
        """Safely extract text from element"""
        try:
            if selector:
                found = element.select_one(selector)
                return found.get_text(strip=True) if found else ''
            return element.get_text(strip=True)
        except:
            return ''
    
    @staticmethod
    def _safe_extract_url(element, selector: str, base_url: str) -> str:
        """Safely extract URL from element"""
        try:
            if selector:
                found = element.select_one(selector)
                if found:
                    url = found.get('href', '')
                    if url and not url.startswith('http'):
                        from urllib.parse import urljoin
                        url = urljoin(base_url, url)
                    return url
            return ''
        except:
            return ''

# Analytics Service
class AnalyticsService:
    """Advanced analytics service with real-time processing"""
    
    @staticmethod
    def track_event(request, event_type: str, **kwargs):
        """Track analytics event with enriched data"""
        try:
            from flask_login import current_user
            
            analytics = Analytics(
                user_id=current_user.id if current_user and current_user.is_authenticated else None,
                session_id=request.cookies.get('session_id', str(uuid.uuid4())),
                user_ip=request.remote_addr,
                event_type=event_type,
                event_category=kwargs.get('category'),
                event_label=kwargs.get('label'),
                event_value=kwargs.get('value'),
                page_url=request.url,
                page_title=kwargs.get('page_title'),
                referrer=request.referrer,
                user_agent=request.user_agent.string[:500] if request.user_agent else None,
                custom_data=kwargs.get('custom_data', {})
            )
            
            db.session.add(analytics)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Analytics tracking error: {str(e)}")
    
    @staticmethod
    def get_dashboard_metrics(date_from: datetime, date_to: datetime) -> Dict:
        """Get comprehensive dashboard metrics"""
        try:
            metrics = {
                'overview': {
                    'total_users': db.session.query(func.count(func.distinct(Analytics.user_id))).filter(
                        Analytics.created_at.between(date_from, date_to)
                    ).scalar() or 0,
                    
                    'total_sessions': db.session.query(func.count(func.distinct(Analytics.session_id))).filter(
                        Analytics.created_at.between(date_from, date_to)
                    ).scalar() or 0,
                    
                    'total_pageviews': db.session.query(func.count(Analytics.id)).filter(
                        Analytics.event_type == 'pageview',
                        Analytics.created_at.between(date_from, date_to)
                    ).scalar() or 0
                },
                'traffic': {
                    'sources': [],
                    'pages': []
                }
            }
            
            # Get top pages
            top_pages = db.session.query(
                Analytics.page_url,
                func.count(Analytics.id).label('views')
            ).filter(
                Analytics.event_type == 'pageview',
                Analytics.created_at.between(date_from, date_to)
            ).group_by(Analytics.page_url).order_by(desc('views')).limit(10).all()
            
            metrics['traffic']['pages'] = [
                {'url': page[0], 'views': page[1]} for page in top_pages
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {str(e)}")
            return {
                'overview': {'total_users': 0, 'total_sessions': 0, 'total_pageviews': 0},
                'traffic': {'sources': [], 'pages': []}
            }

# Database maintenance and optimization
class DatabaseMaintenance:
    """Database maintenance utilities"""
    
    @staticmethod
    def backup_database() -> Optional[str]:
        """Create database backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"backup_{timestamp}.sql"
            logger.info(f"Database backup initiated: {backup_file}")
            # Actual backup implementation would go here
            return backup_file
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return None
    
    @staticmethod
    def clean_old_data(days: int = 90):
        """Clean old data from tables"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean old analytics
            deleted = Analytics.query.filter(
                Analytics.created_at < cutoff_date
            ).delete()
            
            # Clean expired sessions
            UserSession.query.filter(
                UserSession.expires_at < datetime.utcnow()
            ).delete()
            
            # Archive old jobs
            Job.query.filter(
                Job.expires_at < cutoff_date,
                Job.status != JobStatus.ARCHIVED
            ).update({Job.status: JobStatus.ARCHIVED})
            
            db.session.commit()
            logger.info(f"Cleaned {deleted} old records")
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Data cleanup failed: {str(e)}")

# Event listeners for audit logging (optional - can be enabled when needed)
def enable_audit_logging():
    """Enable audit logging for all models"""
    @event.listens_for(db.session, 'before_commit')
    def receive_before_commit(session):
        """Log all database changes before commit"""
        for obj in session.new:
            if not isinstance(obj, AuditLog):
                _create_audit_log(obj, 'CREATE')
        
        for obj in session.dirty:
            if not isinstance(obj, AuditLog) and session.is_modified(obj):
                _create_audit_log(obj, 'UPDATE')
        
        for obj in session.deleted:
            if not isinstance(obj, AuditLog):
                _create_audit_log(obj, 'DELETE')

def _create_audit_log(obj, action):
    """Create audit log entry"""
    try:
        from flask import has_request_context, request
        from flask_login import current_user
        
        audit = AuditLog(
            table_name=obj.__tablename__,
            record_id=obj.id if hasattr(obj, 'id') else None,
            action=action,
            new_values=obj.to_dict() if hasattr(obj, 'to_dict') else None
        )
        
        if has_request_context():
            audit.user_id = current_user.id if current_user.is_authenticated else None
            audit.ip_address = request.remote_addr
            audit.user_agent = request.user_agent.string[:500]
        
        db.session.add(audit)
        
    except Exception as e:
        logger.warning(f"Failed to create audit log: {str(e)}")