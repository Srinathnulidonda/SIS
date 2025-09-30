import os
import secrets
import requests
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_migrate import Migrate
from sqlalchemy import func, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
import cloudinary
import cloudinary.uploader
import cloudinary.api
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import jwt
import uuid
from werkzeug.utils import secure_filename
from enum import Enum


app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/sridhar_services'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
app.config['JWT_EXPIRATION_HOURS'] = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv('REDIS_URL', 'redis://localhost:6379')
)
CORS(app, supports_credentials=True, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))


class UserRole(str, Enum):
    SUPER_ADMIN = 'super_admin'
    ADMIN = 'admin'


class JobStatus(str, Enum):
    DRAFT = 'draft'
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


class JobType(str, Enum):
    GOVERNMENT = 'government'
    PRIVATE = 'private'
    REMOTE = 'remote'


class ServiceStatus(str, Enum):
    ACTIVE = 'active'
    INACTIVE = 'inactive'


class AdStatus(str, Enum):
    ACTIVE = 'active'
    PAUSED = 'paused'
    ARCHIVED = 'archived'


class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole), nullable=False, default=UserRole.ADMIN)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_login = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    
    jobs = db.relationship('Job', back_populates='created_by', foreign_keys='Job.created_by_id')
    services = db.relationship('Service', back_populates='created_by', foreign_keys='Service.created_by_id')
    audit_logs = db.relationship('AuditLog', back_populates='user')

    def set_password(self, password: str):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self, include_sensitive=False):
        data = {
            'id': str(self.id),
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role.value,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        return data


class JobSource(db.Model):
    __tablename__ = 'job_sources'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String(255), nullable=False, unique=True)
    source_type = db.Column(db.String(50), nullable=False)
    api_url = db.Column(db.Text)
    api_key = db.Column(db.String(255))
    config = db.Column(JSONB, default={})
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_sync = db.Column(db.DateTime(timezone=True))
    sync_frequency_minutes = db.Column(db.Integer, default=60)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    jobs = db.relationship('Job', back_populates='source')

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'source_type': self.source_type,
            'api_url': self.api_url,
            'is_active': self.is_active,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'sync_frequency_minutes': self.sync_frequency_minutes,
            'created_at': self.created_at.isoformat(),
        }


class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=False)
    company = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255))
    job_type = db.Column(db.Enum(JobType), nullable=False)
    status = db.Column(db.Enum(JobStatus), nullable=False, default=JobStatus.PENDING)
    salary_min = db.Column(db.Numeric(12, 2))
    salary_max = db.Column(db.Numeric(12, 2))
    application_url = db.Column(db.Text)
    application_deadline = db.Column(db.Date)
    qualifications = db.Column(JSONB, default=[])
    tags = db.Column(JSONB, default=[])
    poster_url = db.Column(db.Text)
    poster_cloudinary_id = db.Column(db.String(255))
    source_id = db.Column(UUID(as_uuid=True), db.ForeignKey('job_sources.id'))
    external_id = db.Column(db.String(255))
    view_count = db.Column(db.Integer, default=0, nullable=False)
    apply_count = db.Column(db.Integer, default=0, nullable=False)
    is_featured = db.Column(db.Boolean, default=False, nullable=False)
    published_at = db.Column(db.DateTime(timezone=True))
    created_by_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    source = db.relationship('JobSource', back_populates='jobs')
    created_by = db.relationship('User', back_populates='jobs', foreign_keys=[created_by_id])

    __table_args__ = (
        Index('idx_job_status_type', 'status', 'job_type'),
        Index('idx_job_published', 'published_at'),
        Index('idx_job_deadline', 'application_deadline'),
    )

    def to_dict(self, include_analytics=False):
        data = {
            'id': str(self.id),
            'title': self.title,
            'description': self.description,
            'company': self.company,
            'location': self.location,
            'job_type': self.job_type.value,
            'status': self.status.value,
            'salary_min': float(self.salary_min) if self.salary_min else None,
            'salary_max': float(self.salary_max) if self.salary_max else None,
            'application_url': self.application_url,
            'application_deadline': self.application_deadline.isoformat() if self.application_deadline else None,
            'qualifications': self.qualifications,
            'tags': self.tags,
            'poster_url': self.poster_url,
            'is_featured': self.is_featured,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_analytics:
            data.update({
                'view_count': self.view_count,
                'apply_count': self.apply_count,
            })
        return data


class Service(db.Model):
    __tablename__ = 'services'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    icon_url = db.Column(db.Text)
    icon_cloudinary_id = db.Column(db.String(255))
    contact_person = db.Column(db.String(255))
    contact_phone = db.Column(db.String(20))
    contact_email = db.Column(db.String(255))
    contact_address = db.Column(db.Text)
    whatsapp_number = db.Column(db.String(20))
    pricing_info = db.Column(db.Text)
    features = db.Column(JSONB, default=[])
    status = db.Column(db.Enum(ServiceStatus), nullable=False, default=ServiceStatus.ACTIVE)
    view_count = db.Column(db.Integer, default=0, nullable=False)
    inquiry_count = db.Column(db.Integer, default=0, nullable=False)
    display_order = db.Column(db.Integer, default=0)
    created_by_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    created_by = db.relationship('User', back_populates='services', foreign_keys=[created_by_id])

    __table_args__ = (
        Index('idx_service_status_category', 'status', 'category'),
        Index('idx_service_order', 'display_order'),
    )

    def to_dict(self, include_analytics=False):
        data = {
            'id': str(self.id),
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'icon_url': self.icon_url,
            'contact_person': self.contact_person,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'contact_address': self.contact_address,
            'whatsapp_number': self.whatsapp_number,
            'pricing_info': self.pricing_info,
            'features': self.features,
            'status': self.status.value,
            'display_order': self.display_order,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_analytics:
            data.update({
                'view_count': self.view_count,
                'inquiry_count': self.inquiry_count,
            })
        return data


class Ad(db.Model):
    __tablename__ = 'ads'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String(255), nullable=False)
    ad_unit_id = db.Column(db.String(255), nullable=False)
    placement = db.Column(db.String(100), nullable=False)
    status = db.Column(db.Enum(AdStatus), nullable=False, default=AdStatus.ACTIVE)
    config = db.Column(JSONB, default={})
    impression_count = db.Column(db.BigInteger, default=0, nullable=False)
    click_count = db.Column(db.BigInteger, default=0, nullable=False)
    revenue = db.Column(db.Numeric(12, 2), default=0, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self, include_analytics=False):
        data = {
            'id': str(self.id),
            'name': self.name,
            'ad_unit_id': self.ad_unit_id,
            'placement': self.placement,
            'status': self.status.value,
            'config': self.config,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_analytics:
            data.update({
                'impression_count': self.impression_count,
                'click_count': self.click_count,
                'revenue': float(self.revenue),
            })
        return data


class AnalyticsEvent(db.Model):
    __tablename__ = 'analytics_events'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = db.Column(db.String(50), nullable=False)
    entity_type = db.Column(db.String(50))
    entity_id = db.Column(UUID(as_uuid=True))
    user_id = db.Column(UUID(as_uuid=True))
    session_id = db.Column(db.String(255))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    referrer = db.Column(db.Text)
    event_data = db.Column(JSONB, default={})
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_analytics_type_entity', 'event_type', 'entity_type', 'entity_id'),
        Index('idx_analytics_session', 'session_id'),
        Index('idx_analytics_created', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'event_type': self.event_type,
            'entity_type': self.entity_type,
            'entity_id': str(self.entity_id) if self.entity_id else None,
            'event_data': self.event_data,
            'created_at': self.created_at.isoformat(),
        }


class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    entity_type = db.Column(db.String(50), nullable=False)
    entity_id = db.Column(UUID(as_uuid=True))
    changes = db.Column(JSONB, default={})
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    
    user = db.relationship('User', back_populates='audit_logs')

    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
    )

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'user_email': self.user.email if self.user else None,
            'action': self.action,
            'entity_type': self.entity_type,
            'entity_id': str(self.entity_id) if self.entity_id else None,
            'changes': self.changes,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat(),
        }


class SiteSettings(db.Model):
    __tablename__ = 'site_settings'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = db.Column(db.String(100), nullable=False, unique=True, index=True)
    value = db.Column(JSONB, nullable=False)
    description = db.Column(db.Text)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': str(self.id),
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


def generate_token(user_id: str, role: str) -> str:
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user():
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        payload = decode_token(token)
        if payload:
            user = User.query.get(payload['user_id'])
            if user and user.is_active:
                return user
    
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user and user.is_active:
            return user
    
    return None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(user, *args, **kwargs)
    return decorated_function


def super_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        if user.role != UserRole.SUPER_ADMIN:
            return jsonify({'error': 'Super admin access required'}), 403
        return f(user, *args, **kwargs)
    return decorated_function


def log_audit(user_id: str, action: str, entity_type: str, entity_id: str = None, changes: dict = None):
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        changes=changes or {},
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )
    db.session.add(audit_log)
    db.session.commit()


def log_analytics(event_type: str, entity_type: str = None, entity_id: str = None, event_data: dict = None):
    user = get_current_user()
    event = AnalyticsEvent(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        user_id=user.id if user else None,
        session_id=session.get('session_id'),
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
        referrer=request.headers.get('Referer'),
        event_data=event_data or {}
    )
    db.session.add(event)
    db.session.commit()


@app.before_request
def before_request():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
@super_admin_required
def register(current_user):
    data = request.get_json()
    
    if not all(k in data for k in ['email', 'password', 'full_name', 'role']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    try:
        role = UserRole(data['role'])
    except ValueError:
        return jsonify({'error': 'Invalid role'}), 400
    
    user = User(
        email=data['email'],
        full_name=data['full_name'],
        role=role,
        created_by_id=current_user.id
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    log_audit(current_user.id, 'CREATE_USER', 'user', str(user.id), {'email': user.email, 'role': user.role.value})
    
    return jsonify({
        'message': 'User created successfully',
        'user': user.to_dict()
    }), 201


@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json()
    
    if not all(k in data for k in ['email', 'password']):
        return jsonify({'error': 'Missing email or password'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403
    
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    session['user_id'] = str(user.id)
    session['role'] = user.role.value
    session.permanent = True
    
    token = generate_token(str(user.id), user.role.value)
    
    log_analytics('LOGIN', 'user', str(user.id))
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout(current_user):
    log_analytics('LOGOUT', 'user', str(current_user.id))
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200


@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user_info(current_user):
    return jsonify({'user': current_user.to_dict()}), 200


@app.route('/api/users', methods=['GET'])
@super_admin_required
def list_users(current_user):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    query = User.query
    
    role = request.args.get('role')
    if role:
        query = query.filter_by(role=UserRole(role))
    
    is_active = request.args.get('is_active')
    if is_active is not None:
        query = query.filter_by(is_active=is_active.lower() == 'true')
    
    pagination = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'users': [user.to_dict() for user in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/users/<user_id>', methods=['GET'])
@super_admin_required
def get_user(current_user, user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({'user': user.to_dict()}), 200


@app.route('/api/users/<user_id>', methods=['PUT'])
@super_admin_required
def update_user(current_user, user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    changes = {}
    
    if 'full_name' in data:
        changes['full_name'] = {'old': user.full_name, 'new': data['full_name']}
        user.full_name = data['full_name']
    
    if 'email' in data and data['email'] != user.email:
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        changes['email'] = {'old': user.email, 'new': data['email']}
        user.email = data['email']
    
    if 'role' in data:
        try:
            new_role = UserRole(data['role'])
            changes['role'] = {'old': user.role.value, 'new': new_role.value}
            user.role = new_role
        except ValueError:
            return jsonify({'error': 'Invalid role'}), 400
    
    if 'is_active' in data:
        changes['is_active'] = {'old': user.is_active, 'new': data['is_active']}
        user.is_active = data['is_active']
    
    if 'password' in data:
        user.set_password(data['password'])
        changes['password'] = 'changed'
    
    db.session.commit()
    log_audit(current_user.id, 'UPDATE_USER', 'user', user_id, changes)
    
    return jsonify({
        'message': 'User updated successfully',
        'user': user.to_dict()
    }), 200


@app.route('/api/users/<user_id>', methods=['DELETE'])
@super_admin_required
def delete_user(current_user, user_id):
    if str(current_user.id) == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    db.session.delete(user)
    db.session.commit()
    
    log_audit(current_user.id, 'DELETE_USER', 'user', user_id, {'email': user.email})
    
    return jsonify({'message': 'User deleted successfully'}), 200


@app.route('/api/jobs', methods=['GET'])
@limiter.limit("100 per minute")
def list_jobs():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    query = Job.query.filter_by(status=JobStatus.APPROVED)
    
    job_type = request.args.get('job_type')
    if job_type:
        try:
            query = query.filter_by(job_type=JobType(job_type))
        except ValueError:
            pass
    
    search = request.args.get('search')
    if search:
        query = query.filter(
            db.or_(
                Job.title.ilike(f'%{search}%'),
                Job.company.ilike(f'%{search}%'),
                Job.description.ilike(f'%{search}%')
            )
        )
    
    location = request.args.get('location')
    if location:
        query = query.filter(Job.location.ilike(f'%{location}%'))
    
    is_featured = request.args.get('is_featured')
    if is_featured:
        query = query.filter_by(is_featured=is_featured.lower() == 'true')
    
    pagination = query.order_by(
        Job.is_featured.desc(),
        Job.published_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    log_analytics('LIST_JOBS', event_data={'page': page, 'search': search, 'job_type': job_type})
    
    return jsonify({
        'jobs': [job.to_dict() for job in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/jobs/<job_id>', methods=['GET'])
@limiter.limit("200 per minute")
def get_job(job_id):
    job = Job.query.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status != JobStatus.APPROVED:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Job not found'}), 404
    
    job.view_count += 1
    db.session.commit()
    
    log_analytics('VIEW_JOB', 'job', job_id)
    
    return jsonify({'job': job.to_dict(include_analytics=True)}), 200


@app.route('/api/jobs', methods=['POST'])
@login_required
@limiter.limit("30 per hour")
def create_job(current_user):
    data = request.get_json()
    
    required_fields = ['title', 'description', 'company', 'job_type']
    if not all(k in data for k in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        job_type = JobType(data['job_type'])
    except ValueError:
        return jsonify({'error': 'Invalid job type'}), 400
    
    status = JobStatus.APPROVED if current_user.role == UserRole.SUPER_ADMIN else JobStatus.PENDING
    
    job = Job(
        title=data['title'],
        description=data['description'],
        company=data['company'],
        location=data.get('location'),
        job_type=job_type,
        status=status,
        salary_min=data.get('salary_min'),
        salary_max=data.get('salary_max'),
        application_url=data.get('application_url'),
        application_deadline=datetime.fromisoformat(data['application_deadline']) if data.get('application_deadline') else None,
        qualifications=data.get('qualifications', []),
        tags=data.get('tags', []),
        poster_url=data.get('poster_url'),
        is_featured=data.get('is_featured', False) if current_user.role == UserRole.SUPER_ADMIN else False,
        created_by_id=current_user.id,
        published_at=datetime.utcnow() if status == JobStatus.APPROVED else None
    )
    
    db.session.add(job)
    db.session.commit()
    
    log_audit(current_user.id, 'CREATE_JOB', 'job', str(job.id), {'title': job.title, 'status': status.value})
    log_analytics('CREATE_JOB', 'job', str(job.id))
    
    return jsonify({
        'message': 'Job created successfully',
        'job': job.to_dict()
    }), 201


@app.route('/api/jobs/<job_id>', methods=['PUT'])
@login_required
@limiter.limit("30 per hour")
def update_job(current_user, job_id):
    job = Job.query.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.created_by_id != current_user.id and current_user.role != UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Permission denied'}), 403
    
    data = request.get_json()
    changes = {}
    
    updatable_fields = [
        'title', 'description', 'company', 'location', 'salary_min', 'salary_max',
        'application_url', 'application_deadline', 'qualifications', 'tags', 'poster_url'
    ]
    
    for field in updatable_fields:
        if field in data:
            old_value = getattr(job, field)
            new_value = data[field]
            if field == 'application_deadline' and new_value:
                new_value = datetime.fromisoformat(new_value).date()
            if old_value != new_value:
                changes[field] = {'old': str(old_value), 'new': str(new_value)}
                setattr(job, field, new_value)
    
    if 'job_type' in data:
        try:
            new_type = JobType(data['job_type'])
            if job.job_type != new_type:
                changes['job_type'] = {'old': job.job_type.value, 'new': new_type.value}
                job.job_type = new_type
        except ValueError:
            return jsonify({'error': 'Invalid job type'}), 400
    
    if current_user.role == UserRole.SUPER_ADMIN:
        if 'status' in data:
            try:
                new_status = JobStatus(data['status'])
                if job.status != new_status:
                    changes['status'] = {'old': job.status.value, 'new': new_status.value}
                    job.status = new_status
                    if new_status == JobStatus.APPROVED and not job.published_at:
                        job.published_at = datetime.utcnow()
            except ValueError:
                return jsonify({'error': 'Invalid status'}), 400
        
        if 'is_featured' in data:
            if job.is_featured != data['is_featured']:
                changes['is_featured'] = {'old': job.is_featured, 'new': data['is_featured']}
                job.is_featured = data['is_featured']
    
    db.session.commit()
    log_audit(current_user.id, 'UPDATE_JOB', 'job', job_id, changes)
    
    return jsonify({
        'message': 'Job updated successfully',
        'job': job.to_dict()
    }), 200


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
@login_required
def delete_job(current_user, job_id):
    job = Job.query.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.created_by_id != current_user.id and current_user.role != UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Permission denied'}), 403
    
    db.session.delete(job)
    db.session.commit()
    
    log_audit(current_user.id, 'DELETE_JOB', 'job', job_id, {'title': job.title})
    
    return jsonify({'message': 'Job deleted successfully'}), 200


@app.route('/api/jobs/<job_id>/apply', methods=['POST'])
@limiter.limit("50 per hour")
def track_job_apply(job_id):
    job = Job.query.get(job_id)
    if not job or job.status != JobStatus.APPROVED:
        return jsonify({'error': 'Job not found'}), 404
    
    job.apply_count += 1
    db.session.commit()
    
    log_analytics('APPLY_JOB', 'job', job_id)
    
    return jsonify({'message': 'Application tracked'}), 200


@app.route('/api/jobs/pending', methods=['GET'])
@super_admin_required
def list_pending_jobs(current_user):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = Job.query.filter_by(status=JobStatus.PENDING).order_by(
        Job.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'jobs': [job.to_dict() for job in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/jobs/<job_id>/moderate', methods=['POST'])
@super_admin_required
def moderate_job(current_user, job_id):
    job = Job.query.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    data = request.get_json()
    action = data.get('action')
    
    if action not in ['approve', 'reject']:
        return jsonify({'error': 'Invalid action'}), 400
    
    old_status = job.status
    
    if action == 'approve':
        job.status = JobStatus.APPROVED
        job.published_at = datetime.utcnow()
    else:
        job.status = JobStatus.REJECTED
    
    db.session.commit()
    
    log_audit(
        current_user.id,
        'MODERATE_JOB',
        'job',
        job_id,
        {'action': action, 'old_status': old_status.value, 'new_status': job.status.value}
    )
    
    return jsonify({
        'message': f'Job {action}d successfully',
        'job': job.to_dict()
    }), 200


@app.route('/api/services', methods=['GET'])
@limiter.limit("100 per minute")
def list_services():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    query = Service.query.filter_by(status=ServiceStatus.ACTIVE)
    
    category = request.args.get('category')
    if category:
        query = query.filter_by(category=category)
    
    search = request.args.get('search')
    if search:
        query = query.filter(
            db.or_(
                Service.title.ilike(f'%{search}%'),
                Service.description.ilike(f'%{search}%')
            )
        )
    
    pagination = query.order_by(
        Service.display_order.asc(),
        Service.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    log_analytics('LIST_SERVICES', event_data={'page': page, 'category': category})
    
    return jsonify({
        'services': [service.to_dict() for service in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/services/<service_id>', methods=['GET'])
@limiter.limit("200 per minute")
def get_service(service_id):
    service = Service.query.get(service_id)
    if not service:
        return jsonify({'error': 'Service not found'}), 404
    
    if service.status != ServiceStatus.ACTIVE:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Service not found'}), 404
    
    service.view_count += 1
    db.session.commit()
    
    log_analytics('VIEW_SERVICE', 'service', service_id)
    
    return jsonify({'service': service.to_dict(include_analytics=True)}), 200


@app.route('/api/services', methods=['POST'])
@login_required
@limiter.limit("30 per hour")
def create_service(current_user):
    data = request.get_json()
    
    required_fields = ['title', 'description', 'category']
    if not all(k in data for k in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    service = Service(
        title=data['title'],
        description=data['description'],
        category=data['category'],
        icon_url=data.get('icon_url'),
        contact_person=data.get('contact_person'),
        contact_phone=data.get('contact_phone'),
        contact_email=data.get('contact_email'),
        contact_address=data.get('contact_address'),
        whatsapp_number=data.get('whatsapp_number'),
        pricing_info=data.get('pricing_info'),
        features=data.get('features', []),
        status=ServiceStatus.ACTIVE,
        display_order=data.get('display_order', 0),
        created_by_id=current_user.id
    )
    
    db.session.add(service)
    db.session.commit()
    
    log_audit(current_user.id, 'CREATE_SERVICE', 'service', str(service.id), {'title': service.title})
    log_analytics('CREATE_SERVICE', 'service', str(service.id))
    
    return jsonify({
        'message': 'Service created successfully',
        'service': service.to_dict()
    }), 201


@app.route('/api/services/<service_id>', methods=['PUT'])
@login_required
@limiter.limit("30 per hour")
def update_service(current_user, service_id):
    service = Service.query.get(service_id)
    if not service:
        return jsonify({'error': 'Service not found'}), 404
    
    if service.created_by_id != current_user.id and current_user.role != UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Permission denied'}), 403
    
    data = request.get_json()
    changes = {}
    
    updatable_fields = [
        'title', 'description', 'category', 'icon_url', 'contact_person',
        'contact_phone', 'contact_email', 'contact_address', 'whatsapp_number',
        'pricing_info', 'features', 'display_order'
    ]
    
    for field in updatable_fields:
        if field in data:
            old_value = getattr(service, field)
            new_value = data[field]
            if old_value != new_value:
                changes[field] = {'old': str(old_value), 'new': str(new_value)}
                setattr(service, field, new_value)
    
    if 'status' in data and current_user.role == UserRole.SUPER_ADMIN:
        try:
            new_status = ServiceStatus(data['status'])
            if service.status != new_status:
                changes['status'] = {'old': service.status.value, 'new': new_status.value}
                service.status = new_status
        except ValueError:
            return jsonify({'error': 'Invalid status'}), 400
    
    db.session.commit()
    log_audit(current_user.id, 'UPDATE_SERVICE', 'service', service_id, changes)
    
    return jsonify({
        'message': 'Service updated successfully',
        'service': service.to_dict()
    }), 200


@app.route('/api/services/<service_id>', methods=['DELETE'])
@login_required
def delete_service(current_user, service_id):
    service = Service.query.get(service_id)
    if not service:
        return jsonify({'error': 'Service not found'}), 404
    
    if service.created_by_id != current_user.id and current_user.role != UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Permission denied'}), 403
    
    db.session.delete(service)
    db.session.commit()
    
    log_audit(current_user.id, 'DELETE_SERVICE', 'service', service_id, {'title': service.title})
    
    return jsonify({'message': 'Service deleted successfully'}), 200


@app.route('/api/services/<service_id>/inquiry', methods=['POST'])
@limiter.limit("50 per hour")
def track_service_inquiry(service_id):
    service = Service.query.get(service_id)
    if not service or service.status != ServiceStatus.ACTIVE:
        return jsonify({'error': 'Service not found'}), 404
    
    service.inquiry_count += 1
    db.session.commit()
    
    log_analytics('INQUIRY_SERVICE', 'service', service_id)
    
    return jsonify({'message': 'Inquiry tracked'}), 200


@app.route('/api/upload', methods=['POST'])
@login_required
@limiter.limit("20 per hour")
def upload_file(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'}), 400
    
    folder = request.form.get('folder', 'sridhar_services')
    
    try:
        result = cloudinary.uploader.upload(
            file,
            folder=folder,
            transformation=[
                {'width': 1200, 'height': 630, 'crop': 'limit'},
                {'quality': 'auto'},
                {'fetch_format': 'auto'}
            ]
        )
        
        log_audit(
            current_user.id,
            'UPLOAD_FILE',
            'file',
            result['public_id'],
            {'url': result['secure_url']}
        )
        
        return jsonify({
            'message': 'File uploaded successfully',
            'url': result['secure_url'],
            'public_id': result['public_id'],
            'width': result['width'],
            'height': result['height']
        }), 200
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/ads', methods=['GET'])
@super_admin_required
def list_ads(current_user):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    query = Ad.query
    
    status = request.args.get('status')
    if status:
        try:
            query = query.filter_by(status=AdStatus(status))
        except ValueError:
            pass
    
    pagination = query.order_by(Ad.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'ads': [ad.to_dict(include_analytics=True) for ad in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/ads/active', methods=['GET'])
@limiter.limit("100 per minute")
def list_active_ads():
    placement = request.args.get('placement')
    
    query = Ad.query.filter_by(status=AdStatus.ACTIVE)
    if placement:
        query = query.filter_by(placement=placement)
    
    ads = query.all()
    
    return jsonify({
        'ads': [ad.to_dict() for ad in ads]
    }), 200


@app.route('/api/ads', methods=['POST'])
@super_admin_required
@limiter.limit("30 per hour")
def create_ad(current_user):
    data = request.get_json()
    
    required_fields = ['name', 'ad_unit_id', 'placement']
    if not all(k in data for k in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    ad = Ad(
        name=data['name'],
        ad_unit_id=data['ad_unit_id'],
        placement=data['placement'],
        status=AdStatus.ACTIVE,
        config=data.get('config', {})
    )
    
    db.session.add(ad)
    db.session.commit()
    
    log_audit(current_user.id, 'CREATE_AD', 'ad', str(ad.id), {'name': ad.name, 'placement': ad.placement})
    
    return jsonify({
        'message': 'Ad created successfully',
        'ad': ad.to_dict()
    }), 201


@app.route('/api/ads/<ad_id>', methods=['PUT'])
@super_admin_required
def update_ad(current_user, ad_id):
    ad = Ad.query.get(ad_id)
    if not ad:
        return jsonify({'error': 'Ad not found'}), 404
    
    data = request.get_json()
    changes = {}
    
    for field in ['name', 'ad_unit_id', 'placement', 'config']:
        if field in data:
            old_value = getattr(ad, field)
            if old_value != data[field]:
                changes[field] = {'old': str(old_value), 'new': str(data[field])}
                setattr(ad, field, data[field])
    
    if 'status' in data:
        try:
            new_status = AdStatus(data['status'])
            if ad.status != new_status:
                changes['status'] = {'old': ad.status.value, 'new': new_status.value}
                ad.status = new_status
        except ValueError:
            return jsonify({'error': 'Invalid status'}), 400
    
    db.session.commit()
    log_audit(current_user.id, 'UPDATE_AD', 'ad', ad_id, changes)
    
    return jsonify({
        'message': 'Ad updated successfully',
        'ad': ad.to_dict()
    }), 200


@app.route('/api/ads/<ad_id>', methods=['DELETE'])
@super_admin_required
def delete_ad(current_user, ad_id):
    ad = Ad.query.get(ad_id)
    if not ad:
        return jsonify({'error': 'Ad not found'}), 404
    
    db.session.delete(ad)
    db.session.commit()
    
    log_audit(current_user.id, 'DELETE_AD', 'ad', ad_id, {'name': ad.name})
    
    return jsonify({'message': 'Ad deleted successfully'}), 200


@app.route('/api/ads/<ad_id>/track', methods=['POST'])
@limiter.limit("1000 per minute")
def track_ad_event(ad_id):
    ad = Ad.query.get(ad_id)
    if not ad:
        return jsonify({'error': 'Ad not found'}), 404
    
    data = request.get_json()
    event_type = data.get('event_type')
    
    if event_type == 'impression':
        ad.impression_count += 1
        log_analytics('AD_IMPRESSION', 'ad', ad_id)
    elif event_type == 'click':
        ad.click_count += 1
        log_analytics('AD_CLICK', 'ad', ad_id)
    else:
        return jsonify({'error': 'Invalid event type'}), 400
    
    db.session.commit()
    
    return jsonify({'message': 'Event tracked'}), 200


@app.route('/api/job-sources', methods=['GET'])
@super_admin_required
def list_job_sources(current_user):
    sources = JobSource.query.order_by(JobSource.name).all()
    return jsonify({
        'sources': [source.to_dict() for source in sources]
    }), 200


@app.route('/api/job-sources', methods=['POST'])
@super_admin_required
def create_job_source(current_user):
    data = request.get_json()
    
    required_fields = ['name', 'source_type']
    if not all(k in data for k in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    source = JobSource(
        name=data['name'],
        source_type=data['source_type'],
        api_url=data.get('api_url'),
        api_key=data.get('api_key'),
        config=data.get('config', {}),
        sync_frequency_minutes=data.get('sync_frequency_minutes', 60)
    )
    
    db.session.add(source)
    db.session.commit()
    
    log_audit(current_user.id, 'CREATE_JOB_SOURCE', 'job_source', str(source.id), {'name': source.name})
    
    return jsonify({
        'message': 'Job source created successfully',
        'source': source.to_dict()
    }), 201


@app.route('/api/job-sources/<source_id>', methods=['PUT'])
@super_admin_required
def update_job_source(current_user, source_id):
    source = JobSource.query.get(source_id)
    if not source:
        return jsonify({'error': 'Job source not found'}), 404
    
    data = request.get_json()
    changes = {}
    
    updatable_fields = ['name', 'source_type', 'api_url', 'api_key', 'config', 'is_active', 'sync_frequency_minutes']
    
    for field in updatable_fields:
        if field in data:
            old_value = getattr(source, field)
            if old_value != data[field]:
                changes[field] = {'old': str(old_value), 'new': str(data[field])}
                setattr(source, field, data[field])
    
    db.session.commit()
    log_audit(current_user.id, 'UPDATE_JOB_SOURCE', 'job_source', source_id, changes)
    
    return jsonify({
        'message': 'Job source updated successfully',
        'source': source.to_dict()
    }), 200


@app.route('/api/job-sources/<source_id>/sync', methods=['POST'])
@super_admin_required
def sync_job_source(current_user, source_id):
    source = JobSource.query.get(source_id)
    if not source:
        return jsonify({'error': 'Job source not found'}), 404
    
    try:
        sync_jobs_from_source(source)
        return jsonify({'message': 'Sync completed successfully'}), 200
    except Exception as e:
        return jsonify({'error': f'Sync failed: {str(e)}'}), 500


@app.route('/api/analytics/dashboard', methods=['GET'])
@super_admin_required
def get_analytics_dashboard(current_user):
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    total_jobs = Job.query.filter_by(status=JobStatus.APPROVED).count()
    total_services = Service.query.filter_by(status=ServiceStatus.ACTIVE).count()
    total_users = User.query.filter_by(is_active=True).count()
    
    pending_jobs = Job.query.filter_by(status=JobStatus.PENDING).count()
    
    job_views = db.session.query(func.sum(Job.view_count)).scalar() or 0
    service_views = db.session.query(func.sum(Service.view_count)).scalar() or 0
    
    visitors = AnalyticsEvent.query.filter(
        AnalyticsEvent.created_at >= start_date
    ).distinct(AnalyticsEvent.session_id).count()
    
    page_views = AnalyticsEvent.query.filter(
        AnalyticsEvent.created_at >= start_date,
        AnalyticsEvent.event_type.in_(['VIEW_JOB', 'VIEW_SERVICE', 'LIST_JOBS', 'LIST_SERVICES'])
    ).count()
    
    ad_impressions = db.session.query(func.sum(Ad.impression_count)).scalar() or 0
    ad_clicks = db.session.query(func.sum(Ad.click_count)).scalar() or 0
    
    top_jobs = db.session.query(
        Job.id, Job.title, Job.view_count, Job.apply_count
    ).filter_by(status=JobStatus.APPROVED).order_by(
        Job.view_count.desc()
    ).limit(10).all()
    
    top_services = db.session.query(
        Service.id, Service.title, Service.view_count, Service.inquiry_count
    ).filter_by(status=ServiceStatus.ACTIVE).order_by(
        Service.view_count.desc()
    ).limit(10).all()
    
    return jsonify({
        'summary': {
            'total_jobs': total_jobs,
            'total_services': total_services,
            'total_users': total_users,
            'pending_jobs': pending_jobs,
            'job_views': job_views,
            'service_views': service_views,
            'visitors': visitors,
            'page_views': page_views,
            'ad_impressions': ad_impressions,
            'ad_clicks': ad_clicks,
        },
        'top_jobs': [
            {
                'id': str(job[0]),
                'title': job[1],
                'views': job[2],
                'applies': job[3]
            }
            for job in top_jobs
        ],
        'top_services': [
            {
                'id': str(service[0]),
                'title': service[1],
                'views': service[2],
                'inquiries': service[3]
            }
            for service in top_services
        ]
    }), 200


@app.route('/api/analytics/events', methods=['GET'])
@super_admin_required
def get_analytics_events(current_user):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    query = AnalyticsEvent.query
    
    event_type = request.args.get('event_type')
    if event_type:
        query = query.filter_by(event_type=event_type)
    
    entity_type = request.args.get('entity_type')
    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    
    start_date = request.args.get('start_date')
    if start_date:
        query = query.filter(AnalyticsEvent.created_at >= datetime.fromisoformat(start_date))
    
    pagination = query.order_by(AnalyticsEvent.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'events': [event.to_dict() for event in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/audit-logs', methods=['GET'])
@super_admin_required
def get_audit_logs(current_user):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    query = AuditLog.query
    
    user_id = request.args.get('user_id')
    if user_id:
        query = query.filter_by(user_id=user_id)
    
    action = request.args.get('action')
    if action:
        query = query.filter_by(action=action)
    
    entity_type = request.args.get('entity_type')
    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    
    start_date = request.args.get('start_date')
    if start_date:
        query = query.filter(AuditLog.created_at >= datetime.fromisoformat(start_date))
    
    pagination = query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'logs': [log.to_dict() for log in pagination.items],
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages
    }), 200


@app.route('/api/settings', methods=['GET'])
@super_admin_required
def get_settings(current_user):
    settings = SiteSettings.query.all()
    return jsonify({
        'settings': {s.key: s.value for s in settings}
    }), 200


@app.route('/api/settings/<key>', methods=['GET'])
@limiter.limit("100 per minute")
def get_setting(key):
    setting = SiteSettings.query.filter_by(key=key).first()
    if not setting:
        return jsonify({'error': 'Setting not found'}), 404
    return jsonify({'setting': setting.to_dict()}), 200


@app.route('/api/settings', methods=['POST'])
@super_admin_required
def update_setting(current_user):
    data = request.get_json()
    
    if 'key' not in data or 'value' not in data:
        return jsonify({'error': 'Missing key or value'}), 400
    
    setting = SiteSettings.query.filter_by(key=data['key']).first()
    
    if setting:
        old_value = setting.value
        setting.value = data['value']
        if 'description' in data:
            setting.description = data['description']
        action = 'UPDATE_SETTING'
    else:
        setting = SiteSettings(
            key=data['key'],
            value=data['value'],
            description=data.get('description')
        )
        db.session.add(setting)
        old_value = None
        action = 'CREATE_SETTING'
    
    db.session.commit()
    
    log_audit(
        current_user.id,
        action,
        'setting',
        str(setting.id),
        {'key': data['key'], 'old_value': old_value, 'new_value': data['value']}
    )
    
    return jsonify({
        'message': 'Setting updated successfully',
        'setting': setting.to_dict()
    }), 200


def sync_jobs_from_source(source: JobSource):
    if not source.is_active or not source.api_url:
        return
    
    try:
        headers = {}
        if source.api_key:
            headers['Authorization'] = f'Bearer {source.api_key}'
        
        response = requests.get(source.api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        jobs_data = response.json()
        
        if not isinstance(jobs_data, list):
            jobs_data = jobs_data.get('jobs', [])
        
        imported_count = 0
        
        for job_data in jobs_data:
            external_id = job_data.get('id') or job_data.get('external_id')
            
            if external_id:
                existing_job = Job.query.filter_by(
                    source_id=source.id,
                    external_id=str(external_id)
                ).first()
                
                if existing_job:
                    continue
            
            try:
                job_type = JobType.GOVERNMENT
                if 'private' in job_data.get('type', '').lower():
                    job_type = JobType.PRIVATE
                elif 'remote' in job_data.get('type', '').lower():
                    job_type = JobType.REMOTE
            except:
                job_type = JobType.GOVERNMENT
            
            job = Job(
                title=job_data.get('title', 'Untitled Job'),
                description=job_data.get('description', ''),
                company=job_data.get('company', 'Unknown'),
                location=job_data.get('location'),
                job_type=job_type,
                status=JobStatus.PENDING,
                salary_min=job_data.get('salary_min'),
                salary_max=job_data.get('salary_max'),
                application_url=job_data.get('application_url'),
                application_deadline=datetime.fromisoformat(job_data['deadline']) if job_data.get('deadline') else None,
                qualifications=job_data.get('qualifications', []),
                tags=job_data.get('tags', []),
                source_id=source.id,
                external_id=str(external_id) if external_id else None
            )
            
            db.session.add(job)
            imported_count += 1
        
        source.last_sync = datetime.utcnow()
        db.session.commit()
        
        print(f"Imported {imported_count} jobs from {source.name}")
        
    except Exception as e:
        print(f"Error syncing jobs from {source.name}: {str(e)}")
        db.session.rollback()


def scheduled_job_sync():
    with app.app_context():
        sources = JobSource.query.filter_by(is_active=True).all()
        
        for source in sources:
            if not source.last_sync or \
               (datetime.utcnow() - source.last_sync).total_seconds() >= source.sync_frequency_minutes * 60:
                sync_jobs_from_source(source)


def scheduled_job_expiry_check():
    with app.app_context():
        expired_jobs = Job.query.filter(
            Job.status == JobStatus.APPROVED,
            Job.application_deadline < datetime.utcnow().date()
        ).all()
        
        for job in expired_jobs:
            job.status = JobStatus.EXPIRED
        
        if expired_jobs:
            db.session.commit()
            print(f"Marked {len(expired_jobs)} jobs as expired")


scheduler = BackgroundScheduler()
scheduler.add_job(
    scheduled_job_sync,
    CronTrigger(minute='*/30'),
    id='job_sync',
    replace_existing=True
)
scheduler.add_job(
    scheduled_job_expiry_check,
    CronTrigger(hour='0', minute='0'),
    id='job_expiry_check',
    replace_existing=True
)


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db.session.execute(text('SELECT 1'))
        db_status = 'healthy'
    except Exception as e:
        db_status = f'unhealthy: {str(e)}'
    
    return jsonify({
        'status': 'healthy' if db_status == 'healthy' else 'degraded',
        'database': db_status,
        'scheduler': 'running' if scheduler.running else 'stopped'
    }), 200 if db_status == 'healthy' else 503


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        if not User.query.filter_by(role=UserRole.SUPER_ADMIN).first():
            admin = User(
                email=os.getenv('SUPER_ADMIN_EMAIL', 'admin@sridharservices.com'),
                full_name='Super Admin',
                role=UserRole.SUPER_ADMIN,
                is_active=True
            )
            admin.set_password(os.getenv('SUPER_ADMIN_PASSWORD', 'ChangeMe123!'))
            db.session.add(admin)
            db.session.commit()
            print(f"Super admin created: {admin.email}")
    
    if not scheduler.running:
        scheduler.start()
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)