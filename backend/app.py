import os
import secrets
import json
import asyncio
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any

from flask import (
    Flask, request, jsonify, render_template, 
    redirect, url_for, flash, session, g, abort,
    send_from_directory, make_response
)
from flask_login import (
    LoginManager, login_user, logout_user, 
    login_required, current_user
)
from flask_cors import CORS
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_compress import Compress
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from prometheus_flask_exporter import PrometheusMetrics
from celery import Celery
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.redis import RedisJobStore
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from dotenv import load_dotenv
import redis
import logging
from logging.handlers import RotatingFileHandler
import stripe
from email_validator import validate_email
import jwt
import hashlib
import bleach

# Import models and services
from models import (
    db, User, Job, Ad, Analytics, UserSession, UserPermission,
    AuditLog, JobFetcherService, AnalyticsService, DatabaseMaintenance,
    UserRole, JobStatus, JobSource, AdType, redis_client
)

# Load environment variables
load_dotenv()

# Initialize Sentry for error tracking
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )

# Create Flask app with optimizations
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)

# Production configuration
class Config:
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://user:pass@localhost/sridhar_services'
    ).replace('postgres://', 'postgresql://')
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 40,
        'connect_args': {
            'connect_timeout': 10,
            'application_name': 'sridhar_services'
        }
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = False
    
    # Redis & Caching
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'sridhar_cache_'
    
    # Celery
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_TIMEZONE = 'UTC'
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_STRATEGY = 'fixed-window-elastic-expiry'
    RATELIMIT_DEFAULT = '100/hour'
    
    # File uploads
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
    
    # Stripe
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db, compare_type=True)
csrf = CSRFProtect(app)
compress = Compress(app)
cache = Cache(app)

# Security headers with Talisman
talisman = Talisman(
    app,
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,
    session_cookie_secure=True,
    session_cookie_http_only=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
        'font-src': "'self' https://fonts.gstatic.com",
        'img-src': "'self' data: https:",
        'connect-src': "'self' https://api.stripe.com"
    }
)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=app.config['REDIS_URL'],
    default_limits=["1000 per hour", "100 per minute"]
)

# CORS configuration
CORS(app, 
    origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
    allow_headers=['Content-Type', 'Authorization', 'X-CSRF-Token'],
    expose_headers=['X-Total-Count', 'X-Page', 'X-Per-Page'],
    supports_credentials=True,
    max_age=3600
)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.session_protection = 'strong'

# Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info('sridhar_services', 'Sridhar Internet Services', version='2.0.0')

# Celery configuration
celery = Celery(app.name)
celery.conf.update(app.config)

# APScheduler configuration
jobstores = {
    'default': RedisJobStore(
        host='localhost',
        port=6379,
        db=1
    )
}
scheduler = BackgroundScheduler(jobstores=jobstores, timezone='UTC')
scheduler.start()

# Configure logging
if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/sridhar_services.log',
        maxBytes=10485760,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Sridhar Services startup')

# Stripe configuration
if app.config['STRIPE_SECRET_KEY']:
    stripe.api_key = app.config['STRIPE_SECRET_KEY']

# Request handling middleware
@app.before_request
def before_request():
    """Pre-request processing"""
    g.start_time = datetime.utcnow()
    g.request_id = secrets.token_hex(16)
    
    # Session management
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(32)
        session.permanent = True
    
    # Security checks
    if current_user.is_authenticated:
        # Check if session is still valid
        user_session = UserSession.query.filter_by(
            user_id=current_user.id,
            session_token=session.get('session_token'),
            is_active=True
        ).first()
        
        if not user_session or user_session.expires_at < datetime.utcnow():
            logout_user()
            flash('Your session has expired. Please login again.', 'warning')
            return redirect(url_for('auth.login'))
    
    # Track request
    if not request.path.startswith('/static'):
        AnalyticsService.track_event(
            request,
            'pageview',
            page_title=request.endpoint
        )

@app.after_request
def after_request(response):
    """Post-request processing"""
    if hasattr(g, 'start_time'):
        elapsed = (datetime.utcnow() - g.start_time).total_seconds()
        response.headers['X-Response-Time'] = str(elapsed)
    
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    # Security headers
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors"""
    return jsonify({
        'error': 'Bad Request',
        'message': str(e),
        'request_id': g.get('request_id')
    }), 400

@app.errorhandler(401)
def unauthorized(e):
    """Handle unauthorized errors"""
    return jsonify({
        'error': 'Unauthorized',
        'message': 'Authentication required',
        'request_id': g.get('request_id')
    }), 401

@app.errorhandler(403)
def forbidden(e):
    """Handle forbidden errors"""
    return jsonify({
        'error': 'Forbidden',
        'message': 'You do not have permission to access this resource',
        'request_id': g.get('request_id')
    }), 403

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found',
        'request_id': g.get('request_id')
    }), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Too Many Requests',
        'message': f"Rate limit exceeded: {e.description}",
        'request_id': g.get('request_id')
    }), 429

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    db.session.rollback()
    app.logger.error(f"Internal error: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'request_id': g.get('request_id')
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all other exceptions"""
    if isinstance(e, HTTPException):
        return e
    
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'Server Error',
        'message': 'An unexpected error occurred',
        'request_id': g.get('request_id')
    }), 500

# User loader
@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return User.query.get(user_id)

# Custom decorators
def require_api_key(f):
    """Decorator to require API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        user = User.query.filter_by(api_key=api_key).first()
        
        if not user or not user.is_active:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check rate limit for API
        key = f"api_rate_limit:{user.id}"
        try:
            current = redis_client.incr(key)
            if current == 1:
                redis_client.expire(key, 3600)
            
            if current > user.api_rate_limit:
                return jsonify({'error': 'API rate limit exceeded'}), 429
        except:
            pass
        
        g.api_user = user
        return f(*args, **kwargs)
    
    return decorated_function

def permission_required(permission):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if not current_user.has_permission(permission):
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def validate_json(*required_fields):
    """Decorator to validate JSON input"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
@validate_json('email', 'password', 'name')
def register():
    """User registration with email verification"""
    data = request.get_json()
    
    # Validate email
    try:
        valid = validate_email(data['email'])
        email = valid.email
    except:
        return jsonify({'error': 'Invalid email address'}), 400
    
    # Check if user exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Check if username exists
    username = data.get('username', email.split('@')[0])
    if User.query.filter_by(username=username).first():
        username = f"{username}_{secrets.token_hex(4)}"
    
    try:
        # Create user
        user = User(
            email=email,
            username=username,
            name=bleach.clean(data['name']),
            role=UserRole.VIEWER
        )
        user.set_password(data['password'])
        
        # Generate verification token
        verification_token = user.generate_auth_token(expires_in=86400)
        
        # Send verification email (using Celery task)
        send_verification_email.delay(email, verification_token)
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'Registration successful. Please check your email to verify your account.',
            'user_id': str(user.id)
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per hour")
@validate_json('email', 'password')
def login():
    """User login with session management"""
    data = request.get_json()
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403
    
    if not user.is_verified:
        return jsonify({'error': 'Please verify your email first'}), 403
    
    # Create session
    session_token = secrets.token_hex(32)
    user_session = UserSession(
        user_id=user.id,
        session_token=session_token,
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string[:500],
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    
    db.session.add(user_session)
    db.session.commit()
    
    # Login user
    login_user(user, remember=True)
    session['session_token'] = session_token
    
    # Generate JWT token
    access_token = user.generate_auth_token()
    
    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': str(user.id),
            'email': user.email,
            'name': user.name,
            'role': user.role.value
        },
        'access_token': access_token,
        'csrf_token': generate_csrf()
    }), 200

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """User logout with session cleanup"""
    # Invalidate current session
    if 'session_token' in session:
        UserSession.query.filter_by(
            session_token=session['session_token']
        ).update({'is_active': False})
        db.session.commit()
    
    logout_user()
    session.clear()
    
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/api/auth/verify/<token>', methods=['GET'])
def verify_email(token):
    """Verify email address"""
    user = User.verify_auth_token(token)
    
    if not user:
        return jsonify({'error': 'Invalid or expired token'}), 400
    
    user.is_verified = True
    user.email_verified_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'message': 'Email verified successfully'}), 200

# Public API routes
@app.route('/api/health', methods=['GET'])
@cache.cached(timeout=10)
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except:
        db_status = 'unhealthy'
    
    try:
        # Check Redis
        redis_client.ping()
        redis_status = 'healthy'
    except:
        redis_status = 'unhealthy'
    
    return jsonify({
        'status': 'healthy' if db_status == 'healthy' and redis_status == 'healthy' else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': db_status,
            'redis': redis_status,
            'version': '2.0.0'
        }
    })

@app.route('/api/jobs', methods=['GET'])
@cache.cached(timeout=60, query_string=True)
def get_jobs():
    """Get jobs with advanced filtering and pagination"""
    # Parse query parameters
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    
    # Filters
    category = request.args.get('category')
    location = request.args.get('location')
    company = request.args.get('company')
    job_type = request.args.get('job_type')
    salary_min = request.args.get('salary_min', type=float)
    search = request.args.get('search')
    sort_by = request.args.get('sort_by', 'created_at')
    order = request.args.get('order', 'desc')
    
    # Build query
    query = Job.query.filter(
        Job.status == JobStatus.APPROVED,
        Job.expires_at > datetime.utcnow(),
        Job.deleted_at.is_(None)
    )
    
    # Apply filters
    if category:
        query = query.filter(Job.category == category)
    
    if location:
        query = query.filter(Job.location.ilike(f'%{location}%'))
    
    if company:
        query = query.filter(Job.company.ilike(f'%{company}%'))
    
    if job_type:
        query = query.filter(Job.job_type == job_type)
    if salary_min:
        query = query.filter(Job.salary_max >= salary_min)
    
    # Full-text search
    if search:
        query = query.filter(
            db.func.to_tsvector('english', Job.title + ' ' + Job.description).match(search)
        )
    
    # Sorting
    if sort_by == 'salary':
        order_column = Job.salary_max if order == 'desc' else Job.salary_min
    elif sort_by == 'quality':
        order_column = Job.quality_score
    else:
        order_column = getattr(Job, sort_by, Job.created_at)
    
    if order == 'desc':
        query = query.order_by(order_column.desc())
    else:
        query = query.order_by(order_column.asc())
    
    # Pagination
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Increment view count for displayed jobs
    job_ids = [job.id for job in paginated.items]
    if job_ids:
        Job.query.filter(Job.id.in_(job_ids)).update(
            {Job.view_count: Job.view_count + 1},
            synchronize_session=False
        )
        db.session.commit()
    
    return jsonify({
        'jobs': [job.to_dict(exclude=['deleted_at']) for job in paginated.items],
        'pagination': {
            'total': paginated.total,
            'pages': paginated.pages,
            'current_page': page,
            'per_page': per_page,
            'has_next': paginated.has_next,
            'has_prev': paginated.has_prev
        }
    }), 200

@app.route('/api/jobs/<uuid:job_id>', methods=['GET'])
@cache.cached(timeout=300)
def get_job(job_id):
    """Get single job details"""
    job = Job.query.filter_by(
        id=job_id,
        status=JobStatus.APPROVED
    ).first_or_404()
    
    # Increment view count
    job.view_count += 1
    db.session.commit()
    
    # Track event
    AnalyticsService.track_event(
        request,
        'job_view',
        category='jobs',
        label=str(job_id)
    )
    
    return jsonify(job.to_dict(exclude=['deleted_at'])), 200

@app.route('/api/jobs/<uuid:job_id>/apply', methods=['POST'])
@limiter.limit("10 per hour")
def apply_for_job(job_id):
    """Apply for a job"""
    job = Job.query.filter_by(
        id=job_id,
        status=JobStatus.APPROVED
    ).first_or_404()
    
    # Increment apply count
    job.apply_count += 1
    db.session.commit()
    
    # Track event
    AnalyticsService.track_event(
        request,
        'job_apply',
        category='jobs',
        label=str(job_id),
        value=1
    )
    
    # Process application (send to external URL or save)
    if job.source_url:
        return jsonify({
            'message': 'Redirecting to application',
            'redirect_url': job.source_url
        }), 200
    
    return jsonify({'message': 'Application submitted successfully'}), 200

@app.route('/api/categories', methods=['GET'])
@cache.cached(timeout=3600)
def get_categories():
    """Get job categories with counts"""
    categories = db.session.query(
        Job.category,
        db.func.count(Job.id).label('count')
    ).filter(
        Job.status == JobStatus.APPROVED,
        Job.expires_at > datetime.utcnow(),
        Job.deleted_at.is_(None)
    ).group_by(Job.category).order_by('count').all()
    
    return jsonify([
        {'name': cat[0], 'count': cat[1]} 
        for cat in categories if cat[0]
    ]), 200

# Admin API routes
@app.route('/api/admin/dashboard', methods=['GET'])
@admin_required
def admin_dashboard():
    """Admin dashboard with comprehensive metrics"""
    # Job statistics
    total_jobs = Job.query.count()
    pending_jobs = Job.query.filter_by(status=JobStatus.PENDING).count()
    approved_jobs = Job.query.filter_by(status=JobStatus.APPROVED).count()
    expired_jobs = Job.query.filter(Job.expires_at < datetime.utcnow()).count()
    
    # User statistics
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    verified_users = User.query.filter_by(is_verified=True).count()
    
    # Recent activity
    recent_jobs = Job.query.order_by(Job.created_at.desc()).limit(10).all()
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    # Analytics summary
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    analytics_summary = AnalyticsService.get_dashboard_metrics(
        datetime.combine(week_ago, datetime.min.time()),
        datetime.utcnow()
    )
    
    return jsonify({
        'statistics': {
            'jobs': {
                'total': total_jobs,
                'pending': pending_jobs,
                'approved': approved_jobs,
                'expired': expired_jobs
            },
            'users': {
                'total': total_users,
                'active': active_users,
                'verified': verified_users
            }
        },
        'recent_activity': {
            'jobs': [job.to_dict(exclude=['description', 'requirements']) for job in recent_jobs],
            'users': [
                {
                    'id': str(user.id),
                    'name': user.name,
                    'email': user.email,
                    'created_at': user.created_at.isoformat()
                } for user in recent_users
            ]
        },
        'analytics': analytics_summary
    }), 200

@app.route('/api/admin/jobs', methods=['GET'])
@admin_required
def admin_get_jobs():
    """Get all jobs for admin management"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)
    status = request.args.get('status')
    source = request.args.get('source')
    
    query = Job.query
    
    if status:
        query = query.filter_by(status=JobStatus[status.upper()])
    
    if source:
        query = query.filter_by(source=JobSource[source.upper()])
    
    paginated = query.order_by(Job.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'jobs': [job.to_dict() for job in paginated.items],
        'pagination': {
            'total': paginated.total,
            'pages': paginated.pages,
            'current_page': page,
            'per_page': per_page
        }
    }), 200

@app.route('/api/admin/jobs', methods=['POST'])
@admin_required
@validate_json('title', 'description')
def admin_create_job():
    """Create new job"""
    data = request.get_json()
    
    try:
        job = Job(
            title=bleach.clean(data['title']),
            description=bleach.clean(data['description']),
            category=data.get('category'),
            company=bleach.clean(data.get('company', '')),
            location=bleach.clean(data.get('location', '')),
            salary_min=data.get('salary_min'),
            salary_max=data.get('salary_max'),
            job_type=data.get('job_type'),
            requirements=bleach.clean(data.get('requirements', '')),
            source=JobSource.MANUAL,
            status=JobStatus.APPROVED,
            created_by=current_user.id,
            approved_by=current_user.id,
            approved_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=int(data.get('days_to_expire', 30)))
        )
        
        # Generate slug
        job.slug = Job.generate_slug(job.title)
        
        # Calculate quality score
        job.calculate_quality_score()
        
        # Update search vector
        job.update_search_vector()
        
        db.session.add(job)
        db.session.commit()
        
        return jsonify({
            'message': 'Job created successfully',
            'job': job.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Job creation error: {str(e)}")
        return jsonify({'error': 'Failed to create job'}), 500

@app.route('/api/admin/jobs/<uuid:job_id>', methods=['PUT'])
@admin_required
def admin_update_job(job_id):
    """Update job"""
    job = Job.query.get_or_404(job_id)
    data = request.get_json()
    
    # Update allowed fields
    allowed_fields = [
        'title', 'description', 'category', 'company', 
        'location', 'salary_min', 'salary_max', 'job_type',
        'requirements', 'meta_title', 'meta_description'
    ]
    
    for field in allowed_fields:
        if field in data:
            value = bleach.clean(data[field]) if isinstance(data[field], str) else data[field]
            setattr(job, field, value)
    
    # Update slug if title changed
    if 'title' in data:
        job.slug = Job.generate_slug(job.title)
    
    # Recalculate quality score
    job.calculate_quality_score()
    
    # Update search vector
    job.update_search_vector()
    
    job.version += 1
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'Job updated successfully',
            'job': job.to_dict()
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/jobs/<uuid:job_id>', methods=['DELETE'])
@admin_required
def admin_delete_job(job_id):
    """Soft delete job"""
    job = Job.query.get_or_404(job_id)
    
    job.soft_delete()
    
    return jsonify({'message': 'Job deleted successfully'}), 200

@app.route('/api/admin/jobs/<uuid:job_id>/approve', methods=['POST'])
@admin_required
def admin_approve_job(job_id):
    """Approve or reject job"""
    job = Job.query.get_or_404(job_id)
    data = request.get_json()
    
    action = data.get('action', 'approve')
    
    if action == 'approve':
        job.status = JobStatus.APPROVED
        job.approved_by = current_user.id
        job.approved_at = datetime.utcnow()
    elif action == 'reject':
        job.status = JobStatus.REJECTED
    else:
        return jsonify({'error': 'Invalid action'}), 400
    
    db.session.commit()
    
    return jsonify({
        'message': f'Job {action}d successfully',
        'job': job.to_dict()
    }), 200

@app.route('/api/admin/jobs/bulk-approve', methods=['POST'])
@admin_required
@validate_json('job_ids', 'action')
def admin_bulk_approve_jobs():
    """Bulk approve/reject jobs"""
    data = request.get_json()
    job_ids = data['job_ids']
    action = data['action']
    
    if action not in ['approve', 'reject']:
        return jsonify({'error': 'Invalid action'}), 400
    
    try:
        if action == 'approve':
            Job.query.filter(Job.id.in_(job_ids)).update({
                Job.status: JobStatus.APPROVED,
                Job.approved_by: current_user.id,
                Job.approved_at: datetime.utcnow()
            }, synchronize_session=False)
        else:
            Job.query.filter(Job.id.in_(job_ids)).update({
                Job.status: JobStatus.REJECTED
            }, synchronize_session=False)
        
        db.session.commit()
        
        return jsonify({
            'message': f'{len(job_ids)} jobs {action}d successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/fetch-jobs', methods=['POST'])
@admin_required
def admin_fetch_jobs():
    """Manually trigger job fetching"""
    data = request.get_json()
    source = data.get('source', 'all')
    
    # Queue job fetching task
    fetch_jobs_task.delay(source)
    
    return jsonify({
        'message': f'Job fetching initiated for source: {source}'
    }), 202

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    """Get all users"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)
    role = request.args.get('role')
    
    query = User.query
    
    if role:
        query = query.filter_by(role=UserRole[role.upper()])
    
    paginated = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'users': [
            {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'name': user.name,
                'role': user.role.value,
                'is_active': user.is_active,
                'is_verified': user.is_verified,
                'created_at': user.created_at.isoformat(),
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None
            } for user in paginated.items
        ],
        'pagination': {
            'total': paginated.total,
            'pages': paginated.pages,
            'current_page': page,
            'per_page': per_page
        }
    }), 200

@app.route('/api/admin/users/<uuid:user_id>', methods=['PUT'])
@admin_required
def admin_update_user(user_id):
    """Update user"""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    # Prevent modifying super admin
    if user.role == UserRole.SUPER_ADMIN and current_user.role != UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Cannot modify super admin'}), 403
    
    # Update fields
    if 'role' in data and current_user.role == UserRole.SUPER_ADMIN:
        user.role = UserRole[data['role'].upper()]
    
    if 'is_active' in data:
        user.is_active = data['is_active']
    
    if 'api_rate_limit' in data:
        user.api_rate_limit = data['api_rate_limit']
    
    try:
        db.session.commit()
        return jsonify({'message': 'User updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<uuid:user_id>', methods=['DELETE'])
@permission_required('delete_users')
def admin_delete_user(user_id):
    """Delete user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting super admin
    if user.role == UserRole.SUPER_ADMIN:
        return jsonify({'error': 'Cannot delete super admin'}), 403
    
    # Soft delete
    user.soft_delete()
    
    return jsonify({'message': 'User deleted successfully'}), 200

# Manager API routes
@app.route('/api/manager/analytics', methods=['GET'])
@permission_required('view_analytics')
def manager_analytics():
    """Get analytics dashboard"""
    # Parse date range
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    
    if date_from:
        date_from = datetime.fromisoformat(date_from)
    else:
        date_from = datetime.utcnow() - timedelta(days=30)
    
    if date_to:
        date_to = datetime.fromisoformat(date_to)
    else:
        date_to = datetime.utcnow()
    
    # Get metrics
    metrics = AnalyticsService.get_dashboard_metrics(date_from, date_to)
    
    # Get ad performance
    ads_performance = db.session.query(
        Ad.ad_type,
        db.func.sum(Ad.impressions).label('impressions'),
        db.func.sum(Ad.clicks).label('clicks'),
        db.func.sum(Ad.revenue).label('revenue'),
        db.func.avg(Ad.ctr).label('avg_ctr')
    ).filter(
        Ad.date.between(date_from.date(), date_to.date())
    ).group_by(Ad.ad_type).all()
    
    # Get job performance
    job_performance = db.session.query(
        db.func.sum(Job.view_count).label('total_views'),
        db.func.sum(Job.apply_count).label('total_applies'),
        db.func.avg(Job.quality_score).label('avg_quality')
    ).filter(
        Job.created_at.between(date_from, date_to)
    ).first()
    
    return jsonify({
        'date_range': {
            'from': date_from.isoformat(),
            'to': date_to.isoformat()
        },
        'metrics': metrics,
        'ads': [
            {
                'type': ad[0].value if ad[0] else 'unknown',
                'impressions': ad[1] or 0,
                'clicks': ad[2] or 0,
                'revenue': float(ad[3]) if ad[3] else 0,
                'ctr': float(ad[4]) if ad[4] else 0
            } for ad in ads_performance
        ],
        'jobs': {
            'total_views': job_performance[0] or 0 if job_performance else 0,
            'total_applies': job_performance[1] or 0 if job_performance else 0,
            'avg_quality': float(job_performance[2]) if job_performance and job_performance[2] else 0
        }
    }), 200

@app.route('/api/manager/analytics/export', methods=['GET'])
@permission_required('export_analytics')
def export_analytics():
    """Export analytics data"""
    format = request.args.get('format', 'csv')
    date_from = request.args.get('from', (datetime.utcnow() - timedelta(days=30)).isoformat())
    date_to = request.args.get('to', datetime.utcnow().isoformat())
    
    # Queue export task
    export_task_id = export_analytics_task.delay(
        current_user.id,
        date_from,
        date_to,
        format
    ).id
    
    return jsonify({
        'message': 'Export initiated',
        'task_id': export_task_id
    }), 202

@app.route('/api/manager/backup', methods=['POST'])
@permission_required('manage_backups')
def create_backup():
    """Create database backup"""
    # Queue backup task
    backup_task.delay()
    
    return jsonify({
        'message': 'Backup initiated'
    }), 202

@app.route('/api/manager/audit-logs', methods=['GET'])
@permission_required('view_audit_logs')
def get_audit_logs():
    """Get audit logs"""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)
    table_name = request.args.get('table')
    action = request.args.get('action')
    user_id = request.args.get('user_id')
    
    query = AuditLog.query
    
    if table_name:
        query = query.filter_by(table_name=table_name)
    
    if action:
        query = query.filter_by(action=action)
    
    if user_id:
        query = query.filter_by(user_id=user_id)
    
    paginated = query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'logs': [log.to_dict() for log in paginated.items],
        'pagination': {
            'total': paginated.total,
            'pages': paginated.pages,
            'current_page': page,
            'per_page': per_page
        }
    }), 200

# Ad tracking endpoints
@app.route('/api/ads/impression', methods=['POST'])
@limiter.limit("1000 per minute")
def track_ad_impression():
    """Track ad impression"""
    data = request.get_json()
    ad_type = data.get('ad_type', 'banner')
    campaign_id = data.get('campaign_id')
    
    try:
        today = datetime.utcnow().date()
        hour = datetime.utcnow().hour
        
        # Find or create ad record
        ad = Ad.query.filter_by(
            ad_type=AdType[ad_type.upper()],
            campaign_id=campaign_id,
            date=today,
            hour=hour
        ).first()
        
        if ad:
            ad.impressions += 1
        else:
            ad = Ad(
                ad_type=AdType[ad_type.upper()],
                campaign_id=campaign_id,
                date=today,
                hour=hour,
                impressions=1
            )
            db.session.add(ad)
        
        ad.calculate_metrics()
        db.session.commit()
        
        return jsonify({'status': 'tracked'}), 200
        
    except Exception as e:
        app.logger.error(f"Ad impression tracking error: {str(e)}")
        return jsonify({'status': 'error'}), 200

@app.route('/api/ads/click', methods=['POST'])
@limiter.limit("100 per minute")
def track_ad_click():
    """Track ad click"""
    data = request.get_json()
    ad_type = data.get('ad_type', 'banner')
    campaign_id = data.get('campaign_id')
    
    try:
        today = datetime.utcnow().date()
        hour = datetime.utcnow().hour
        
        ad = Ad.query.filter_by(
            ad_type=AdType[ad_type.upper()],
            campaign_id=campaign_id,
            date=today,
            hour=hour
        ).first()
        
        if ad:
            ad.clicks += 1
            ad.calculate_metrics()
            db.session.commit()
        
        # Track in analytics
        AnalyticsService.track_event(
            request,
            'ad_click',
            category='ads',
            label=f"{ad_type}_{campaign_id}"
        )
        
        return jsonify({'status': 'tracked'}), 200
        
    except Exception as e:
        app.logger.error(f"Ad click tracking error: {str(e)}")
        return jsonify({'status': 'error'}), 200

# Webhook endpoints
@app.route('/api/webhooks/stripe', methods=['POST'])
@csrf.exempt
def stripe_webhook():
    """Handle Stripe webhooks"""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, app.config['STRIPE_WEBHOOK_SECRET']
        )
        
        # Handle different event types
        if event['type'] == 'payment_intent.succeeded':
            payment_intent = event['data']['object']
            # Process successful payment
            app.logger.info(f"Payment succeeded: {payment_intent['id']}")
        
        elif event['type'] == 'customer.subscription.created':
            subscription = event['data']['object']
            # Process new subscription
            app.logger.info(f"Subscription created: {subscription['id']}")
        
        return jsonify({'status': 'success'}), 200
        
    except ValueError:
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({'error': 'Invalid signature'}), 400

# Celery tasks
@celery.task(bind=True, max_retries=3)
def fetch_jobs_task(self, source='all'):
    """Async task to fetch jobs"""
    try:
        fetcher = JobFetcherService()
        jobs = []
        
        if source in ['all', 'rss']:
            rss_feeds = {
                'PIB': 'https://pib.gov.in/RssMain.aspx',
                'Employment News': 'http://employmentnews.gov.in/Feed',
                'SarkariResult': 'https://www.sarkariresult.com/rss.xml'
            }
            jobs.extend(asyncio.run(fetcher.fetch_rss_async(rss_feeds)))
        
        if source in ['all', 'api']:
            api_configs = [
                {
                    'name': 'Adzuna',
                    'url': 'https://api.adzuna.com/v1/api/jobs/in/search',
                    'params': {
                        'app_id': os.getenv('ADZUNA_APP_ID'),
                        'app_key': os.getenv('ADZUNA_API_KEY'),
                        'results_per_page': 50
                    },
                    'parser': {
                        'path': 'results',
                        'title': 'title',
                        'description': 'description',
                        'company': 'company.display_name',
                        'location': 'location.display_name',
                        'url': 'redirect_url'
                    }
                }
            ]
            jobs.extend(fetcher.fetch_api_jobs_batch(api_configs))
        
        if source in ['all', 'scraper']:
            scraper_configs = [
                {
                    'name': 'TSPSC',
                    'url': 'https://www.tspsc.gov.in',
                    'base_url': 'https://www.tspsc.gov.in',
                    'job_selector': '.notification-item',
                    'title_selector': 'h3',
                    'description_selector': '.description',
                    'url_selector': 'a'
                }
            ]
            jobs.extend(fetcher.scrape_websites_intelligent(scraper_configs))
        
        # Save jobs to database
        if jobs:
            Job.bulk_insert(jobs)
            app.logger.info(f"Fetched and saved {len(jobs)} jobs from {source}")
        
        return {'status': 'success', 'jobs_fetched': len(jobs)}
        
    except Exception as e:
        app.logger.error(f"Job fetching failed: {str(e)}")
        self.retry(countdown=60)

@celery.task
def send_verification_email(email, token):
    """Send email verification"""
    try:
        # Implement email sending logic
        # Using SendGrid, AWS SES, or other email service
        app.logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        app.logger.error(f"Email sending failed: {str(e)}")
        return False

@celery.task
def export_analytics_task(user_id, date_from, date_to, format):
    """Export analytics data"""
    try:
        # Generate export file
        # Upload to cloud storage
        # Send notification to user
        app.logger.info(f"Analytics exported for user {user_id}")
        return {'status': 'success'}
    except Exception as e:
        app.logger.error(f"Export failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

@celery.task
def backup_task():
    """Perform database backup"""
    try:
        backup_file = DatabaseMaintenance.backup_database()
        if backup_file:
            app.logger.info(f"Backup completed: {backup_file}")
            return {'status': 'success', 'file': backup_file}
    except Exception as e:
        app.logger.error(f"Backup failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

@celery.task
def cleanup_task():
    """Periodic cleanup task"""
    try:
        DatabaseMaintenance.clean_old_data(days=90)
        DatabaseMaintenance.vacuum_analyze()
        DatabaseMaintenance.update_statistics()
        app.logger.info("Cleanup task completed")
    except Exception as e:
        app.logger.error(f"Cleanup failed: {str(e)}")

# Scheduled tasks
scheduler.add_job(
    func=lambda: fetch_jobs_task.delay('all'),
    trigger='interval',
    hours=6,
    id='fetch_all_jobs',
    replace_existing=True
)

scheduler.add_job(
    func=lambda: cleanup_task.delay(),
    trigger='cron',
    hour=3,
    minute=0,
    id='daily_cleanup',
    replace_existing=True
)

scheduler.add_job(
    func=lambda: backup_task.delay(),
    trigger='cron',
    hour=2,
    minute=0,
    id='daily_backup',
    replace_existing=True
)

# CLI commands
@app.cli.command()
def init_db():
    """Initialize database"""
    db.create_all()
    print("Database initialized")

@app.cli.command()
def create_superadmin():
    """Create super admin user"""
    import getpass
    
    email = input("Email: ")
    username = input("Username: ")
    name = input("Name: ")
    password = getpass.getpass("Password: ")
    
    try:
        user = User(
            email=email,
            username=username,
            name=name,
            role=UserRole.SUPER_ADMIN,
            is_active=True,
            is_verified=True,
            email_verified_at=datetime.utcnow()
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        print(f"Super admin created: {email}")
    except Exception as e:
        print(f"Error: {str(e)}")

@app.cli.command()
def seed_data():
    """Seed sample data"""
    from faker import Faker
    fake = Faker()
    
    # Create sample jobs
    for _ in range(100):
        job = Job(
            title=fake.job(),
            description=fake.text(500),
            company=fake.company(),
            location=fake.city(),
            category=fake.random_element(['IT', 'Government', 'Banking', 'Healthcare']),
            salary_min=fake.random_int(20000, 50000),
            salary_max=fake.random_int(50001, 150000),
            source=JobSource.MANUAL,
            status=JobStatus.APPROVED,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        job.calculate_quality_score()
        db.session.add(job)
    
    db.session.commit()
    print("Sample data seeded")

# Application factory pattern support
def create_app(config_name='production'):
    """Create Flask application"""
    app.config.from_object(Config)
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    cache.init_app(app)
    compress.init_app(app)
    
    return app

# Main entry point
if __name__ == '__main__':
    # Development server (use Gunicorn in production)
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )