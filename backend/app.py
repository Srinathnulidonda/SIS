#backend/app.py
import os
import logging
import secrets
import re
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, List
from urllib.parse import urlencode, quote
import hashlib
import jwt

from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import or_, and_, func, text, inspect
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.dialects.postgresql import UUID
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import cloudinary
import cloudinary.uploader
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
import feedparser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if not os.environ.get('PRODUCTION') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.url_map.strict_slashes = False

DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://database_v3r2_user:QaF6Nczo8NaoB6XZo09SpCJ3EVvnNPNx@dpg-d3cik4j7mgec73aho5qg-a.oregon-postgres.render.com/database_v3r2'
)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'connect_args': {
        'connect_timeout': 10,
        'options': '-c timezone=utc'
    }
}

JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d3cikjqdbo4c73e72slg:mirq8x6uekGSDV0O3eb1eVjUG3GuYkVe@red-d3cikjqdbo4c73e72slg:6379')

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'ddnjtoppb'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '555997244452184'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', '310ZOZJRsBwACtN9oTTQsC4S4j4')
)

db = SQLAlchemy(app)

CORS(app, 
     origins=[
         "https://sridharinternetservice.vercel.app",
         "http://127.0.0.1:5500",
         "http://localhost:5500",
         "https://sridharinternetservice.onrender.com"
     ],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Authorization"],
     methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])

try:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri=REDIS_URL,
        default_limits=["500 per day", "100 per hour"],
        storage_options={"socket_connect_timeout": 30},
        headers_enabled=True
    )
    logger.info("Rate limiter initialized with Redis")
except Exception as e:
    logger.warning(f"Redis connection failed, using in-memory rate limiting: {e}")
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["500 per day", "100 per hour"]
    )

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = generate_password_hash('admin123')

def generate_token(username):
    payload = {
        'username': username,
        'is_admin': True,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = db.Column(db.String(300), nullable=False, index=True)
    slug = db.Column(db.String(350), unique=True, nullable=False, index=True)
    company = db.Column(db.String(200), nullable=False, index=True)
    location = db.Column(db.String(200), index=True)
    job_type = db.Column(db.String(50), index=True)
    sub_category = db.Column(db.String(50), index=True)
    salary_min = db.Column(db.Numeric(12, 2), nullable=True)
    salary_max = db.Column(db.Numeric(12, 2), nullable=True)
    experience_min = db.Column(db.Integer, nullable=True)
    experience_max = db.Column(db.Integer, nullable=True)
    education = db.Column(db.String(200), nullable=True)
    skills = db.Column(db.JSON, nullable=True)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=True)
    responsibilities = db.Column(db.Text, nullable=True)
    benefits = db.Column(db.Text, nullable=True)
    apply_url = db.Column(db.String(500), nullable=True)
    apply_email = db.Column(db.String(200), nullable=True)
    company_logo = db.Column(db.String(500), nullable=True)
    company_website = db.Column(db.String(300), nullable=True)
    external_id = db.Column(db.String(200), unique=True, index=True)
    source = db.Column(db.String(50), default='manual', index=True)
    is_approved = db.Column(db.Boolean, default=False, index=True)
    is_active = db.Column(db.Boolean, default=True, index=True)
    is_remote = db.Column(db.Boolean, default=False, index=True)
    is_featured = db.Column(db.Boolean, default=False, index=True)
    views_count = db.Column(db.Integer, default=0)
    application_deadline = db.Column(db.DateTime, nullable=True)
    expires_at = db.Column(db.DateTime, index=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    @property
    def salary(self):
        if self.salary_min and self.salary_max:
            return f"₹{self.salary_min:,.0f} - ₹{self.salary_max:,.0f}"
        elif self.salary_min:
            return f"₹{self.salary_min:,.0f}+"
        elif self.salary_max:
            return f"Up to ₹{self.salary_max:,.0f}"
        return None

    def to_dict(self, detail=False):
        data = {
            'id': str(self.id),
            'title': self.title,
            'slug': self.slug,
            'company': self.company,
            'location': self.location,
            'job_type': self.job_type,
            'job_category': self.sub_category,
            'salary': self.salary,
            'is_remote': self.is_remote,
            'is_featured': self.is_featured,
            'company_logo': self.company_logo,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'application_deadline': self.application_deadline.isoformat() if self.application_deadline else None
        }
        if detail:
            data.update({
                'description': self.description,
                'requirements': self.requirements,
                'responsibilities': self.responsibilities,
                'benefits': self.benefits,
                'skills': self.skills or [],
                'education': self.education,
                'experience_min': self.experience_min,
                'experience_max': self.experience_max,
                'salary_min': float(self.salary_min) if self.salary_min else None,
                'salary_max': float(self.salary_max) if self.salary_max else None,
                'apply_url': self.apply_url,
                'apply_email': self.apply_email,
                'company_website': self.company_website,
                'source': self.source,
                'is_approved': self.is_approved,
                'external_id': self.external_id,
                'views_count': self.views_count,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            })
        return data

    def increment_views(self):
        self.views_count = (self.views_count or 0) + 1
        try:
            db.session.commit()
        except:
            db.session.rollback()

class JobAnalytics(db.Model):
    __tablename__ = 'job_analytics'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = db.Column(UUID(as_uuid=True), db.ForeignKey('jobs.id'), nullable=False, index=True)
    event_type = db.Column(db.String(50), nullable=False, index=True)
    user_ip = db.Column(db.String(50))
    user_agent = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

class JobSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    title = fields.Str(required=True, validate=validate.Length(min=3, max=300))
    company = fields.Str(required=True, validate=validate.Length(min=2, max=200))
    location = fields.Str(validate=validate.Length(max=200))
    job_type = fields.Str(validate=validate.OneOf([
        'full-time', 'part-time', 'contract', 'freelance', 'internship', 'temporary'
    ]))
    job_category = fields.Str(validate=validate.OneOf([
        'government', 'private', 'mnc', 'startup', 'public-sector', 'other'
    ]))
    salary = fields.Str(validate=validate.Length(max=100))
    salary_min = fields.Decimal(as_string=False, places=2)
    salary_max = fields.Decimal(as_string=False, places=2)
    experience_min = fields.Int(validate=validate.Range(min=0, max=50))
    experience_max = fields.Int(validate=validate.Range(min=0, max=50))
    education = fields.Str(validate=validate.Length(max=200))
    skills = fields.List(fields.Str())
    description = fields.Str(required=True, validate=validate.Length(min=10))
    requirements = fields.Str()
    responsibilities = fields.Str()
    benefits = fields.Str()
    apply_url = fields.Url()
    apply_email = fields.Email()
    company_website = fields.Url()
    is_remote = fields.Bool()
    is_featured = fields.Bool()
    application_deadline = fields.DateTime()
    expires_at = fields.DateTime()

def generate_slug(text: str, max_length: int = 250) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:max_length]

def sanitize_html(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    return text

def categorize_company(company_name: str) -> str:
    if not company_name:
        return 'other'
    
    company_lower = company_name.lower()
    
    govt_keywords = ['government', 'ministry', 'department', 'public service', 'commission', 
                     'psc', 'upsc', 'ssc', 'railway', 'airport authority', 'municipality',
                     'corporation', 'board', 'authority', 'council']
    
    mnc_companies = ['google', 'microsoft', 'amazon', 'facebook', 'meta', 'apple', 
                     'ibm', 'oracle', 'sap', 'adobe', 'salesforce', 'intel', 'cisco',
                     'accenture', 'deloitte', 'cognizant', 'infosys', 'tcs', 'wipro',
                     'hcl', 'tech mahindra', 'capgemini', 'dxc', 'ey', 'pwc', 'kpmg']
    
    if any(keyword in company_lower for keyword in govt_keywords):
        return 'government'
    
    if any(mnc in company_lower for mnc in mnc_companies):
        return 'mnc'
    
    if 'limited' in company_lower or 'ltd' in company_lower:
        return 'private'
    
    return 'other'

def paginate(query, page: int, per_page: int = 20):
    try:
        total = query.count()
        items = query.limit(per_page).offset((page - 1) * per_page).all()
        return {
            'items': items,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page if total > 0 else 0
        }
    except Exception as e:
        logger.error(f"Pagination error: {e}")
        return {
            'items': [],
            'total': 0,
            'page': page,
            'per_page': per_page,
            'pages': 0
        }

def require_admin_login(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            logger.warning(f"Missing Authorization header from {get_remote_address()}")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authorization header required',
                'status': 401
            }), 401
        
        try:
            token = auth_header.replace('Bearer ', '')
            payload = verify_token(token)
            
            if not payload or not payload.get('is_admin'):
                logger.warning(f"Invalid token from {get_remote_address()}")
                return jsonify({
                    'error': 'Unauthorized',
                    'message': 'Invalid or expired token',
                    'status': 401
                }), 401
            
            g.current_user = payload
            g.is_admin = True
            return f(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authentication failed',
                'status': 401
            }), 401
            
    return decorated_function

def track_analytics(event_type: str):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            result = f(*args, **kwargs)
            
            if 'slug' in kwargs and event_type in ['view', 'click']:
                try:
                    job = Job.query.filter_by(slug=kwargs['slug']).first()
                    if job:
                        analytics = JobAnalytics(
                            job_id=job.id,
                            event_type=event_type,
                            user_ip=get_remote_address(),
                            user_agent=request.headers.get('User-Agent', '')[:500]
                        )
                        db.session.add(analytics)
                        db.session.commit()
                except Exception as e:
                    logger.error(f"Analytics tracking error: {e}")
                    db.session.rollback()
            
            return result
        return decorated_function
    return decorator

@app.after_request
def after_request(response):
    if request.path.startswith('/api/'):
        origin = request.headers.get('Origin')
        if origin in ["https://sridharinternetservice.vercel.app", "http://127.0.0.1:5500", "http://localhost:5500", "https://sridharinternetservice.onrender.com"]:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
            response.headers['Access-Control-Expose-Headers'] = 'Authorization'
    return response

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'error': 'Bad Request',
        'message': str(e),
        'status': 400
    }), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found',
        'status': 404
    }), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Too Many Requests',
        'message': 'Rate limit exceeded. Please try again later.',
        'status': 429
    }), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}", exc_info=True)
    db.session.rollback()
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status': 500
    }), 500

@app.errorhandler(ValidationError)
def validation_error(e):
    return jsonify({
        'error': 'Validation Error',
        'messages': e.messages,
        'status': 422
    }), 422

@app.errorhandler(OperationalError)
def database_error(e):
    logger.error(f"Database error: {e}", exc_info=True)
    db.session.rollback()
    return jsonify({
        'error': 'Database Error',
        'message': 'Database operation failed. Please try again.',
        'status': 503
    }), 503

@app.route('/api/admin/login', methods=['POST'])
@limiter.limit("5 per minute")
def admin_login():
    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'JSON data required'
            }), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Username and password required'
            }), 400
        
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            token = generate_token(username)
            
            logger.info(f"Admin login successful from {get_remote_address()}")
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'token': token,
                'admin': {
                    'username': username,
                    'login_time': datetime.utcnow().isoformat()
                }
            }), 200
        else:
            logger.warning(f"Failed admin login attempt from {get_remote_address()} with username: {username}")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logger.error(f"Error during admin login: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Login failed'
        }), 500

@app.route('/api/admin/logout', methods=['POST'])
@require_admin_login
def admin_logout():
    try:
        username = g.current_user.get('username')
        logger.info(f"Admin logout: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        logger.error(f"Error during admin logout: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Logout failed'
        }), 500

@app.route('/api/admin/profile', methods=['GET'])
@require_admin_login
def admin_profile():
    try:
        return jsonify({
            'success': True,
            'admin': {
                'username': g.current_user.get('username'),
                'is_admin': g.current_user.get('is_admin'),
                'token_expires': datetime.fromtimestamp(g.current_user.get('exp')).isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching admin profile: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to fetch profile'
        }), 500

@app.route('/api/health', methods=['GET'])
@limiter.exempt
def health_check():
    health_status = {
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'components': {}
    }
    
    try:
        db.session.execute(text('SELECT 1'))
        health_status['components']['database'] = 'healthy'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status['components']['database'] = 'unhealthy'
        health_status['status'] = 'degraded'
    
    try:
        if hasattr(limiter, 'storage'):
            limiter.storage.ping()
            health_status['components']['redis'] = 'healthy'
        else:
            health_status['components']['redis'] = 'unavailable'
    except:
        health_status['components']['redis'] = 'unavailable'
    
    try:
        cloudinary.api.ping()
        health_status['components']['cloudinary'] = 'healthy'
    except:
        health_status['components']['cloudinary'] = 'unavailable'
    
    status_code = 200 if health_status['status'] == 'ok' else 503
    return jsonify(health_status), status_code

@app.route('/api/health/detailed', methods=['GET'])
@limiter.limit("10 per minute")
@require_admin_login
def detailed_health_check():
    try:
        metrics = {
            'database': {
                'total_jobs': Job.query.count(),
                'active_jobs': Job.query.filter_by(is_active=True).count(),
                'approved_jobs': Job.query.filter_by(is_approved=True).count(),
                'pending_jobs': Job.query.filter_by(is_approved=False, is_active=True).count(),
            },
            'jobs_by_category': dict(
                db.session.query(Job.sub_category, func.count(Job.id))
                .filter(Job.is_active == True, Job.sub_category.isnot(None))
                .group_by(Job.sub_category)
                .all()
            ),
            'timestamp': datetime.utcnow().isoformat()
        }
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({'error': 'Health check failed', 'message': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
@limiter.limit("200 per hour")
def get_jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        is_approved = request.args.get('is_approved', type=str)
        job_type = request.args.get('job_type', type=str)
        job_category = request.args.get('job_category', type=str)
        location = request.args.get('location', type=str)
        is_remote = request.args.get('is_remote', type=str)
        is_featured = request.args.get('is_featured', type=str)
        search = request.args.get('search', type=str)
        company = request.args.get('company', type=str)
        source = request.args.get('source', type=str)
        
        is_admin = hasattr(g, 'is_admin') and g.is_admin
        
        query = Job.query
        
        if not is_admin:
            query = query.filter_by(is_approved=True, is_active=True)
        elif is_approved is not None:
            query = query.filter_by(is_approved=is_approved.lower() in ['true', '1'])
        
        if job_type:
            query = query.filter_by(job_type=job_type)
        
        if job_category:
            query = query.filter_by(sub_category=job_category)
        
        if location:
            query = query.filter(Job.location.ilike(f"%{location}%"))
        
        if company:
            query = query.filter(Job.company.ilike(f"%{company}%"))
        
        if source:
            query = query.filter_by(source=source)
        
        if is_remote is not None and is_remote.lower() in ['true', '1']:
            query = query.filter_by(is_remote=True)
        
        if is_featured is not None and is_featured.lower() in ['true', '1']:
            query = query.filter_by(is_featured=True)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(or_(
                Job.title.ilike(search_term),
                Job.company.ilike(search_term),
                Job.description.ilike(search_term),
                Job.location.ilike(search_term)
            ))
        
        query = query.order_by(Job.is_featured.desc(), Job.created_at.desc())
        
        result = paginate(query, page, per_page)
        
        response = jsonify({
            'success': True,
            'jobs': [j.to_dict() for j in result['items']],
            'pagination': {
                'page': result['page'],
                'per_page': result['per_page'],
                'total': result['total'],
                'pages': result['pages']
            }
        })
        
        response.headers['X-Total-Count'] = str(result['total'])
        return response, 200
        
    except Exception as e:
        logger.error(f"Error fetching jobs: {e}")
        return jsonify({'error': 'Failed to fetch jobs', 'message': str(e)}), 500

@app.route('/api/jobs/<slug>', methods=['GET'])
@limiter.limit("200 per hour")
@track_analytics('view')
def get_job(slug):
    try:
        is_admin = hasattr(g, 'is_admin') and g.is_admin
        
        query = Job.query.filter_by(slug=slug)
        if not is_admin:
            query = query.filter_by(is_approved=True, is_active=True)
        
        job = query.first()
        if not job:
            return jsonify({
                'error': 'Not Found',
                'message': 'Job not found',
                'status': 404
            }), 404
        
        job.increment_views()
        
        return jsonify({
            'success': True,
            'job': job.to_dict(detail=True)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching job {slug}: {e}")
        return jsonify({'error': 'Failed to fetch job', 'message': str(e)}), 500

@app.route('/api/jobs', methods=['POST'])
@limiter.limit("30 per hour")
@require_admin_login
def create_job():
    try:
        schema = JobSchema()
        data = schema.load(request.json)
        
        data['title'] = sanitize_html(data['title'])
        data['company'] = sanitize_html(data['company'])
        data['description'] = sanitize_html(data['description'])
        
        if data.get('requirements'):
            data['requirements'] = sanitize_html(data['requirements'])
        if data.get('responsibilities'):
            data['responsibilities'] = sanitize_html(data['responsibilities'])
        if data.get('benefits'):
            data['benefits'] = sanitize_html(data['benefits'])
        
        sub_category = data.pop('job_category', None)
        if not sub_category:
            sub_category = categorize_company(data['company'])
        
        slug = generate_slug(f"{data['title']}-{data['company']}")
        base_slug = slug
        counter = 1
        while Job.query.filter_by(slug=slug).first():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        job = Job(
            slug=slug,
            sub_category=sub_category,
            is_approved=True,
            source='manual',
            **data
        )
        
        db.session.add(job)
        db.session.commit()
        
        logger.info(f"Job created: {job.id} - {job.title} at {job.company}")
        
        return jsonify({
            'success': True,
            'message': 'Job created successfully',
            'job': job.to_dict(detail=True)
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'messages': e.messages}), 422
    except IntegrityError as e:
        db.session.rollback()
        logger.error(f"Integrity error creating job: {e}")
        return jsonify({'error': 'Conflict', 'message': 'Job already exists'}), 409
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating job: {e}")
        return jsonify({'error': 'Failed to create job', 'message': str(e)}), 500

@app.route('/api/jobs/<slug>', methods=['PUT', 'PATCH'])
@limiter.limit("30 per hour")
@require_admin_login
def update_job(slug):
    try:
        job = Job.query.filter_by(slug=slug).first()
        if not job:
            return jsonify({'error': 'Not Found', 'message': 'Job not found'}), 404
        
        schema = JobSchema(partial=True)
        data = schema.load(request.json)
        
        if 'title' in data or 'company' in data:
            title = sanitize_html(data.get('title', job.title))
            company = sanitize_html(data.get('company', job.company))
            
            new_slug = generate_slug(f"{title}-{company}")
            if new_slug != job.slug:
                base_slug = new_slug
                counter = 1
                while Job.query.filter(Job.slug == new_slug, Job.id != job.id).first():
                    new_slug = f"{base_slug}-{counter}"
                    counter += 1
                job.slug = new_slug
            
            if 'title' in data:
                data['title'] = title
            if 'company' in data:
                data['company'] = company
        
        if 'description' in data:
            data['description'] = sanitize_html(data['description'])
        if 'requirements' in data:
            data['requirements'] = sanitize_html(data['requirements'])
        if 'responsibilities' in data:
            data['responsibilities'] = sanitize_html(data['responsibilities'])
        if 'benefits' in data:
            data['benefits'] = sanitize_html(data['benefits'])
        
        if 'job_category' in data:
            job.sub_category = data.pop('job_category')
        
        for key, value in data.items():
            if key != 'slug':
                setattr(job, key, value)
        
        job.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Job updated: {job.id} - {job.title}")
        
        return jsonify({
            'success': True,
            'message': 'Job updated successfully',
            'job': job.to_dict(detail=True)
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': 'Validation Error', 'messages': e.messages}), 422
    except IntegrityError as e:
        db.session.rollback()
        logger.error(f"Integrity error updating job: {e}")
        return jsonify({'error': 'Conflict', 'message': 'Job slug already exists'}), 409
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating job: {e}")
        return jsonify({'error': 'Failed to update job', 'message': str(e)}), 500

@app.route('/api/jobs/<slug>', methods=['DELETE'])
@limiter.limit("20 per hour")
@require_admin_login
def delete_job(slug):
    try:
        job = Job.query.filter_by(slug=slug).first()
        if not job:
            return jsonify({'error': 'Not Found', 'message': 'Job not found'}), 404
        
        job_title = job.title
        db.session.delete(job)
        db.session.commit()
        
        logger.info(f"Job deleted: {job_title}")
        
        return jsonify({
            'success': True,
            'message': 'Job deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting job: {e}")
        return jsonify({'error': 'Failed to delete job', 'message': str(e)}), 500

@app.route('/api/jobs/<slug>/approve', methods=['POST'])
@limiter.limit("100 per hour")
@require_admin_login
def approve_job(slug):
    try:
        job = Job.query.filter_by(slug=slug).first()
        if not job:
            return jsonify({'error': 'Not Found', 'message': 'Job not found'}), 404
        
        job.is_approved = True
        job.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Job approved: {job.id} - {job.title}")
        
        return jsonify({
            'success': True,
            'message': 'Job approved successfully',
            'job': job.to_dict(detail=True)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error approving job: {e}")
        return jsonify({'error': 'Failed to approve job', 'message': str(e)}), 500

@app.route('/api/jobs/<slug>/reject', methods=['POST'])
@limiter.limit("100 per hour")
@require_admin_login
def reject_job(slug):
    try:
        job = Job.query.filter_by(slug=slug).first()
        if not job:
            return jsonify({'error': 'Not Found', 'message': 'Job not found'}), 404
        
        job.is_approved = False
        job.is_active = False
        job.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Job rejected: {job.id} - {job.title}")
        
        return jsonify({
            'success': True,
            'message': 'Job rejected successfully',
            'job': job.to_dict(detail=True)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error rejecting job: {e}")
        return jsonify({'error': 'Failed to reject job', 'message': str(e)}), 500

@app.route('/api/admin/jobs/pending', methods=['GET'])
@limiter.limit("200 per hour")
@require_admin_login
def get_pending_jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)
        source = request.args.get('source', type=str)
        job_category = request.args.get('job_category', type=str)
        
        query = Job.query.filter_by(is_approved=False, is_active=True)
        
        if source:
            query = query.filter_by(source=source)
        
        if job_category:
            query = query.filter_by(sub_category=job_category)
        
        query = query.order_by(Job.created_at.desc())
        
        result = paginate(query, page, per_page)
        
        stats = {
            'total_pending': result['total'],
            'by_source': {},
            'by_category': {}
        }
        
        source_stats = db.session.query(
            Job.source, func.count(Job.id)
        ).filter_by(
            is_approved=False, is_active=True
        ).group_by(Job.source).all()
        stats['by_source'] = dict(source_stats)
        
        category_stats = db.session.query(
            Job.sub_category, func.count(Job.id)
        ).filter(
            Job.is_approved == False,
            Job.is_active == True,
            Job.sub_category.isnot(None)
        ).group_by(Job.sub_category).all()
        stats['by_category'] = dict(category_stats)
        
        return jsonify({
            'success': True,
            'jobs': [j.to_dict(detail=True) for j in result['items']],
            'pagination': {
                'page': result['page'],
                'per_page': result['per_page'],
                'total': result['total'],
                'pages': result['pages']
            },
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching pending jobs: {e}")
        return jsonify({'error': 'Failed to fetch pending jobs', 'message': str(e)}), 500

@app.route('/api/admin/jobs/bulk-approve', methods=['POST'])
@limiter.limit("30 per hour")
@require_admin_login
def bulk_approve_jobs():
    try:
        data = request.json
        job_ids = data.get('job_ids', [])
        
        if not job_ids or not isinstance(job_ids, list):
            return jsonify({
                'error': 'Bad Request',
                'message': 'job_ids must be a non-empty array'
            }), 400
        
        if len(job_ids) > 100:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Cannot approve more than 100 jobs at once'
            }), 400
        
        try:
            job_uuids = [uuid.UUID(job_id) if isinstance(job_id, str) else job_id for job_id in job_ids]
        except ValueError:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid UUID format in job_ids'
            }), 400
        
        jobs = Job.query.filter(Job.id.in_(job_uuids)).all()
        
        if not jobs:
            return jsonify({
                'error': 'Not Found',
                'message': 'No jobs found with provided IDs'
            }), 404
        
        approved_count = 0
        for job in jobs:
            job.is_approved = True
            job.updated_at = datetime.utcnow()
            approved_count += 1
        
        db.session.commit()
        logger.info(f"Bulk approved {approved_count} jobs")
        
        return jsonify({
            'success': True,
            'message': f'Successfully approved {approved_count} jobs',
            'approved_count': approved_count,
            'job_ids': [str(j.id) for j in jobs]
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error bulk approving jobs: {e}")
        return jsonify({'error': 'Failed to approve jobs', 'message': str(e)}), 500

@app.route('/api/admin/jobs/bulk-reject', methods=['POST'])
@limiter.limit("30 per hour")
@require_admin_login
def bulk_reject_jobs():
    try:
        data = request.json
        job_ids = data.get('job_ids', [])
        
        if not job_ids or not isinstance(job_ids, list):
            return jsonify({
                'error': 'Bad Request',
                'message': 'job_ids must be a non-empty array'
            }), 400
        
        if len(job_ids) > 100:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Cannot reject more than 100 jobs at once'
            }), 400
        
        try:
            job_uuids = [uuid.UUID(job_id) if isinstance(job_id, str) else job_id for job_id in job_ids]
        except ValueError:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid UUID format in job_ids'
            }), 400
        
        jobs = Job.query.filter(Job.id.in_(job_uuids)).all()
        
        if not jobs:
            return jsonify({
                'error': 'Not Found',
                'message': 'No jobs found with provided IDs'
            }), 404
        
        rejected_count = 0
        for job in jobs:
            job.is_approved = False
            job.is_active = False
            job.updated_at = datetime.utcnow()
            rejected_count += 1
        
        db.session.commit()
        logger.info(f"Bulk rejected {rejected_count} jobs")
        
        return jsonify({
            'success': True,
            'message': f'Successfully rejected {rejected_count} jobs',
            'rejected_count': rejected_count,
            'job_ids': [str(j.id) for j in jobs]
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error bulk rejecting jobs: {e}")
        return jsonify({'error': 'Failed to reject jobs', 'message': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@limiter.limit("200 per hour")
@require_admin_login
def get_admin_stats():
    try:
        days = request.args.get('days', 30, type=int)
        since_date = datetime.utcnow() - timedelta(days=days)
        
        stats = {
            'jobs': {
                'total': Job.query.count(),
                'approved': Job.query.filter_by(is_approved=True).count(),
                'pending': Job.query.filter_by(is_approved=False, is_active=True).count(),
                'active': Job.query.filter_by(is_active=True, is_approved=True).count(),
                'featured': Job.query.filter_by(is_featured=True, is_active=True).count(),
                'remote': Job.query.filter_by(is_remote=True, is_active=True, is_approved=True).count(),
                'recent': Job.query.filter(Job.created_at >= since_date).count(),
                'by_source': dict(
                    db.session.query(Job.source, func.count(Job.id))
                    .group_by(Job.source).all()
                ),
                'by_category': dict(
                    db.session.query(Job.sub_category, func.count(Job.id))
                    .filter(Job.sub_category.isnot(None))
                    .group_by(Job.sub_category).all()
                ),
                'by_type': dict(
                    db.session.query(Job.job_type, func.count(Job.id))
                    .filter(Job.job_type.isnot(None))
                    .group_by(Job.job_type).all()
                )
            },
            'analytics': {
                'total_views': db.session.query(func.sum(Job.views_count)).scalar() or 0,
                'recent_views': db.session.query(func.count(JobAnalytics.id))
                    .filter(JobAnalytics.created_at >= since_date)
                    .scalar() or 0
            },
            'recent_jobs': [
                j.to_dict() for j in Job.query
                .filter_by(is_approved=False, is_active=True)
                .order_by(Job.created_at.desc())
                .limit(10).all()
            ],
            'top_viewed_jobs': [
                j.to_dict() for j in Job.query
                .filter_by(is_active=True, is_approved=True)
                .order_by(Job.views_count.desc())
                .limit(10).all()
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return jsonify({'error': 'Failed to get statistics', 'message': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@limiter.limit("20 per hour")
@require_admin_login
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Bad Request', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Bad Request', 'message': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Invalid file type. Allowed: {", ".join(app.config["UPLOAD_EXTENSIONS"])}'
            }), 400
        
        folder = request.form.get('folder', 'sridhar-services')
        
        result = cloudinary.uploader.upload(
            file,
            folder=folder,
            transformation=[
                {'width': 1200, 'height': 800, 'crop': 'limit'},
                {'quality': 'auto:good'},
                {'fetch_format': 'auto'}
            ],
            resource_type='auto'
        )
        
        logger.info(f"Image uploaded to Cloudinary: {result['public_id']}")
        
        return jsonify({
            'success': True,
            'message': 'Image uploaded successfully',
            'url': result['secure_url'],
            'public_id': result['public_id'],
            'width': result.get('width'),
            'height': result.get('height'),
            'format': result.get('format'),
            'size': result.get('bytes')
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {e}")
        return jsonify({'error': 'Failed to upload image', 'message': str(e)}), 500

def fetch_jobs_from_remotive():
    try:
        session = requests_retry_session()
        response = session.get('https://remotive.com/api/remote-jobs', timeout=30)
        response.raise_for_status()
        data = response.json()
        
        jobs_added = 0
        for job_data in data.get('jobs', [])[:20]:
            external_id = f"remotive_{job_data.get('id')}"
            
            if Job.query.filter_by(external_id=external_id).first():
                continue
            
            title = job_data.get('title', 'Untitled Position')
            company = job_data.get('company_name', 'Company')
            location = job_data.get('candidate_required_location', 'Remote')
            
            if not any(term in location.lower() for term in ['india', 'worldwide', 'anywhere', 'remote']):
                continue
            
            slug = generate_slug(f"{title}-{company}-{job_data.get('id')}")
            base_slug = slug
            counter = 1
            while Job.query.filter_by(slug=slug).first():
                slug = f"{base_slug}-{counter}"
                counter += 1
            
            job_category = categorize_company(company)
            
            job = Job(
                title=title[:300],
                slug=slug,
                company=company[:200],
                location=location[:200],
                job_type=job_data.get('job_type', 'full-time').lower(),
                sub_category=job_category,
                description=job_data.get('description', 'No description available')[:10000],
                apply_url=job_data.get('url', ''),
                company_logo=job_data.get('company_logo', ''),
                external_id=external_id,
                source='remotive',
                is_approved=False,
                is_active=True,
                is_remote=True,
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            
            db.session.add(job)
            jobs_added += 1
        
        if jobs_added > 0:
            db.session.commit()
            logger.info(f"Added {jobs_added} jobs from Remotive API")
        
        return jobs_added
    except Exception as e:
        logger.error(f"Error fetching jobs from Remotive: {e}")
        db.session.rollback()
        return 0

def sync_external_jobs():
    with app.app_context():
        logger.info("Starting external job sync...")
        total_added = 0
        
        try:
            count = fetch_jobs_from_remotive()
            total_added += count
            logger.info(f"Fetched {count} jobs from Remotive")
            
            logger.info(f"External job sync completed. Total jobs added: {total_added}")
            
            cleanup_old_unapproved_jobs()
            
        except Exception as e:
            logger.error(f"Critical error in job sync: {e}")
        
        return total_added

def cleanup_expired_jobs():
    with app.app_context():
        try:
            expired_jobs = Job.query.filter(
                Job.expires_at < datetime.utcnow(),
                Job.is_active == True
            ).all()
            
            for job in expired_jobs:
                job.is_active = False
                job.updated_at = datetime.utcnow()
            
            if expired_jobs:
                db.session.commit()
                logger.info(f"Deactivated {len(expired_jobs)} expired jobs")
        except Exception as e:
            logger.error(f"Error cleaning up expired jobs: {e}")
            db.session.rollback()

def cleanup_old_unapproved_jobs():
    with app.app_context():
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            old_jobs = Job.query.filter(
                Job.is_approved == False,
                Job.created_at < cutoff_date,
                Job.source != 'manual'
            ).all()
            
            for job in old_jobs:
                db.session.delete(job)
            
            if old_jobs:
                db.session.commit()
                logger.info(f"Deleted {len(old_jobs)} old unapproved jobs")
        except Exception as e:
            logger.error(f"Error cleaning up old unapproved jobs: {e}")
            db.session.rollback()

def update_job_stats():
    with app.app_context():
        try:
            top_viewed = Job.query.filter_by(
                is_active=True,
                is_approved=True
            ).order_by(Job.views_count.desc()).limit(10).all()
            
            Job.query.update({'is_featured': False})
            
            for job in top_viewed:
                job.is_featured = True
            
            db.session.commit()
            logger.info("Updated featured jobs based on view count")
            
        except Exception as e:
            logger.error(f"Error updating job stats: {e}")
            db.session.rollback()

scheduler = BackgroundScheduler(timezone='Asia/Kolkata')

scheduler.add_job(
    func=sync_external_jobs,
    trigger=CronTrigger(hour=6, minute=0),
    id='sync_external_jobs_morning',
    name='Sync external jobs (morning)',
    replace_existing=True
)

scheduler.add_job(
    func=sync_external_jobs,
    trigger=CronTrigger(hour=18, minute=0),
    id='sync_external_jobs_evening',
    name='Sync external jobs (evening)',
    replace_existing=True
)

scheduler.add_job(
    func=cleanup_expired_jobs,
    trigger=CronTrigger(hour=0, minute=0),
    id='cleanup_expired_jobs',
    name='Cleanup expired jobs',
    replace_existing=True
)

scheduler.add_job(
    func=update_job_stats,
    trigger=CronTrigger(day_of_week='mon', hour=3, minute=0),
    id='update_job_stats',
    name='Update job statistics',
    replace_existing=True
)

@app.route('/api/admin/sync-jobs', methods=['POST'])
@limiter.limit("10 per hour")
@require_admin_login
def trigger_job_sync():
    try:
        count = sync_external_jobs()
        message = f'Synced {count} jobs from external sources'
        
        logger.info(f"Manual job sync triggered: {message}")
        
        return jsonify({
            'success': True,
            'message': message,
            'jobs_added': count
        }), 200
        
    except Exception as e:
        logger.error(f"Error in manual job sync: {e}")
        return jsonify({'error': 'Failed to sync jobs', 'message': str(e)}), 500

@app.route('/api/admin/cleanup', methods=['POST'])
@limiter.limit("10 per hour")
@require_admin_login
def trigger_cleanup():
    try:
        cleanup_expired_jobs()
        cleanup_old_unapproved_jobs()
        
        return jsonify({
            'success': True,
            'message': 'Cleanup operations completed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}")
        return jsonify({'error': 'Failed to perform cleanup', 'message': str(e)}), 500

@app.route('/api/admin/export', methods=['GET'])
@limiter.limit("10 per hour")
@require_admin_login
def export_data():
    try:
        data_type = request.args.get('type', 'jobs')
        
        if data_type == 'jobs':
            jobs = Job.query.all()
            data = [j.to_dict(detail=True) for j in jobs]
        elif data_type == 'analytics':
            analytics = db.session.query(
                JobAnalytics.job_id,
                func.count(JobAnalytics.id).label('count'),
                JobAnalytics.event_type
            ).group_by(
                JobAnalytics.job_id,
                JobAnalytics.event_type
            ).all()
            
            data = [
                {
                    'job_id': str(a.job_id),
                    'event_type': a.event_type,
                    'count': a.count
                }
                for a in analytics
            ]
        else:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid type. Valid types: jobs, analytics'
            }), 400
        
        return jsonify({
            'success': True,
            'type': data_type,
            'count': len(data),
            'data': data,
            'exported_at': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': 'Failed to export data', 'message': str(e)}), 500

@app.route('/api/admin/clear-database', methods=['POST'])
@limiter.limit("5 per hour")
@require_admin_login
def clear_database():
    try:
        confirmation = request.json.get('confirmation')
        if confirmation != 'DELETE_ALL_DATA':
            return jsonify({
                'error': 'Bad Request',
                'message': 'Confirmation required. Send {"confirmation": "DELETE_ALL_DATA"}'
            }), 400
        
        tables_cleared = []
        
        try:
            JobAnalytics.query.delete()
            tables_cleared.append('job_analytics')
        except Exception as e:
            logger.warning(f"Failed to clear job_analytics: {e}")
        
        try:
            Job.query.delete()
            tables_cleared.append('jobs')
        except Exception as e:
            logger.warning(f"Failed to clear jobs: {e}")
        
        db.session.commit()
        
        logger.info(f"Database cleared. Tables: {tables_cleared}")
        
        return jsonify({
            'success': True,
            'message': f'Database cleared successfully. Tables cleared: {", ".join(tables_cleared)}',
            'tables_cleared': tables_cleared
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error clearing database: {e}")
        return jsonify({'error': 'Failed to clear database', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
@limiter.exempt
def index():
    return jsonify({
        'name': 'Sridhar Internet Services API',
        'version': '2.0.0',
        'description': 'Professional job portal API focused on Indian market',
        'regions': ['Telangana', 'Andhra Pradesh'],
        'authentication': 'JWT token-based authentication',
        'admin_credentials': 'Please use /api/admin/login endpoint',
        'endpoints': {
            'health': {
                'url': '/api/health',
                'methods': ['GET'],
                'description': 'Health check endpoint'
            },
            'admin_auth': {
                'login': '/api/admin/login',
                'logout': '/api/admin/logout',
                'profile': '/api/admin/profile'
            },
            'jobs': {
                'url': '/api/jobs',
                'methods': ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
                'description': 'Job listing endpoints'
            },
            'admin': {
                'pending_jobs': '/api/admin/jobs/pending',
                'bulk_approve': '/api/admin/jobs/bulk-approve',
                'bulk_reject': '/api/admin/jobs/bulk-reject',
                'stats': '/api/admin/stats',
                'sync_jobs': '/api/admin/sync-jobs',
                'cleanup': '/api/admin/cleanup',
                'export': '/api/admin/export',
                'clear_database': '/api/admin/clear-database'
            }
        },
        'job_sources': [
            'manual', 'remotive'
        ],
        'job_categories': [
            'government', 'private', 'mnc', 'startup', 'public-sector'
        ],
        'documentation': 'https://github.com/sridhar-services/api',
        'contact': 'admin@sridharservices.com'
    }), 200

@app.route('/api', methods=['GET'])
@limiter.exempt
def api_info():
    return index()

def add_missing_columns():
    with app.app_context():
        try:
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('jobs')]
            
            missing_columns = []
            
            column_definitions = {
                'experience_min': 'ALTER TABLE jobs ADD COLUMN experience_min INTEGER',
                'experience_max': 'ALTER TABLE jobs ADD COLUMN experience_max INTEGER',
                'education': 'ALTER TABLE jobs ADD COLUMN education VARCHAR(200)',
                'skills': 'ALTER TABLE jobs ADD COLUMN skills JSON',
                'requirements': 'ALTER TABLE jobs ADD COLUMN requirements TEXT',
                'responsibilities': 'ALTER TABLE jobs ADD COLUMN responsibilities TEXT',
                'benefits': 'ALTER TABLE jobs ADD COLUMN benefits TEXT',
                'apply_url': 'ALTER TABLE jobs ADD COLUMN apply_url VARCHAR(500)',
                'apply_email': 'ALTER TABLE jobs ADD COLUMN apply_email VARCHAR(200)',
                'company_logo': 'ALTER TABLE jobs ADD COLUMN company_logo VARCHAR(500)',
                'company_website': 'ALTER TABLE jobs ADD COLUMN company_website VARCHAR(300)',
                'external_id': 'ALTER TABLE jobs ADD COLUMN external_id VARCHAR(200)',
                'source': "ALTER TABLE jobs ADD COLUMN source VARCHAR(50) DEFAULT 'manual'",
                'is_approved': 'ALTER TABLE jobs ADD COLUMN is_approved BOOLEAN DEFAULT FALSE',
                'is_active': 'ALTER TABLE jobs ADD COLUMN is_active BOOLEAN DEFAULT TRUE',
                'is_remote': 'ALTER TABLE jobs ADD COLUMN is_remote BOOLEAN DEFAULT FALSE',
                'is_featured': 'ALTER TABLE jobs ADD COLUMN is_featured BOOLEAN DEFAULT FALSE',
                'views_count': 'ALTER TABLE jobs ADD COLUMN views_count INTEGER DEFAULT 0',
                'application_deadline': 'ALTER TABLE jobs ADD COLUMN application_deadline TIMESTAMP',
                'expires_at': 'ALTER TABLE jobs ADD COLUMN expires_at TIMESTAMP',
                'created_at': 'ALTER TABLE jobs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                'updated_at': 'ALTER TABLE jobs ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
            }
            
            for column_name, alter_statement in column_definitions.items():
                if column_name not in columns:
                    try:
                        db.session.execute(text(alter_statement))
                        missing_columns.append(column_name)
                        logger.info(f"Added column: {column_name}")
                    except Exception as col_error:
                        logger.warning(f"Could not add column {column_name}: {col_error}")
                
            if missing_columns:
                db.session.commit()
                logger.info(f"Successfully added missing columns: {missing_columns}")
            else:
                logger.info("All required columns exist")
                
            index_statements = [
                'CREATE INDEX IF NOT EXISTS idx_jobs_slug ON jobs(slug)',
                'CREATE INDEX IF NOT EXISTS idx_jobs_is_approved ON jobs(is_approved)',
                'CREATE INDEX IF NOT EXISTS idx_jobs_is_active ON jobs(is_active)',
                'CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)',
                'CREATE INDEX IF NOT EXISTS idx_jobs_external_id ON jobs(external_id)',
                'CREATE INDEX IF NOT EXISTS idx_job_analytics_job_id ON job_analytics(job_id)',
                'CREATE INDEX IF NOT EXISTS idx_job_analytics_event_type ON job_analytics(event_type)'
            ]
            
            for index_stmt in index_statements:
                try:
                    db.session.execute(text(index_stmt))
                except Exception as idx_error:
                    logger.debug(f"Index may already exist: {idx_error}")
                    
            db.session.commit()
                
        except Exception as e:
            logger.error(f"Error adding columns: {e}")
            db.session.rollback()

def init_db():
    with app.app_context():
        try:
            db.session.execute(text('SELECT 1'))
            
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            logger.info(f"Existing tables: {existing_tables}")
            
            tables_to_create = []
            if 'jobs' not in existing_tables:
                tables_to_create.append('jobs')
            if 'job_analytics' not in existing_tables:
                tables_to_create.append('job_analytics')
            
            if tables_to_create:
                logger.info(f"Creating tables: {tables_to_create}")
                db.create_all()
                logger.info("Database tables created successfully")
            else:
                logger.info("All required tables already exist")
                add_missing_columns()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            logger.info("Application will continue with existing database structure")

if __name__ == '__main__':
    init_db()
    
    try:
        scheduler.start()
        logger.info("Background scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=debug
    )
else:
    init_db()
    
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Background scheduler started (production mode)")
    except Exception as e:
        logger.error(f"Failed to start scheduler in production: {e}")