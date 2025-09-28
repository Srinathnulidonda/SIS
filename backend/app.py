import os
import logging
import secrets
import asyncio
import aiohttp
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import requests
from urllib.parse import urlparse, urljoin
import re
import time
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET

from flask import Flask, request, jsonify, session, g, abort, current_app, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from flask_caching import Cache
import jwt
import bcrypt
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, Unauthorized, Forbidden, NotFound
import cloudinary
import cloudinary.uploader
import cloudinary.api
from celery import Celery
from celery.schedules import crontab
import redis
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy import and_, or_, desc, asc, func, text, Index
from sqlalchemy.ext.hybrid import hybrid_property
from email_validator import validate_email, EmailNotValidError
import uuid
from bs4 import BeautifulSoup
import feedparser
import requests_cache
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from marshmallow import Schema, fields, validate, ValidationError
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
import bleach
from dateutil import parser as date_parser
import pytz
from flask_talisman import Talisman
from flask_compress import Compress


structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address)
cache = Cache()
jwt_manager = JWTManager()
compress = Compress()

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_JOBS = Gauge('active_jobs_total', 'Total active jobs')
JOB_SYNC_ERRORS = Counter('job_sync_errors_total', 'Job sync errors', ['source'])

blacklisted_tokens = set()

INDIAN_STATES = {
    'telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam', 'Mahbubnagar', 'Medak', 'Nalgonda', 'Rangareddy', 'Adilabad'],
    'andhra_pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Rajahmundry', 'Tirupati', 'Kadapa', 'Anantapur', 'Chittoor', 'Eluru', 'Ongole', 'Srikakulam', 'Vizianagaram']
}

INDIAN_JOB_CATEGORIES = [
    'Government Jobs', 'Banking & Finance', 'Railway Jobs', 'Police & Defence', 'Teaching & Education',
    'Healthcare & Medical', 'IT & Software', 'Engineering', 'Civil Services', 'PSU Jobs',
    'State Government', 'Central Government', 'Court Jobs', 'University Jobs', 'Research & Development',
    'Agriculture & Rural Development', 'Telecommunications', 'Power & Energy', 'Transport',
    'Administrative Services', 'Technical Services', 'Legal Services', 'Accounts & Audit'
]

EDUCATION_QUALIFICATIONS = [
    '10th Pass', '12th Pass', 'ITI', 'Diploma', 'Graduate', 'Post Graduate', 'B.Tech', 'B.E',
    'M.Tech', 'M.E', 'MBA', 'MCA', 'BCA', 'B.Sc', 'M.Sc', 'B.Com', 'M.Com', 'B.A', 'M.A',
    'LLB', 'LLM', 'MBBS', 'MD', 'B.Ed', 'M.Ed', 'Ph.D', 'CA', 'CS', 'CMA', 'Any Degree'
]

@dataclass
class JobAPIConfig:
    name: str
    base_url: str
    api_key_required: bool = False
    rate_limit_per_hour: int = 100
    supports_search: bool = True
    data_format: str = 'json'
    auth_header: str = 'Authorization'
    location_filter: bool = True
    indian_specific: bool = True


class IndianJobDataExtractor:
    @staticmethod
    def extract_sarkari_result(data: Dict) -> Dict:
        return {
            'title': data.get('post_name', data.get('title', '')),
            'company': data.get('department', data.get('organization', 'Government of India')),
            'location': IndianJobDataExtractor._normalize_indian_location(data.get('location', data.get('state', ''))),
            'description': BeautifulSoup(data.get('description', data.get('details', '')), 'html.parser').get_text(),
            'job_type': 'government',
            'application_url': data.get('apply_link', data.get('official_website', '')),
            'external_id': data.get('notification_id', data.get('id', '')),
            'application_deadline': data.get('last_date', data.get('application_end_date', '')),
            'category': IndianJobDataExtractor._categorize_indian_job(data.get('category', data.get('department', ''))),
            'salary_range': data.get('salary', data.get('pay_scale', '')),
            'education_required': data.get('qualification', data.get('eligibility', '')),
            'experience_level': data.get('experience', ''),
            'age_limit': data.get('age_limit', ''),
            'exam_date': data.get('exam_date', ''),
            'notification_pdf': data.get('notification_pdf', ''),
            'language': 'hindi,english,telugu' if 'telangana' in data.get('location', '').lower() or 'andhra' in data.get('location', '').lower() else 'hindi,english'
        }
    
    @staticmethod
    def extract_freshersworld(data: Dict) -> Dict:
        location = IndianJobDataExtractor._normalize_indian_location(data.get('job_location', data.get('location', '')))
        return {
            'title': data.get('job_title', data.get('title', '')),
            'company': data.get('company_name', ''),
            'location': location,
            'description': BeautifulSoup(data.get('job_description', ''), 'html.parser').get_text(),
            'job_type': 'private',
            'application_url': data.get('apply_url', ''),
            'external_id': str(data.get('job_id', '')),
            'salary_range': IndianJobDataExtractor._format_indian_salary(data.get('salary', '')),
            'experience_level': data.get('experience', ''),
            'education_required': data.get('qualification', ''),
            'category': IndianJobDataExtractor._categorize_indian_job(data.get('functional_area', data.get('industry', ''))),
            'skills_required': data.get('key_skills', ''),
            'employment_type': data.get('employment_type', 'Full Time')
        }
    
    @staticmethod
    def extract_naukri(data: Dict) -> Dict:
        location = IndianJobDataExtractor._normalize_indian_location(data.get('placeholders', [{}])[0].get('label', ''))
        return {
            'title': data.get('title', ''),
            'company': data.get('companyName', ''),
            'location': location,
            'description': BeautifulSoup(data.get('jobDescription', ''), 'html.parser').get_text(),
            'job_type': 'private',
            'application_url': f"https://www.naukri.com{data.get('jdURL', '')}",
            'external_id': data.get('jobId', ''),
            'salary_range': IndianJobDataExtractor._format_indian_salary(data.get('packagelabel', '')),
            'experience_level': data.get('experienceLabel', ''),
            'education_required': data.get('education', ''),
            'category': IndianJobDataExtractor._categorize_indian_job(data.get('industry', '')),
            'skills_required': ', '.join(data.get('tagsAndSkills', [])),
            'posted_date': data.get('createdDate', '')
        }
    
    @staticmethod
    def extract_indeed_india(data: Dict) -> Dict:
        location = IndianJobDataExtractor._normalize_indian_location(data.get('formattedLocation', ''))
        return {
            'title': data.get('title', ''),
            'company': data.get('company', ''),
            'location': location,
            'description': BeautifulSoup(data.get('summary', ''), 'html.parser').get_text(),
            'job_type': 'private',
            'application_url': f"https://in.indeed.com/viewjob?jk={data.get('jobkey', '')}",
            'external_id': data.get('jobkey', ''),
            'salary_range': IndianJobDataExtractor._format_indian_salary(data.get('salarySnippet', {}).get('text', '')),
            'posted_date': data.get('date', ''),
            'category': IndianJobDataExtractor._categorize_indian_job(data.get('title', '')),
            'remote_eligible': 'remote' in data.get('title', '').lower() or 'work from home' in data.get('summary', '').lower()
        }
    
    @staticmethod
    def extract_government_jobs(data: Dict) -> Dict:
        return {
            'title': data.get('post_name', data.get('job_title', '')),
            'company': data.get('organization', data.get('department', 'Government of India')),
            'location': IndianJobDataExtractor._normalize_indian_location(data.get('location', data.get('state', ''))),
            'description': BeautifulSoup(data.get('job_details', data.get('description', '')), 'html.parser').get_text(),
            'job_type': 'government',
            'application_url': data.get('apply_online_link', data.get('website', '')),
            'external_id': data.get('notification_no', data.get('advt_no', '')),
            'application_deadline': data.get('last_date_to_apply', ''),
            'category': 'Government Jobs',
            'salary_range': data.get('pay_scale', data.get('salary_range', '')),
            'education_required': data.get('educational_qualification', ''),
            'age_limit': data.get('age_limit', ''),
            'total_vacancies': data.get('total_posts', ''),
            'selection_process': data.get('selection_process', ''),
            'application_fee': data.get('application_fee', ''),
            'important_dates': data.get('important_dates', {})
        }
    
    @staticmethod
    def extract_freejobalert(data: Dict) -> Dict:
        location = IndianJobDataExtractor._normalize_indian_location(data.get('location', ''))
        return {
            'title': data.get('job_title', ''),
            'company': data.get('company_name', data.get('organization', '')),
            'location': location,
            'description': BeautifulSoup(data.get('job_description', ''), 'html.parser').get_text(),
            'job_type': IndianJobDataExtractor._determine_job_type(data.get('job_type', data.get('category', ''))),
            'application_url': data.get('apply_link', ''),
            'external_id': data.get('job_id', ''),
            'application_deadline': data.get('last_date', ''),
            'category': IndianJobDataExtractor._categorize_indian_job(data.get('category', '')),
            'salary_range': IndianJobDataExtractor._format_indian_salary(data.get('salary', '')),
            'education_required': data.get('qualification', ''),
            'experience_level': data.get('experience', ''),
            'state': IndianJobDataExtractor._extract_state_from_location(location)
        }
    
    @staticmethod
    def _normalize_indian_location(location: str) -> str:
        if not location:
            return ''
        
        location = location.strip()
        
        for state, cities in INDIAN_STATES.items():
            for city in cities:
                if city.lower() in location.lower():
                    state_name = 'Telangana' if state == 'telangana' else 'Andhra Pradesh'
                    return f"{city}, {state_name}"
        
        location_mappings = {
            'hyd': 'Hyderabad, Telangana',
            'vizag': 'Visakhapatnam, Andhra Pradesh',
            'vjw': 'Vijayawada, Andhra Pradesh',
            'secondary': 'Secunderabad, Telangana',
            'gnt': 'Guntur, Andhra Pradesh',
            'tpt': 'Tirupati, Andhra Pradesh'
        }
        
        for abbr, full_location in location_mappings.items():
            if abbr in location.lower():
                return full_location
        
        if any(keyword in location.lower() for keyword in ['telangana', 'ts', 't.s']):
            return f"{location}, Telangana"
        elif any(keyword in location.lower() for keyword in ['andhra pradesh', 'andhra', 'ap', 'a.p']):
            return f"{location}, Andhra Pradesh"
        
        return location
    
    @staticmethod
    def _extract_state_from_location(location: str) -> str:
        if 'telangana' in location.lower():
            return 'Telangana'
        elif 'andhra pradesh' in location.lower() or 'andhra' in location.lower():
            return 'Andhra Pradesh'
        return ''
    
    @staticmethod
    def _categorize_indian_job(category_text: str) -> str:
        if not category_text:
            return 'General'
        
        category_text = category_text.lower()
        
        category_mappings = {
            'banking': 'Banking & Finance',
            'railway': 'Railway Jobs',
            'police': 'Police & Defence',
            'defence': 'Police & Defence',
            'teaching': 'Teaching & Education',
            'education': 'Teaching & Education',
            'medical': 'Healthcare & Medical',
            'healthcare': 'Healthcare & Medical',
            'it': 'IT & Software',
            'software': 'IT & Software',
            'engineering': 'Engineering',
            'civil': 'Civil Services',
            'administrative': 'Administrative Services',
            'technical': 'Technical Services',
            'accounts': 'Accounts & Audit',
            'audit': 'Accounts & Audit',
            'legal': 'Legal Services',
            'agriculture': 'Agriculture & Rural Development',
            'power': 'Power & Energy',
            'energy': 'Power & Energy',
            'transport': 'Transport',
            'university': 'University Jobs',
            'research': 'Research & Development'
        }
        
        for keyword, mapped_category in category_mappings.items():
            if keyword in category_text:
                return mapped_category
        
        return 'Government Jobs' if any(word in category_text for word in ['government', 'govt', 'sarkari']) else 'General'
    
    @staticmethod
    def _determine_job_type(job_type_text: str) -> str:
        if not job_type_text:
            return 'private'
        
        job_type_text = job_type_text.lower()
        
        if any(word in job_type_text for word in ['government', 'govt', 'sarkari', 'public sector', 'psu']):
            return 'government'
        elif any(word in job_type_text for word in ['remote', 'work from home', 'wfh']):
            return 'remote'
        else:
            return 'private'
    
    @staticmethod
    def _format_indian_salary(salary_text: str) -> str:
        if not salary_text:
            return ''
        
        salary_text = salary_text.strip()
        
        if 'lpa' in salary_text.lower() or 'per annum' in salary_text.lower():
            return salary_text
        
        numbers = re.findall(r'[\d,]+', salary_text)
        if numbers:
            amount = int(numbers[0].replace(',', ''))
            if amount > 100000:
                return f"₹{amount//100000}.{(amount%100000)//10000} LPA"
            elif amount > 1000:
                return f"₹{amount//1000}K per month"
            else:
                return f"₹{amount} per month"
        
        return salary_text


INDIAN_JOB_APIS = {
    'sarkari_result': JobAPIConfig(
        name='Sarkari Result',
        base_url='https://www.sarkariresult.com/api/latest-jobs',
        supports_search=True,
        rate_limit_per_hour=50,
        indian_specific=True
    ),
    'freshersworld': JobAPIConfig(
        name='FreshersWorld India',
        base_url='https://www.freshersworld.com/api/jobs',
        supports_search=True,
        rate_limit_per_hour=100,
        indian_specific=True
    ),
    'naukri': JobAPIConfig(
        name='Naukri.com',
        base_url='https://www.naukri.com/jobapi/v3/search',
        api_key_required=True,
        rate_limit_per_hour=200,
        indian_specific=True
    ),
    'indeed_india': JobAPIConfig(
        name='Indeed India',
        base_url='https://in.indeed.com/api/jobs',
        supports_search=True,
        rate_limit_per_hour=150,
        indian_specific=True
    ),
    'government_jobs_india': JobAPIConfig(
        name='Government Jobs India',
        base_url='https://www.governmentjobsindia.com/api/jobs',
        supports_search=True,
        rate_limit_per_hour=75,
        indian_specific=True
    ),
    'freejobalert': JobAPIConfig(
        name='Free Job Alert',
        base_url='https://www.freejobalert.com/api/jobs',
        supports_search=True,
        rate_limit_per_hour=80,
        indian_specific=True
    ),
    'employment_news': JobAPIConfig(
        name='Employment News',
        base_url='https://www.employmentnews.gov.in/api/notifications',
        supports_search=True,
        rate_limit_per_hour=60,
        indian_specific=True
    ),
    'ibps': JobAPIConfig(
        name='IBPS Jobs',
        base_url='https://www.ibps.in/api/notifications',
        supports_search=False,
        rate_limit_per_hour=30,
        indian_specific=True
    ),
    'upsc': JobAPIConfig(
        name='UPSC Jobs',
        base_url='https://upsc.gov.in/api/examinations',
        supports_search=False,
        rate_limit_per_hour=20,
        indian_specific=True
    ),
    'ssc': JobAPIConfig(
        name='SSC Jobs',
        base_url='https://ssc.nic.in/api/notifications',
        supports_search=False,
        rate_limit_per_hour=25,
        indian_specific=True
    ),
    'railway_jobs': JobAPIConfig(
        name='Railway Recruitment',
        base_url='https://www.rrcb.gov.in/api/notifications',
        supports_search=False,
        rate_limit_per_hour=40,
        indian_specific=True
    ),
    'tspsc': JobAPIConfig(
        name='TSPSC Jobs',
        base_url='https://www.tspsc.gov.in/api/notifications',
        supports_search=True,
        rate_limit_per_hour=50,
        indian_specific=True
    ),
    'appsc': JobAPIConfig(
        name='APPSC Jobs',
        base_url='https://psc.ap.gov.in/api/notifications',
        supports_search=True,
        rate_limit_per_hour=50,
        indian_specific=True
    )
}


class UserSchema(Schema):
    username = fields.Str(required=True, validate=validate.Length(min=3, max=80))
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8))
    role = fields.Str(validate=validate.OneOf(['admin', 'super_admin']))
    preferred_locations = fields.List(fields.Str(), missing=[])
    preferred_categories = fields.List(fields.Str(), missing=[])
    language_preference = fields.Str(validate=validate.OneOf(['english', 'hindi', 'telugu', 'mixed']), missing='english')


class JobSchema(Schema):
    title = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    company = fields.Str(required=True, validate=validate.Length(min=1, max=150))
    location = fields.Str(validate=validate.Length(max=100))
    job_type = fields.Str(required=True, validate=validate.OneOf(['government', 'private', 'remote']))
    category = fields.Str(required=True, validate=validate.OneOf(INDIAN_JOB_CATEGORIES))
    description = fields.Str()
    requirements = fields.Str()
    salary_range = fields.Str(validate=validate.Length(max=100))
    experience_level = fields.Str(validate=validate.Length(max=50))
    education_required = fields.Str(validate=validate.OneOf(EDUCATION_QUALIFICATIONS + ['Any Degree', 'As per notification']))
    application_url = fields.Url()
    application_deadline = fields.Date()
    age_limit = fields.Str(validate=validate.Length(max=50))
    application_fee = fields.Str(validate=validate.Length(max=100))
    selection_process = fields.Str()
    total_vacancies = fields.Str(validate=validate.Length(max=50))
    state = fields.Str(validate=validate.OneOf(['Telangana', 'Andhra Pradesh', 'All India']))


class ServiceSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=150))
    category = fields.Str(required=True, validate=validate.OneOf([
        'LIC Insurance', 'PAN Card Services', 'Aadhar Services', 'Passport Services',
        'Printing & Xerox', 'Travel & Tickets', 'Bill Payments', 'Banking Services',
        'Loan Services', 'Property Services', 'Legal Services', 'Educational Services',
        'Government Forms', 'Certificates', 'Translation Services', 'Computer Services'
    ]))
    description = fields.Str()
    price_range = fields.Str(validate=validate.Length(max=100))
    contact_info = fields.Dict()
    service_areas = fields.Str(validate=validate.Length(max=200))
    languages_supported = fields.List(fields.Str(validate=validate.OneOf(['English', 'Hindi', 'Telugu'])), missing=['English'])


class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.Enum('super_admin', 'admin', name='user_roles'), nullable=False, default='admin')
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    email_verified = db.Column(db.Boolean, default=False)
    phone_number = db.Column(db.String(15))
    preferred_locations = db.Column(db.ARRAY(db.String(100)), default=list)
    preferred_categories = db.Column(db.ARRAY(db.String(100)), default=list)
    language_preference = db.Column(db.String(20), default='english')
    notification_settings = db.Column(JSONB, default=dict)
    last_login = db.Column(db.DateTime)
    login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    jobs = db.relationship('Job', backref='author', lazy=True, cascade='all, delete-orphan')
    services = db.relationship('Service', backref='author', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password: str):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def is_locked(self) -> bool:
        return self.locked_until and self.locked_until > datetime.utcnow()
    
    def lock_account(self, minutes: int = 30):
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.login_attempts = 0
    
    def reset_login_attempts(self):
        self.login_attempts = 0
        self.locked_until = None
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            'id': str(self.id),
            'username': self.username,
            'email': self.email if include_sensitive else None,
            'role': self.role,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'phone_number': self.phone_number if include_sensitive else None,
            'preferred_locations': self.preferred_locations,
            'preferred_categories': self.preferred_categories,
            'language_preference': self.language_preference,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat()
        }
        return {k: v for k, v in data.items() if v is not None}


class JobSource(db.Model):
    __tablename__ = 'job_sources'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String(100), nullable=False)
    api_identifier = db.Column(db.String(50), nullable=False, unique=True)
    api_url = db.Column(db.String(500))
    api_key = db.Column(db.String(200))
    source_type = db.Column(db.Enum('api', 'manual', 'rss', 'scraper', name='source_types'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    target_states = db.Column(db.ARRAY(db.String(50)), default=['Telangana', 'Andhra Pradesh'])
    priority = db.Column(db.Integer, default=1)
    last_sync = db.Column(db.DateTime)
    last_error = db.Column(db.Text)
    sync_interval_minutes = db.Column(db.Integer, default=60)
    rate_limit_per_hour = db.Column(db.Integer, default=100)
    requests_made_this_hour = db.Column(db.Integer, default=0)
    rate_limit_reset_time = db.Column(db.DateTime, default=datetime.utcnow)
    success_count = db.Column(db.Integer, default=0)
    error_count = db.Column(db.Integer, default=0)
    jobs_imported_today = db.Column(db.Integer, default=0)
    config = db.Column(JSONB, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    jobs = db.relationship('Job', backref='source', lazy=True)
    
    def can_make_request(self) -> bool:
        now = datetime.utcnow()
        if now >= self.rate_limit_reset_time:
            self.requests_made_this_hour = 0
            self.rate_limit_reset_time = now + timedelta(hours=1)
            db.session.commit()
        
        return self.requests_made_this_hour < self.rate_limit_per_hour
    
    def increment_request_count(self):
        self.requests_made_this_hour += 1
        db.session.commit()
    
    def record_success(self, jobs_count: int = 0):
        self.success_count += 1
        self.jobs_imported_today += jobs_count
        self.last_error = None
        self.last_sync = datetime.utcnow()
        db.session.commit()
    
    def record_error(self, error_message: str):
        self.error_count += 1
        self.last_error = error_message
        db.session.commit()


class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = db.Column(db.String(200), nullable=False, index=True)
    company = db.Column(db.String(150), nullable=False, index=True)
    location = db.Column(db.String(100))
    state = db.Column(db.String(50), index=True)
    job_type = db.Column(db.Enum('government', 'private', 'remote', name='job_types'), nullable=False, index=True)
    category = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text)
    requirements = db.Column(db.Text)
    salary_range = db.Column(db.String(100))
    experience_level = db.Column(db.String(50))
    education_required = db.Column(db.String(100))
    age_limit = db.Column(db.String(50))
    application_deadline = db.Column(db.Date, index=True)
    application_url = db.Column(db.String(500))
    application_fee = db.Column(db.String(100))
    selection_process = db.Column(db.Text)
    total_vacancies = db.Column(db.String(50))
    exam_date = db.Column(db.Date)
    result_date = db.Column(db.Date)
    important_dates = db.Column(JSONB, default=dict)
    notification_pdf = db.Column(db.String(500))
    external_id = db.Column(db.String(100), index=True)
    poster_image_url = db.Column(db.String(500))
    status = db.Column(db.Enum('draft', 'published', 'archived', 'moderation', 'expired', name='job_status'), default='published', index=True)
    is_featured = db.Column(db.Boolean, default=False, index=True)
    is_urgent = db.Column(db.Boolean, default=False)
    view_count = db.Column(db.Integer, default=0)
    click_count = db.Column(db.Integer, default=0)
    share_count = db.Column(db.Integer, default=0)
    bookmark_count = db.Column(db.Integer, default=0)
    search_vector = db.Column(TSVECTOR)
    tags = db.Column(db.ARRAY(db.String(50)), default=list)
    remote_eligible = db.Column(db.Boolean, default=False)
    salary_min = db.Column(db.Integer)
    salary_max = db.Column(db.Integer)
    currency = db.Column(db.String(3), default='INR')
    language = db.Column(db.String(50), default='english,hindi')
    scraped_at = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    source_id = db.Column(UUID(as_uuid=True), db.ForeignKey('job_sources.id'))
    
    __table_args__ = (
        Index('ix_job_search', 'search_vector', postgresql_using='gin'),
        Index('ix_job_location_type_state', 'location', 'job_type', 'state'),
        Index('ix_job_category_deadline', 'category', 'application_deadline'),
        Index('ix_job_external_source', 'external_id', 'source_id'),
        Index('ix_job_state_category', 'state', 'category'),
    )
    
    def increment_view_count(self):
        self.view_count += 1
        db.session.commit()
    
    def increment_click_count(self):
        self.click_count += 1
        db.session.commit()
    
    def increment_share_count(self):
        self.share_count += 1
        db.session.commit()
    
    def increment_bookmark_count(self):
        self.bookmark_count += 1
        db.session.commit()
    
    def is_expired(self) -> bool:
        if self.application_deadline:
            return self.application_deadline < datetime.now().date()
        if self.expires_at:
            return self.expires_at < datetime.utcnow()
        return False
    
    def extract_salary_range(self):
        if not self.salary_range:
            return
        
        salary_text = self.salary_range.lower()
        
        if 'lpa' in salary_text or 'per annum' in salary_text:
            numbers = re.findall(r'[\d\.]+', salary_text)
            if len(numbers) >= 2:
                self.salary_min = int(float(numbers[0]) * 100000)
                self.salary_max = int(float(numbers[1]) * 100000)
            elif len(numbers) == 1:
                amount = int(float(numbers[0]) * 100000)
                self.salary_min = amount
                self.salary_max = amount
        else:
            numbers = re.findall(r'\d+', salary_text)
            if len(numbers) >= 2:
                self.salary_min = int(numbers[0])
                self.salary_max = int(numbers[1])
    
    def generate_tags(self):
        text = f"{self.title} {self.description} {self.requirements}".lower()
        
        indian_keywords = [
            'sarkari', 'government', 'psu', 'banking', 'railway', 'ssc', 'upsc', 'ibps',
            'teaching', 'police', 'defence', 'medical', 'engineering', 'clerk', 'officer',
            'assistant', 'junior', 'senior', 'graduate', 'diploma', 'degree', 'hindi',
            'english', 'telugu', 'written exam', 'interview', 'group discussion'
        ]
        
        location_keywords = ['hyderabad', 'vijayawada', 'visakhapatnam', 'guntur', 'warangal',
                           'tirupati', 'nellore', 'kurnool', 'telangana', 'andhra pradesh']
        
        all_keywords = indian_keywords + location_keywords
        
        tags = []
        for keyword in all_keywords:
            if keyword in text and keyword not in tags:
                tags.append(keyword)
        
        if self.job_type == 'government':
            tags.append('sarkari-job')
        
        if self.state:
            tags.append(self.state.lower().replace(' ', '-'))
        
        self.tags = tags[:15]
    
    def to_dict(self, include_stats: bool = False) -> Dict[str, Any]:
        data = {
            'id': str(self.id),
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'state': self.state,
            'job_type': self.job_type,
            'category': self.category,
            'description': self.description,
            'requirements': self.requirements,
            'salary_range': self.salary_range,
            'salary_min': self.salary_min,
            'salary_max': self.salary_max,
            'currency': self.currency,
            'experience_level': self.experience_level,
            'education_required': self.education_required,
            'age_limit': self.age_limit,
            'application_deadline': self.application_deadline.isoformat() if self.application_deadline else None,
            'application_url': self.application_url,
            'application_fee': self.application_fee,
            'selection_process': self.selection_process,
            'total_vacancies': self.total_vacancies,
            'exam_date': self.exam_date.isoformat() if self.exam_date else None,
            'result_date': self.result_date.isoformat() if self.result_date else None,
            'important_dates': self.important_dates,
            'notification_pdf': self.notification_pdf,
            'poster_image_url': self.poster_image_url,
            'status': self.status,
            'is_featured': self.is_featured,
            'is_urgent': self.is_urgent,
            'tags': self.tags,
            'remote_eligible': self.remote_eligible,
            'language': self.language,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author.username,
            'source': self.source.name if self.source else 'Manual'
        }
        if include_stats:
            data.update({
                'view_count': self.view_count,
                'click_count': self.click_count,
                'share_count': self.share_count,
                'bookmark_count': self.bookmark_count,
                'ctr': round((self.click_count / self.view_count * 100), 2) if self.view_count > 0 else 0
            })
        return data


class Service(db.Model):
    __tablename__ = 'services'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String(150), nullable=False, index=True)
    category = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text)
    price_range = db.Column(db.String(100))
    contact_info = db.Column(JSONB, default=dict)
    service_areas = db.Column(db.String(200))
    languages_supported = db.Column(db.ARRAY(db.String(20)), default=['English'])
    office_address = db.Column(db.Text)
    working_hours = db.Column(JSONB, default=dict)
    image_url = db.Column(db.String(500))
    is_active = db.Column(db.Boolean, default=True)
    is_featured = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    view_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    review_count = db.Column(db.Integer, default=0)
    tags = db.Column(db.ARRAY(db.String(50)), default=list)
    availability = db.Column(JSONB, default=dict)
    social_links = db.Column(JSONB, default=dict)
    documents_handled = db.Column(db.ARRAY(db.String(100)), default=list)
    government_authorized = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'), nullable=False)
    
    def increment_view_count(self):
        self.view_count += 1
        db.session.commit()
    
    def to_dict(self, include_stats: bool = False) -> Dict[str, Any]:
        data = {
            'id': str(self.id),
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'price_range': self.price_range,
            'contact_info': self.contact_info,
            'service_areas': self.service_areas,
            'languages_supported': self.languages_supported,
            'office_address': self.office_address,
            'working_hours': self.working_hours,
            'image_url': self.image_url,
            'is_active': self.is_active,
            'is_featured': self.is_featured,
            'is_verified': self.is_verified,
            'rating': self.rating,
            'review_count': self.review_count,
            'tags': self.tags,
            'availability': self.availability,
            'social_links': self.social_links,
            'documents_handled': self.documents_handled,
            'government_authorized': self.government_authorized,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author.username
        }
        if include_stats:
            data['view_count'] = self.view_count
        return data


class Ad(db.Model):
    __tablename__ = 'ads'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = db.Column(db.String(100), nullable=False)
    ad_unit_id = db.Column(db.String(100), nullable=False)
    placement = db.Column(db.Enum('header', 'sidebar', 'footer', 'content', 'mobile_banner', 'interstitial', name='ad_placements'), nullable=False)
    size = db.Column(db.String(50))
    is_active = db.Column(db.Boolean, default=True)
    priority = db.Column(db.Integer, default=1)
    target_locations = db.Column(db.ARRAY(db.String(50)), default=['Telangana', 'Andhra Pradesh'])
    target_categories = db.Column(db.ARRAY(db.String(100)), default=list)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    impression_count = db.Column(db.Integer, default=0)
    click_count = db.Column(db.Integer, default=0)
    revenue = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def is_active_now(self) -> bool:
        now = datetime.utcnow()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return self.is_active
    
    def increment_impression(self):
        self.impression_count += 1
        db.session.commit()
    
    def increment_click(self, revenue: float = 0.0):
        self.click_count += 1
        self.revenue += revenue
        db.session.commit()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'name': self.name,
            'ad_unit_id': self.ad_unit_id,
            'placement': self.placement,
            'size': self.size,
            'is_active': self.is_active,
            'priority': self.priority,
            'target_locations': self.target_locations,
            'target_categories': self.target_categories,
            'impression_count': self.impression_count,
            'click_count': self.click_count,
            'revenue': self.revenue,
            'ctr': round((self.click_count / self.impression_count * 100), 2) if self.impression_count > 0 else 0,
            'cpm': round((self.revenue / self.impression_count * 1000), 2) if self.impression_count > 0 else 0
        }


class AnalyticsEvent(db.Model):
    __tablename__ = 'analytics_events'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = db.Column(db.Enum('page_view', 'job_view', 'service_view', 'ad_impression', 'ad_click', 'search', 'apply_click', 'share', 'bookmark', name='event_types'), nullable=False, index=True)
    resource_id = db.Column(UUID(as_uuid=True), index=True)
    resource_type = db.Column(db.String(50))
    user_agent = db.Column(db.String(500))
    ip_address = db.Column(db.String(45))
    referrer = db.Column(db.String(500))
    session_id = db.Column(db.String(100))
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    location = db.Column(db.String(100))
    device_type = db.Column(db.String(20))
    browser = db.Column(db.String(50))
    metadata = db.Column(JSONB, default=dict)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    @staticmethod
    def log_event(event_type: str, resource_id: str = None, resource_type: str = None, metadata: Dict = None):
        try:
            user_agent = request.headers.get('User-Agent', '')
            device_type = 'mobile' if any(mobile in user_agent.lower() for mobile in ['mobile', 'android', 'iphone']) else 'desktop'
            
            browser = 'unknown'
            if 'chrome' in user_agent.lower():
                browser = 'chrome'
            elif 'firefox' in user_agent.lower():
                browser = 'firefox'
            elif 'safari' in user_agent.lower():
                browser = 'safari'
            elif 'edge' in user_agent.lower():
                browser = 'edge'
            
            event = AnalyticsEvent(
                event_type=event_type,
                resource_id=resource_id,
                resource_type=resource_type,
                user_agent=user_agent,
                ip_address=request.remote_addr,
                referrer=request.headers.get('Referer'),
                session_id=session.get('session_id'),
                user_id=g.current_user.id if hasattr(g, 'current_user') else None,
                device_type=device_type,
                browser=browser,
                metadata=metadata or {}
            )
            db.session.add(event)
            db.session.commit()
        except Exception as e:
            logger.error("Failed to log analytics event", error=str(e), event_type=event_type)


class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.id'))
    action = db.Column(db.String(100), nullable=False, index=True)
    resource_type = db.Column(db.String(50), index=True)
    resource_id = db.Column(UUID(as_uuid=True))
    old_values = db.Column(JSONB)
    new_values = db.Column(JSONB)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='audit_logs')
    
    @staticmethod
    def log_action(action: str, resource_type: str = None, resource_id: str = None, 
                  old_values: Dict = None, new_values: Dict = None, success: bool = True, error_message: str = None):
        try:
            log = AuditLog(
                user_id=g.current_user.id if hasattr(g, 'current_user') else None,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=old_values,
                new_values=new_values,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                success=success,
                error_message=error_message
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logger.error("Failed to create audit log", error=str(e), action=action)


def create_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='Asia/Kolkata',
        enable_utc=True,
        task_routes={
            'sync_indian_jobs': {'queue': 'indian_jobs'},
            'send_notification': {'queue': 'notifications'},
            'process_analytics': {'queue': 'analytics'},
        },
        beat_schedule={
            'sync-indian-job-sources': {
                'task': 'sync_indian_jobs',
                'schedule': crontab(minute=0, hour='*/2'),
            },
            'sync-government-jobs': {
                'task': 'sync_government_jobs',
                'schedule': crontab(minute=30, hour='*/4'),
            },
            'cleanup-expired-jobs': {
                'task': 'cleanup_expired_jobs',
                'schedule': crontab(hour=3, minute=0),
            },
            'update-urgent-jobs': {
                'task': 'update_urgent_jobs',
                'schedule': crontab(minute='*/30'),
            },
        }
    )
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery


def init_cloudinary(app):
    cloudinary.config(
        cloud_name=app.config['CLOUDINARY_CLOUD_NAME'],
        api_key=app.config['CLOUDINARY_API_KEY'],
        api_secret=app.config['CLOUDINARY_API_SECRET'],
        secure=True
    )


def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
        elif 'user_id' in session:
            user = User.query.get(session['user_id'])
            if user and user.is_active:
                g.current_user = user
                return f(*args, **kwargs)
        
        if not token:
            return jsonify({'error': 'Authentication token missing'}), 401
        
        if token in blacklisted_tokens:
            return jsonify({'error': 'Token has been revoked'}), 401
        
        try:
            payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            user = User.query.get(payload['sub'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        if not user or not user.is_active:
            return jsonify({'error': 'User not found or inactive'}), 401
        
        g.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            if required_role == 'super_admin' and g.current_user.role != 'super_admin':
                return jsonify({'error': 'Super admin access required'}), 403
            elif required_role == 'admin' and g.current_user.role not in ['admin', 'super_admin']:
                return jsonify({'error': 'Admin access required'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_json_request(schema_class):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON payload'}), 400
            
            schema = schema_class()
            try:
                validated_data = schema.load(data)
                g.validated_data = validated_data
            except ValidationError as e:
                return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def sanitize_html(content: str) -> str:
    if not content:
        return content
    
    allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    allowed_attributes = {'a': ['href', 'target']}
    
    return bleach.clean(content, tags=allowed_tags, attributes=allowed_attributes, strip=True)


def handle_file_upload(file, folder='general', max_size_mb=10):
    if not file:
        return None, 'No file provided'
    
    if file.content_length and file.content_length > max_size_mb * 1024 * 1024:
        return None, f'File too large. Maximum size is {max_size_mb}MB'
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf'}
    file_ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    
    if file_ext not in allowed_extensions:
        return None, 'Invalid file type. Only images and PDFs are allowed'
    
    try:
        resource_type = "image" if file_ext != '.pdf' else "raw"
        result = cloudinary.uploader.upload(
            file,
            folder=f"sridhar-services/{folder}",
            resource_type=resource_type,
            quality="auto:good" if resource_type == "image" else None,
            fetch_format="auto" if resource_type == "image" else None,
            transformation=[
                {'width': 1200, 'height': 800, 'crop': 'limit'},
                {'quality': 'auto:good'}
            ] if resource_type == "image" else None
        )
        return result['secure_url'], None
    except Exception as e:
        logger.error("File upload failed", error=str(e))
        return None, 'File upload failed'


async def fetch_indian_jobs_from_api_async(source: JobSource, session: aiohttp.ClientSession) -> List[Dict]:
    if not source.can_make_request():
        logger.warning("Rate limit exceeded for source", source=source.name)
        return []
    
    api_config = INDIAN_JOB_APIS.get(source.api_identifier)
    if not api_config:
        logger.error("Unknown API identifier", identifier=source.api_identifier)
        return []
    
    headers = {'User-Agent': 'Sridhar-Services-Bot/1.0 (Indian Jobs Aggregator)'}
    
    if source.api_key and api_config.api_key_required:
        if api_config.auth_header == 'Authorization':
            headers['Authorization'] = f'Bearer {source.api_key}'
        else:
            headers[api_config.auth_header] = source.api_key
    
    params = {}
    if api_config.supports_search and api_config.location_filter:
        params['location'] = 'Telangana,Andhra Pradesh'
        params['state'] = 'TS,AP'
    
    try:
        source.increment_request_count()
        
        async with session.get(source.api_url, headers=headers, params=params, timeout=30) as response:
            if response.status == 429:
                logger.warning("Rate limited by API", source=source.name)
                return []
            
            response.raise_for_status()
            
            if api_config.data_format == 'json':
                data = await response.json()
            else:
                text = await response.text()
                data = feedparser.parse(text)
            
            if isinstance(data, dict):
                if 'jobs' in data:
                    jobs_data = data['jobs']
                elif 'data' in data:
                    jobs_data = data['data']
                elif 'results' in data:
                    jobs_data = data['results']
                elif 'notifications' in data:
                    jobs_data = data['notifications']
                else:
                    jobs_data = [data]
            else:
                jobs_data = data if isinstance(data, list) else []
            
            filtered_jobs = []
            for job in jobs_data:
                location = job.get('location', job.get('state', ''))
                if any(state_keyword in location.lower() for state_keyword in ['telangana', 'andhra pradesh', 'hyderabad', 'vijayawada', 'visakhapatnam', 'ts', 'ap']):
                    filtered_jobs.append(job)
            
            source.record_success(len(filtered_jobs))
            return filtered_jobs[:150]
    
    except asyncio.TimeoutError:
        error_msg = "Request timeout"
        logger.error("API request timeout", source=source.name)
        source.record_error(error_msg)
        JOB_SYNC_ERRORS.labels(source=source.name).inc()
        return []
    
    except Exception as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error("API request failed", source=source.name, error=str(e))
        source.record_error(error_msg)
        JOB_SYNC_ERRORS.labels(source=source.name).inc()
        return []


def process_imported_indian_job(job_data: Dict, source: JobSource) -> Optional[Job]:
    try:
        api_config = INDIAN_JOB_APIS.get(source.api_identifier)
        if not api_config:
            return None
        
        extractor_map = {
            'sarkari_result': IndianJobDataExtractor.extract_sarkari_result,
            'freshersworld': IndianJobDataExtractor.extract_freshersworld,
            'naukri': IndianJobDataExtractor.extract_naukri,
            'indeed_india': IndianJobDataExtractor.extract_indeed_india,
            'government_jobs_india': IndianJobDataExtractor.extract_government_jobs,
            'freejobalert': IndianJobDataExtractor.extract_freejobalert,
        }
        
        extractor = extractor_map.get(source.api_identifier)
        if extractor:
            extracted_data = extractor(job_data)
        else:
            extracted_data = job_data
        
        if not extracted_data.get('external_id'):
            extracted_data['external_id'] = hashlib.md5(
                f"{extracted_data.get('title', '')}{extracted_data.get('company', '')}{source.api_identifier}".encode()
            ).hexdigest()
        
        existing_job = Job.query.filter_by(
            external_id=extracted_data['external_id'],
            source_id=source.id
        ).first()
        
        if existing_job:
            return existing_job
        
        admin_user = User.query.filter_by(role='super_admin').first()
        if not admin_user:
            admin_user = User.query.filter_by(role='admin').first()
        
        if not admin_user:
            logger.error("No admin user found to assign imported job")
            return None
        
        location = extracted_data.get('location', '')
        state = extracted_data.get('state', '')
        if not state and location:
            if any(city in location for city in INDIAN_STATES['telangana']):
                state = 'Telangana'
            elif any(city in location for city in INDIAN_STATES['andhra_pradesh']):
                state = 'Andhra Pradesh'
        
        job = Job(
            title=sanitize_html(extracted_data.get('title', 'Untitled Job'))[:200],
            company=sanitize_html(extracted_data.get('company', 'Unknown Company'))[:150],
            location=location,
            state=state,
            job_type=extracted_data.get('job_type', 'private'),
            category=extracted_data.get('category', 'General'),
            description=sanitize_html(extracted_data.get('description', '')),
            requirements=sanitize_html(extracted_data.get('requirements', '')),
            salary_range=extracted_data.get('salary_range', ''),
            experience_level=extracted_data.get('experience_level', ''),
            education_required=extracted_data.get('education_required', ''),
            age_limit=extracted_data.get('age_limit', ''),
            application_url=extracted_data.get('application_url', ''),
            application_fee=extracted_data.get('application_fee', ''),
            selection_process=extracted_data.get('selection_process', ''),
            total_vacancies=extracted_data.get('total_vacancies', ''),
            important_dates=extracted_data.get('important_dates', {}),
            notification_pdf=extracted_data.get('notification_pdf', ''),
            external_id=extracted_data['external_id'],
            status='moderation',
            author_id=admin_user.id,
            source_id=source.id,
            scraped_at=datetime.utcnow(),
            language=extracted_data.get('language', 'english,hindi'),
            remote_eligible=extracted_data.get('job_type') == 'remote'
        )
        
        if extracted_data.get('application_deadline'):
            try:
                if isinstance(extracted_data['application_deadline'], str):
                    job.application_deadline = date_parser.parse(extracted_data['application_deadline']).date()
                else:
                    job.application_deadline = extracted_data['application_deadline']
            except (ValueError, TypeError):
                pass
        
        if extracted_data.get('exam_date'):
            try:
                if isinstance(extracted_data['exam_date'], str):
                    job.exam_date = date_parser.parse(extracted_data['exam_date']).date()
            except (ValueError, TypeError):
                pass
        
        if job.application_deadline and job.application_deadline <= datetime.now().date() + timedelta(days=7):
            job.is_urgent = True
        
        job.extract_salary_range()
        job.generate_tags()
        
        db.session.add(job)
        db.session.commit()
        
        return job
    
    except Exception as e:
        logger.error("Failed to process imported Indian job", error=str(e), source=source.name)
        return None


async def sync_indian_job_sources():
    with current_app.app_context():
        sources = JobSource.query.filter_by(source_type='api', is_active=True).all()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in sources:
                if source.api_identifier in INDIAN_JOB_APIS:
                    task = fetch_indian_jobs_from_api_async(source, session)
                    tasks.append((source, task))
            
            for source, task in tasks:
                try:
                    jobs_data = await task
                    processed_count = 0
                    
                    for job_data in jobs_data:
                        job = process_imported_indian_job(job_data, source)
                        if job:
                            processed_count += 1
                    
                    logger.info("Synced Indian jobs from source", source=source.name, count=processed_count)
                    
                except Exception as e:
                    logger.error("Failed to sync Indian jobs from source", source=source.name, error=str(e))


def create_app():
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://localhost/sridhar_services_india')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20
    }
    
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    app.config['CELERY_BROKER_URL'] = redis_url + '/0'
    app.config['CELERY_RESULT_BACKEND'] = redis_url + '/0'
    app.config['CACHE_TYPE'] = 'RedisCache'
    app.config['CACHE_REDIS_URL'] = redis_url + '/1'
    app.config['RATELIMIT_STORAGE_URL'] = redis_url + '/2'
    
    app.config['CLOUDINARY_CLOUD_NAME'] = os.environ.get('CLOUDINARY_CLOUD_NAME')
    app.config['CLOUDINARY_API_KEY'] = os.environ.get('CLOUDINARY_API_KEY')
    app.config['CLOUDINARY_API_SECRET'] = os.environ.get('CLOUDINARY_API_SECRET')
    
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
    
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    limiter.init_app(app)
    cache.init_app(app)
    jwt_manager.init_app(app)
    compress.init_app(app)
    
    CORS(app, origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))
    
    Talisman(app, force_https=os.environ.get('FORCE_HTTPS', 'False').lower() == 'true')
    
    init_cloudinary(app)
    celery = create_celery(app)
    
    @celery.task(name='sync_indian_jobs')
    def sync_indian_jobs_task():
        asyncio.run(sync_indian_job_sources())
    
    @celery.task(name='sync_government_jobs')
    def sync_government_jobs_task():
        with app.app_context():
            govt_sources = JobSource.query.filter(
                JobSource.api_identifier.in_(['tspsc', 'appsc', 'upsc', 'ssc', 'railway_jobs'])
            ).all()
            
            for source in govt_sources:
                try:
                    asyncio.run(fetch_indian_jobs_from_api_async(source, aiohttp.ClientSession()))
                except Exception as e:
                    logger.error("Failed to sync government jobs", source=source.name, error=str(e))
    
    @celery.task(name='cleanup_expired_jobs')
    def cleanup_expired_jobs_task():
        with app.app_context():
            try:
                expired_jobs = Job.query.filter(
                    or_(
                        and_(Job.application_deadline.isnot(None), Job.application_deadline < datetime.now().date()),
                        and_(Job.expires_at.isnot(None), Job.expires_at < datetime.utcnow())
                    ),
                    Job.status == 'published'
                ).all()
                
                for job in expired_jobs:
                    job.status = 'expired'
                
                db.session.commit()
                logger.info("Cleaned up expired jobs", count=len(expired_jobs))
                
            except Exception as e:
                logger.error("Failed to cleanup expired jobs", error=str(e))
    
    @celery.task(name='update_urgent_jobs')
    def update_urgent_jobs_task():
        with app.app_context():
            try:
                urgent_deadline = datetime.now().date() + timedelta(days=7)
                jobs_to_update = Job.query.filter(
                    Job.application_deadline <= urgent_deadline,
                    Job.application_deadline >= datetime.now().date(),
                    Job.is_urgent == False,
                    Job.status == 'published'
                ).all()
                
                for job in jobs_to_update:
                    job.is_urgent = True
                
                db.session.commit()
                logger.info("Updated urgent jobs", count=len(jobs_to_update))
                
            except Exception as e:
                logger.error("Failed to update urgent jobs", error=str(e))
    
    @app.before_first_request
    def initialize_app():
        db.create_all()
        
        if not User.query.filter_by(role='super_admin').first():
            admin = User(
                username='admin',
                email='admin@sridharservices.com',
                role='super_admin',
                email_verified=True,
                preferred_locations=['Hyderabad, Telangana', 'Vijayawada, Andhra Pradesh'],
                language_preference='english'
            )
            admin.set_password(os.environ.get('ADMIN_PASSWORD', 'SecureAdmin123!'))
            db.session.add(admin)
            db.session.commit()
            logger.info("Created default super admin user")
        
        indian_job_sources = [
            ('sarkari_result', 'Sarkari Result', 'https://www.sarkariresult.com/api/latest-jobs'),
            ('tspsc', 'TSPSC Jobs', 'https://www.tspsc.gov.in/api/notifications'),
            ('appsc', 'APPSC Jobs', 'https://psc.ap.gov.in/api/notifications'),
            ('freejobalert', 'Free Job Alert', 'https://www.freejobalert.com/api/jobs'),
            ('government_jobs_india', 'Government Jobs India', 'https://www.governmentjobsindia.com/api/jobs'),
            ('employment_news', 'Employment News', 'https://www.employmentnews.gov.in/api/notifications'),
            ('upsc', 'UPSC Jobs', 'https://upsc.gov.in/api/examinations'),
            ('ssc', 'SSC Jobs', 'https://ssc.nic.in/api/notifications'),
            ('railway_jobs', 'Railway Recruitment', 'https://www.rrcb.gov.in/api/notifications'),
            ('ibps', 'IBPS Jobs', 'https://www.ibps.in/api/notifications'),
        ]
        
        for api_id, name, url in indian_job_sources:
            if not JobSource.query.filter_by(api_identifier=api_id).first():
                config = INDIAN_JOB_APIS.get(api_id, {})
                source = JobSource(
                    name=name,
                    api_identifier=api_id,
                    api_url=url,
                    source_type='api',
                    rate_limit_per_hour=getattr(config, 'rate_limit_per_hour', 100),
                    target_states=['Telangana', 'Andhra Pradesh'],
                    priority=1 if 'government' in name.lower() else 2
                )
                db.session.add(source)
        
        db.session.commit()
        logger.info("Initialized Indian job sources")
    
    @app.before_request
    def before_request():
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status='in_progress'
        ).inc()
    
    @app.after_request
    def after_request(response):
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
        
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()
    
    @jwt_manager.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        return jwt_payload['jti'] in blacklisted_tokens
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({'error': 'Forbidden', 'message': 'Access denied'}), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': 'Resource not found'}), 404
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({'error': 'Rate limit exceeded', 'message': 'Too many requests'}), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        logger.error("Internal server error", error=str(error))
        return jsonify({'error': 'Internal server error', 'message': 'Something went wrong'}), 500
    
    @app.route('/api/v1/auth/login', methods=['POST'])
    @limiter.limit("5 per minute")
    @validate_json_request(UserSchema)
    def login():
        data = g.validated_data
        
        user = User.query.filter(
            or_(User.username == data['username'], User.email == data['username'])
        ).first()
        
        if not user:
            AuditLog.log_action('login_failed', error_message='User not found')
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if user.is_locked():
            AuditLog.log_action('login_failed', error_message='Account locked')
            return jsonify({'error': 'Account is temporarily locked'}), 401
        
        if not user.check_password(data['password']):
            user.login_attempts += 1
            if user.login_attempts >= 5:
                user.lock_account()
            db.session.commit()
            
            AuditLog.log_action('login_failed', error_message='Invalid password')
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            AuditLog.log_action('login_failed', error_message='Account inactive')
            return jsonify({'error': 'Account is deactivated'}), 401
        
        user.reset_login_attempts()
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        AuditLog.log_action('login_success', new_values={'user_id': str(user.id)})
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': user.to_dict(include_sensitive=True)
        })
    
    @app.route('/api/v1/jobs', methods=['GET'])
    @cache.cached(timeout=300, query_string=True)
    def get_jobs():
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        job_type = request.args.get('type')
        category = request.args.get('category')
        search = request.args.get('search', '').strip()
        location = request.args.get('location')
        state = request.args.get('state')
        remote_only = request.args.get('remote_only', type=bool)
        government_only = request.args.get('government_only', type=bool)
        urgent_only = request.args.get('urgent_only', type=bool)
        salary_min = request.args.get('salary_min', type=int)
        salary_max = request.args.get('salary_max', type=int)
        sort_by = request.args.get('sort', 'created_at')
        order = request.args.get('order', 'desc')
        language = request.args.get('language', 'all')
        
        query = Job.query.filter_by(status='published')
        
        if job_type:
            query = query.filter_by(job_type=job_type)
        if category:
            query = query.filter_by(category=category)
        if location:
            query = query.filter(Job.location.ilike(f'%{location}%'))
        if state:
            query = query.filter_by(state=state)
        if remote_only:
            query = query.filter_by(remote_eligible=True)
        if government_only:
            query = query.filter_by(job_type='government')
        if urgent_only:
            query = query.filter_by(is_urgent=True)
        if salary_min:
            query = query.filter(Job.salary_min >= salary_min)
        if salary_max:
            query = query.filter(Job.salary_max <= salary_max)
        if language != 'all':
            query = query.filter(Job.language.like(f'%{language}%'))
        
        if search:
            if len(search) > 2:
                query = query.filter(
                    Job.search_vector.match(search)
                )
            else:
                query = query.filter(
                    or_(
                        Job.title.ilike(f'%{search}%'),
                        Job.company.ilike(f'%{search}%'),
                        Job.description.ilike(f'%{search}%')
                    )
                )
            
            AnalyticsEvent.log_event('search', metadata={'query': search, 'results_page': 'jobs'})
        
        if sort_by == 'salary':
            sort_column = Job.salary_max
        elif sort_by == 'company':
            sort_column = Job.company
        elif sort_by == 'views':
            sort_column = Job.view_count
        elif sort_by == 'deadline':
            sort_column = Job.application_deadline
        else:
            sort_column = Job.created_at
        
        if order == 'asc':
            query = query.order_by(asc(sort_column))
        else:
            query = query.order_by(desc(sort_column))
        
        query = query.order_by(desc(Job.is_featured), desc(Job.is_urgent), desc(Job.created_at))
        
        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        AnalyticsEvent.log_event('page_view', metadata={'page': 'jobs', 'search': search, 'state': state})
        
        return jsonify({
            'jobs': [job.to_dict() for job in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'per_page': per_page,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
            'filters': {
                'states': ['Telangana', 'Andhra Pradesh'],
                'categories': INDIAN_JOB_CATEGORIES,
                'job_types': ['government', 'private', 'remote'],
                'education_levels': EDUCATION_QUALIFICATIONS
            }
        })
    
    @app.route('/api/v1/jobs/<job_id>', methods=['GET'])
    def get_job(job_id):
        job = Job.query.get_or_404(job_id)
        
        if job.status != 'published':
            abort(404)
        
        job.increment_view_count()
        AnalyticsEvent.log_event('job_view', resource_id=job_id, resource_type='job', 
                                metadata={'state': job.state, 'category': job.category})
        
        ACTIVE_JOBS.set(Job.query.filter_by(status='published').count())
        
        related_jobs = Job.query.filter(
            Job.id != job.id,
            Job.status == 'published',
            or_(
                Job.category == job.category,
                Job.state == job.state,
                Job.company == job.company
            )
        ).limit(5).all()
        
        return jsonify({
            'job': job.to_dict(include_stats=True),
            'related_jobs': [related_job.to_dict() for related_job in related_jobs]
        })
    
    @app.route('/api/v1/jobs/<job_id>/share', methods=['POST'])
    def share_job(job_id):
        job = Job.query.get_or_404(job_id)
        job.increment_share_count()
        AnalyticsEvent.log_event('share', resource_id=job_id, resource_type='job')
        return jsonify({'message': 'Share tracked'})
    
    @app.route('/api/v1/jobs/<job_id>/bookmark', methods=['POST'])
    def bookmark_job(job_id):
        job = Job.query.get_or_404(job_id)
        job.increment_bookmark_count()
        AnalyticsEvent.log_event('bookmark', resource_id=job_id, resource_type='job')
        return jsonify({'message': 'Bookmark tracked'})
    
    @app.route('/api/v1/jobs/government', methods=['GET'])
    @cache.cached(timeout=600, query_string=True)
    def get_government_jobs():
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 50)
        state = request.args.get('state')
        category = request.args.get('category')
        education = request.args.get('education')
        
        query = Job.query.filter_by(status='published', job_type='government')
        
        if state:
            query = query.filter_by(state=state)
        if category:
            query = query.filter_by(category=category)
        if education:
            query = query.filter(Job.education_required.ilike(f'%{education}%'))
        
        query = query.order_by(desc(Job.is_urgent), desc(Job.application_deadline), desc(Job.created_at))
        
        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        AnalyticsEvent.log_event('page_view', metadata={'page': 'government_jobs', 'state': state})
        
        return jsonify({
            'jobs': [job.to_dict() for job in pagination.items],
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page,
            'per_page': per_page
        })
    
    @app.route('/api/v1/jobs/urgent', methods=['GET'])
    @cache.cached(timeout=180)
    def get_urgent_jobs():
        urgent_jobs = Job.query.filter_by(
            status='published', 
            is_urgent=True
        ).order_by(desc(Job.application_deadline)).limit(20).all()
        
        AnalyticsEvent.log_event('page_view', metadata={'page': 'urgent_jobs'})
        
        return jsonify({
            'jobs': [job.to_dict() for job in urgent_jobs],
            'count': len(urgent_jobs)
        })
    
    @app.route('/api/v1/admin/sync-indian-jobs', methods=['POST'])
    @jwt_required()
    @require_role('super_admin')
    def trigger_indian_job_sync():
        try:
            sync_indian_jobs_task.delay()
            return jsonify({'message': 'Indian job sync started successfully'})
        except Exception as e:
            logger.error("Failed to trigger Indian job sync", error=str(e))
            return jsonify({'error': 'Failed to start job sync'}), 500
    
    @app.route('/api/v1/analytics/jobs', methods=['GET'])
    @cache.cached(timeout=3600)
    def get_job_analytics():
        days = request.args.get('days', 7, type=int)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        total_jobs = Job.query.filter_by(status='published').count()
        telangana_jobs = Job.query.filter_by(status='published', state='Telangana').count()
        ap_jobs = Job.query.filter_by(status='published', state='Andhra Pradesh').count()
        government_jobs = Job.query.filter_by(status='published', job_type='government').count()
        urgent_jobs = Job.query.filter_by(status='published', is_urgent=True).count()
        
        category_stats = db.session.query(
            Job.category,
            func.count(Job.id).label('count')
        ).filter_by(status='published').group_by(Job.category).all()
        
        recent_views = db.session.query(func.count(AnalyticsEvent.id)).filter(
            AnalyticsEvent.event_type == 'job_view',
            AnalyticsEvent.timestamp >= start_date
        ).scalar()
        
        return jsonify({
            'summary': {
                'total_jobs': total_jobs,
                'telangana_jobs': telangana_jobs,
                'andhra_pradesh_jobs': ap_jobs,
                'government_jobs': government_jobs,
                'urgent_jobs': urgent_jobs,
                'recent_views': recent_views
            },
            'category_distribution': [
                {'category': stat.category, 'count': stat.count}
                for stat in category_stats
            ]
        })
    
    @app.route('/api/v1/stats', methods=['GET'])
    @cache.cached(timeout=900)
    def get_public_stats():
        stats = {
            'total_jobs': Job.query.filter_by(status='published').count(),
            'government_jobs': Job.query.filter_by(status='published', job_type='government').count(),
            'private_jobs': Job.query.filter_by(status='published', job_type='private').count(),
            'remote_jobs': Job.query.filter_by(status='published', job_type='remote').count(),
            'telangana_jobs': Job.query.filter_by(status='published', state='Telangana').count(),
            'andhra_pradesh_jobs': Job.query.filter_by(status='published', state='Andhra Pradesh').count(),
            'urgent_jobs': Job.query.filter_by(status='published', is_urgent=True).count(),
            'total_services': Service.query.filter_by(is_active=True).count(),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return jsonify(stats)
    
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        try:
            db.session.execute(text('SELECT 1'))
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0.0-india',
                'region': 'India - Telangana & Andhra Pradesh',
                'services': {
                    'database': 'up',
                    'redis': 'up',
                    'celery': 'up'
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=os.environ.get('FLASK_ENV') == 'development',
        host='0.0.0.0',
        port=port
    )