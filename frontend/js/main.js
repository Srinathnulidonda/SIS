// API Configuration
const API_URL = process.env.API_URL || 'http://localhost:5000/api';
const FRONTEND_URL = window.location.origin;

// Storage Keys
const STORAGE_KEYS = {
    USER: 'user_data',
    TOKEN: 'auth_token',
    THEME: 'theme_preference'
};

// Utility Functions
const utils = {
    // Show toast notification
    showToast: (message, type = 'info') => {
        const toastContainer = document.getElementById('toastContainer') || createToastContainer();
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        toastContainer.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        setTimeout(() => toast.remove(), 5000);
    },

    // Format date
    formatDate: (dateString) => {
        const options = { year: 'numeric', month: 'short', day: 'numeric' };
        return new Date(dateString).toLocaleDateString('en-IN', options);
    },

    // Format currency
    formatCurrency: (amount) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR'
        }).format(amount);
    },

    // Get user from storage
    getUser: () => {
        const userData = localStorage.getItem(STORAGE_KEYS.USER);
        return userData ? JSON.parse(userData) : null;
    },

    // Save user to storage
    saveUser: (user) => {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
    },

    // Clear user data
    clearUser: () => {
        localStorage.removeItem(STORAGE_KEYS.USER);
        localStorage.removeItem(STORAGE_KEYS.TOKEN);
    },

    // Make API request
    apiRequest: async (endpoint, options = {}) => {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include'
        };

        const response = await fetch(`${API_URL}${endpoint}`, {
            ...defaultOptions,
            ...options
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API request failed');
        }

        return response.json();
    }
};

// Create toast container
function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

// Page-specific functions
const pageHandlers = {
    // Initialize home page
    initHomePage: async () => {
        // Track page visit
        trackPageVisit('/');
        
        // Load job ticker
        loadJobTicker();
        
        // Load latest jobs
        loadLatestJobs();
        
        // Track ad impressions
        trackAdImpression('banner');
    },

    // Initialize jobs page
    initJobsPage: async () => {
        trackPageVisit('/jobs');
        
        // Load job filters
        loadJobFilters();
        
        // Load jobs with pagination
        loadJobs(1);
        
        // Setup search
        setupJobSearch();
    },

    // Initialize admin dashboard
    initAdminDashboard: async () => {
        const user = utils.getUser();
        if (!user || !['admin', 'manager'].includes(user.role)) {
            window.location.href = '/';
            return;
        }

        trackPageVisit('/admin');
        
        // Load admin stats
        loadAdminStats();
        
        // Load pending jobs
        loadPendingJobs();
        
        // Setup job management
        setupJobManagement();
    },

    // Initialize manager dashboard
    initManagerDashboard: async () => {
        const user = utils.getUser();
        if (!user || user.role !== 'manager') {
            window.location.href = '/';
            return;
        }

        trackPageVisit('/manager');
        
        // Load analytics
        loadAnalytics();
        
        // Load users
        loadUsers();
        
        // Setup user management
        setupUserManagement();
    }
};

// Job-related functions
async function loadJobTicker() {
    try {
        const data = await utils.apiRequest('/jobs?per_page=5');
        const ticker = document.getElementById('jobTicker');
        if (ticker && data.jobs) {
            const tickerHTML = data.jobs.map(job => 
                `<span class="text-nowrap">
                    <span class="badge bg-warning text-dark me-2">${job.category}</span>
                    ${job.title} - ${job.location || 'Remote'}
                </span>`
            ).join(' • ');
            ticker.innerHTML = tickerHTML + ' • ' + tickerHTML; // Duplicate for continuous scroll
        }
    } catch (error) {
        console.error('Error loading job ticker:', error);
    }
}

async function loadLatestJobs() {
    try {
        const data = await utils.apiRequest('/jobs?per_page=6');
        const container = document.getElementById('latestJobs');
        if (container && data.jobs) {
            container.innerHTML = data.jobs.map(job => `
                <div class="col-md-6 col-lg-4">
                    <div class="job-card">
                        <span class="job-badge ${job.category}">${job.category}</span>
                        <h5 class="mt-3">${job.title}</h5>
                        <p class="text-muted mb-2">
                            <i class="fas fa-map-marker-alt me-2"></i>${job.location || 'Remote'}
                        </p>
                        <p class="text-truncate">${job.description}</p>
                        <div class="d-flex justify-content-between align-items-center mt-3">
                            <small class="text-muted">${utils.formatDate(job.created_at)}</small>
                            <a href="/jobs.html?id=${job.id}" class="btn btn-sm btn-primary">View Details</a>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Error loading latest jobs:', error);
    }
}

async function loadJobs(page = 1) {
    const container = document.getElementById('jobsList');
    if (!container) return;

    container.innerHTML = '<div class="spinner-container"><div class="spinner-border text-primary"></div></div>';

    try {
        const params = new URLSearchParams(window.location.search);
        params.set('page', page);
        
        const data = await utils.apiRequest(`/jobs?${params.toString()}`);
        
        container.innerHTML = data.jobs.map(job => `
            <div class="col-12">
                <div class="job-card mb-3">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <div class="d-flex align-items-center mb-2">
                                <span class="job-badge ${job.category} me-2">${job.category}</span>
                                ${job.job_type ? `<span class="badge bg-secondary">${job.job_type}</span>` : ''}
                            </div>
                            <h4>${job.title}</h4>
                            <p class="text-muted mb-2">
                                <i class="fas fa-map-marker-alt me-2"></i>${job.location || 'Remote'}
                                ${job.salary_range ? `<span class="ms-3"><i class="fas fa-money-bill-wave me-2"></i>${job.salary_range}</span>` : ''}
                            </p>
                            <p class="mb-0">${job.description.substring(0, 200)}...</p>
                        </div>
                        <div class="col-md-4 text-md-end mt-3 mt-md-0">
                            <p class="text-muted small mb-2">Posted ${utils.formatDate(job.created_at)}</p>
                            <a href="${job.link}" target="_blank" class="btn btn-primary">
                                Apply Now <i class="fas fa-external-link-alt ms-2"></i>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        // Add pagination
        if (data.pages > 1) {
            container.innerHTML += createPagination(data.current_page, data.pages);
        }
    } catch (error) {
        container.innerHTML = '<div class="alert alert-danger">Error loading jobs. Please try again later.</div>';
        console.error('Error loading jobs:', error);
    }
}

function createPagination(currentPage, totalPages) {
    let pages = [];
    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || (i >= currentPage - 2 && i <= currentPage + 2)) {
            pages.push(i);
        } else if (pages[pages.length - 1] !== '...') {
            pages.push('...');
        }
    }

    return `
        <nav class="mt-4">
            <ul class="pagination justify-content-center">
                <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                    <a class="page-link" href="#" onclick="loadJobs(${currentPage - 1}); return false;">Previous</a>
                </li>
                ${pages.map(page => 
                    page === '...' 
                        ? '<li class="page-item disabled"><span class="page-link">...</span></li>'
                        : `<li class="page-item ${page === currentPage ? 'active' : ''}">
                            <a class="page-link" href="#" onclick="loadJobs(${page}); return false;">${page}</a>
                          </li>`
                ).join('')}
                <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                    <a class="page-link" href="#" onclick="loadJobs(${currentPage + 1}); return false;">Next</a>
                </li>
            </ul>
        </nav>
    `;
}

// Authentication functions
async function login(email, password, role = 'user') {
    try {
        const data = await utils.apiRequest('/login', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });

        utils.saveUser(data.user);
        utils.showToast('Login successful!', 'success');
        
        // Redirect based on role
        if (data.user.role === 'manager') {
            window.location.href = '/manager.html';
        } else if (data.user.role === 'admin') {
            window.location.href = '/admin.html';
        } else {
            window.location.href = '/';
        }
    } catch (error) {
        utils.showToast(error.message, 'danger');
    }
}

async function logout() {
    try {
        await utils.apiRequest('/logout', { method: 'POST' });
        utils.clearUser();
        utils.showToast('Logged out successfully', 'info');
        window.location.href = '/';
    } catch (error) {
        console.error('Logout error:', error);
        utils.clearUser();
        window.location.href = '/';
    }
}

// Analytics tracking
async function trackPageVisit(page) {
    try {
        await utils.apiRequest('/analytics/page', {
            method: 'POST',
            body: JSON.stringify({ page })
        });
    } catch (error) {
        console.error('Analytics tracking error:', error);
    }
}

async function trackAdImpression(adType) {
    try {
        await utils.apiRequest('/ads/impression', {
            method: 'POST',
            body: JSON.stringify({ ad_type: adType })
        });
    } catch (error) {
        console.error('Ad impression tracking error:', error);
    }
}

async function trackAdClick(adType) {
    try {
        await utils.apiRequest('/ads/click', {
            method: 'POST',
            body: JSON.stringify({ ad_type: adType })
        });
    } catch (error) {
        console.error('Ad click tracking error:', error);
    }
}

// Setup event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize page based on current path
    const path = window.location.pathname;
    
    if (path === '/' || path === '/index.html') {
        pageHandlers.initHomePage();
    } else if (path === '/jobs.html') {
        pageHandlers.initJobsPage();
    } else if (path === '/admin.html') {
        pageHandlers.initAdminDashboard();
    } else if (path === '/manager.html') {
        pageHandlers.initManagerDashboard();
    }

    // Setup ad click tracking
    document.querySelectorAll('.google-ad-banner').forEach(ad => {
        ad.addEventListener('click', () => {
            trackAdClick(ad.dataset.adType || 'banner');
        });
    });

    // Check authentication status
    const user = utils.getUser();
    if (user) {
        updateNavigationForUser(user);
    }
});

// Update navigation based on user role
function updateNavigationForUser(user) {
    const loginDropdown = document.getElementById('loginDropdown');
    if (loginDropdown) {
        loginDropdown.innerHTML = `<i class="fas fa-user-circle"></i> ${user.name}`;
        
        const dropdownMenu = loginDropdown.nextElementSibling;
        if (dropdownMenu) {
            dropdownMenu.innerHTML = `
                ${user.role === 'manager' ? '<li><a class="dropdown-item" href="/manager.html">Manager Dashboard</a></li>' : ''}
                ${['admin', 'manager'].includes(user.role) ? '<li><a class="dropdown-item" href="/admin.html">Admin Dashboard</a></li>' : ''}
                <li><hr class="dropdown-divider"></li>
                <li><a class="dropdown-item" href="#" onclick="logout(); return false;">Logout</a></li>
            `;
        }
    }
}

// Export for use in other scripts
window.appUtils = utils;
window.appAuth = { login, logout };
window.appTracking = { trackPageVisit, trackAdImpression, trackAdClick };