// Application Configuration
// This file centralizes all environment variable access

export const config = {
  // API Configuration
  API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:5002',
  
  // Environment
  ENVIRONMENT: process.env.REACT_APP_ENVIRONMENT || 'development',
  
  // Development settings
  IS_DEVELOPMENT: process.env.REACT_APP_ENVIRONMENT === 'development',
  IS_PRODUCTION: process.env.REACT_APP_ENVIRONMENT === 'production',
  
  // API Endpoints
  ENDPOINTS: {
    CALCULATE: '/api/calculate',
    OPTIMIZE_STEP: '/api/optimize_step',
  }
};

// Helper function to build full API URLs
export const buildApiUrl = (endpoint: string): string => {
  return `${config.API_BASE_URL}${endpoint}`;
};

// Export individual values for convenience
export const { API_BASE_URL, ENVIRONMENT, IS_DEVELOPMENT, IS_PRODUCTION } = config; 