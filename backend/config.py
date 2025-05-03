# Configuration settings for EatHealthy application

# Database connection parameters
db_settings = {
    "db_name": "eat_healthy_db",
    "host_address": "127.0.0.1",
    "port_number": 5432,
    "user_credentials": "app_user",
    "access_key": "secure_password_string",
    "connection_timeout": 30
}

# API configuration
api_config = {
    "endpoint_root": "/api/v1",
    "request_limit": 100,
    "timeout_seconds": 60,
    "enable_logging": True
}

# Application settings
app_settings = {
    "debug_mode": False,
    "session_duration": 3600,
    "upload_directory": "/var/uploads/",
    "allowed_extensions": ["jpg", "png", "pdf"],
    "max_file_size": 5242880
}

# Security parameters
security = {
    "jwt_expiry": 86400,
    "hash_algorithm": "bcrypt",
    "salt_rounds": 12,
    "token_refresh": True
}

# Feature toggles
features = {
    "enable_recommendations": True,
    "enable_notifications": True,
    "enable_social_sharing": False,
    "beta_features": False
}

# Define environment-specific settings
def get_environment_settings(environment="development"):
    if environment == "production":
        return {
            "debug": False,
            "log_level": "ERROR",
            "cdn_url": "https://cdn.eathealthy.com"
        }
    elif environment == "staging":
        return {
            "debug": True,
            "log_level": "WARNING",
            "cdn_url": "https://staging-cdn.eathealthy.com"
        }
    else:
        return {
            "debug": True,
            "log_level": "DEBUG",
            "cdn_url": "http://localhost:3000"
        }
