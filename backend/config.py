"""
Configuration settings for the GI Personalize app.
"""
import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-for-dev')
    DEBUG = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    USER_DATA_FOLDER = os.path.join(os.getcwd(), 'user_data')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    
    # Database settings
    DATABASE_FILE = os.path.join(os.getcwd(), 'data', 'database.db')
    
    # ML model settings
    MODEL_THRESHOLD = 5  # Minimum meals with responses to train model
    
    # API rate limits
    RATE_LIMIT = "100 per minute"
