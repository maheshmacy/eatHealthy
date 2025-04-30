"""
Database utilities for the GI Personalize app.
"""
import os
import json
import sqlite3
import logging
from config import Config

logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initialize the database if it doesn't exist.
    """
    os.makedirs(os.path.dirname(Config.DATABASE_FILE), exist_ok=True)
    
    try:
        conn = sqlite3.connect(Config.DATABASE_FILE)
        cursor = conn.cursor()
        
        # Create users table to track user IDs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)

def get_user_data(user_id):
    """
    Get user data from file storage.
    
    Args:
        user_id (str): User ID
        
    Returns:
        dict: User data or None if not found
    """
    user_file = os.path.join(Config.USER_DATA_FOLDER, f"{user_id}.json")
    
    if not os.path.exists(user_file):
        return None
    
    try:
        with open(user_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading user data for {user_id}: {str(e)}", exc_info=True)
        return None

def save_user_data(user_id, user_data):
    """
    Save user data to file storage.
    
    Args:
        user_id (str): User ID
        user_data (dict): User data to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create user_data directory if it doesn't exist
        os.makedirs(Config.USER_DATA_FOLDER, exist_ok=True)
        
        # Save user data to JSON file
        user_file = os.path.join(Config.USER_DATA_FOLDER, f"{user_id}.json")
        
        with open(user_file, 'w') as f:
            json.dump(user_data, f, indent=2)
        
        # Add user to database if new
        conn = sqlite3.connect(Config.DATABASE_FILE)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving user data for {user_id}: {str(e)}", exc_info=True)
        return False

def delete_user(user_id):
    """
    Delete a user's data.
    
    Args:
        user_id (str): User ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Remove user file
        user_file = os.path.join(Config.USER_DATA_FOLDER, f"{user_id}.json")
        if os.path.exists(user_file):
            os.remove(user_file)
        
        # Remove user from database
        conn = sqlite3.connect(Config.DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        # Remove user model if exists
        model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_model.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        return True
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {str(e)}", exc_info=True)
        return False

def get_all_users():
    """
    Get a list of all user IDs.
    
    Returns:
        list: List of user IDs
    """
    try:
        conn = sqlite3.connect(Config.DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    except Exception as e:
        logger.error(f"Error getting all users: {str(e)}", exc_info=True)
        return []
