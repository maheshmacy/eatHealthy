
import requests
import json
import os
from typing import Dict, Any, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint (change to your Flask backend URL)
API_ENDPOINT = "http://localhost:5000"
# Try to get from environment variable if available
if os.environ.get("API_ENDPOINT"):
    API_ENDPOINT = os.environ.get("API_ENDPOINT")


def create_user(user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new user profile"""
    try:
        response = requests.post(
            f"{API_ENDPOINT}/users",
            json=user_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            return response.json()
        else:
            logger.error(f"Failed to create user: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return None

def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user profile by ID"""
    try:
        response = requests.get(f"{API_ENDPOINT}/users/{user_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get user: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        return None

def update_user(user_id: str, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update user profile"""
    try:
        response = requests.put(
            f"{API_ENDPOINT}/users/{user_id}",
            json=user_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to update user: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        return None

def analyze_food_image(user_id: str, image_file) -> Optional[Dict[str, Any]]:
    """Analyze food image using the API"""
    try:
        files = {'food_image': image_file}
        data = {'user_id': user_id}
        
        response = requests.post(
            f"{API_ENDPOINT}/food/analyze-food",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to analyze food: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error analyzing food: {str(e)}")
        return None


def get_meal_history(
    user_id: str, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    food_type: Optional[str] = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    limit: int = 50
) -> Optional[Dict[str, Any]]:
    """Get meal history for a user with optional filters"""
    try:
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'food_type': food_type,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'limit': limit
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(
            f"{API_ENDPOINT}/meals/{user_id}",
            params=params
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get meal history: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting meal history: {str(e)}")
        return None

def get_meal_details(user_id: str, meal_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific meal"""
    try:
        response = requests.get(f"{API_ENDPOINT}/meals/{user_id}/{meal_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get meal details: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting meal details: {str(e)}")
        return None

def submit_meal_feedback(
    user_id: str,
    meal_id: str,
    feedback_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Submit feedback for a meal"""
    try:
        response = requests.post(
            f"{API_ENDPOINT}/meals/{user_id}/{meal_id}/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to submit feedback: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return None

def get_meal_stats(user_id: str) -> Optional[Dict[str, Any]]:
    """Get meal statistics for a user"""
    try:
        response = requests.get(f"{API_ENDPOINT}/meals/{user_id}/stats")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get meal stats: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting meal stats: {str(e)}")
        return None

def get_image_url(filename: str) -> str:
    """Get the full URL for an image file"""
    return f"{API_ENDPOINT}/api/images/{filename}"