import streamlit as st
import requests
import json
from datetime import datetime

# API endpoint (change to your Flask backend URL)
API_ENDPOINT = "http://localhost:5000"

def create_user(user_data):
    """Create a new user via API"""
    url = f"{API_ENDPOINT}/users"
    response = requests.post(url, json=user_data)
    return response.json() if response.status_code == 201 else None

def get_user(user_id):
    """Get user profile via API"""
    try:
        url = f"{API_ENDPOINT}/users/{user_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def update_user(user_id, user_data):
    """Update user profile via API"""
    try:
        url = f"{API_ENDPOINT}/users/{user_id}"
        response = requests.put(url, json=user_data)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def analyze_food_image(user_id, image_file):
    """Analyze food image via API"""
    url = f"{API_ENDPOINT}/food/analyze-food"
    
    files = {'food_image': image_file}
    data = {'user_id': user_id}
    
    response = requests.post(url, files=files, data=data)
    return response.json() if response.status_code == 200 else None

def get_health_check():
    """Check API health"""
    url = f"{API_ENDPOINT}/health"
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_meal_history(user_id, start_date=None, end_date=None, food_type=None, sort_by='date', sort_order='desc', limit=50):
    """Get meal history for a user with optional filters"""
    try:
        url = f"{API_ENDPOINT}/meals/{user_id}"
        
        # Build query parameters
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if food_type:
            params['food_type'] = food_type
        if sort_by:
            params['sort_by'] = sort_by
        if sort_order:
            params['sort_order'] = sort_order
        if limit:
            params['limit'] = str(limit)
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching meal history: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_meal_details(user_id, meal_id):
    """Get detailed information about a specific meal"""
    try:
        url = f"{API_ENDPOINT}/meals/{user_id}/{meal_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching meal details: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_meal_stats(user_id):
    """Get meal statistics for a user"""
    try:
        url = f"{API_ENDPOINT}/meals/{user_id}/stats"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching meal statistics: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def submit_meal_feedback(user_id, meal_id, feedback_data):
    """Submit feedback for a meal"""
    try:
        url = f"{API_ENDPOINT}/meals/{user_id}/{meal_id}/feedback"
        response = requests.post(url, json=feedback_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error submitting feedback: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None