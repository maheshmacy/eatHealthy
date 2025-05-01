import streamlit as st
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules from separate files
from api_client import *
from helpers import *
from ui_components import *

# Set page configuration
st.set_page_config(
    page_title="EATSMART-AI",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session state management
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'force_rerun' not in st.session_state:
    st.session_state.force_rerun = False
if 'selected_meal' not in st.session_state:
    st.session_state.selected_meal = None
if 'history_view' not in st.session_state:
    st.session_state.history_view = "list"  # Can be "list", "detail", or "stats"
if 'meal_history_data' not in st.session_state:
    st.session_state.meal_history_data = None
if 'meal_stats_data' not in st.session_state:
    st.session_state.meal_stats_data = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# App title and description
st.title("🍏 EATSMART-AI")
st.markdown("### Personalized Glycemic Index Tracking")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "User Registration", "Food Analysis", "History"])

# Check API health
try:
    api_health = get_health_check()
    if api_health and api_health.get('status') == 'healthy':
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Not Connected")
        st.error("Cannot connect to the backend API. Please check if the server is running.")
except Exception as e:
    logger.error(f"API health check failed: {e}")
    st.sidebar.error("❌ API Not Connected")
    st.error(f"Failed to connect to the backend API: {e}")

# Home page
if page == "Home":
    render_home_page()
    
    # If user is logged in, show quick access
    if st.session_state.user_id:
        st.success(f"Logged in as User ID: {st.session_state.user_id}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze Food", use_container_width=True):
                st.session_state.page = "Food Analysis"
                st.rerun()
        with col2:
            if st.button("View Meal History", use_container_width=True):
                st.session_state.page = "History"
                st.rerun()
        
    # Check if a rerun is needed after login
    if st.session_state.force_rerun:
        st.session_state.force_rerun = False
        st.rerun()

# User Registration page
elif page == "User Registration":
    st.header("User Registration")
    
    # Check if user is already logged in
    if st.session_state.user_id:
        render_profile_update_form()
    else:
        render_registration_form()

# Food Analysis page
elif page == "Food Analysis":
    st.header("Food Analysis")
    
    if not st.session_state.user_id:
        st.warning("Please register or login first to analyze food.")
        if st.button("Go to Registration"):
            st.session_state.page = "User Registration"
            st.rerun()
    else:
        st.info(f"Logged in as User ID: {st.session_state.user_id}")
        render_food_analysis_page()

# History page
elif page == "History":
    st.header("Meal History")
    
    if not st.session_state.user_id:
        st.warning("Please register or login first to view your history.")
        if st.button("Go to Registration"):
            st.session_state.page = "User Registration"
            st.rerun()
    else:
        render_meal_history_page()

# Add footer
st.markdown("---")
st.markdown("#### EATSMART-AI: Personalized Glycemic Index Tracking")
st.markdown("Powered by AI for better health decisions")