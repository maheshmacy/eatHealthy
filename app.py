import streamlit as st
import requests
import json
import base64
import os
import pandas as pd
from io import BytesIO
from PIL import Image
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Eat Health AI",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSV file management for user persistence
def init_profiles_csv():
    """Initialize the CSV file for user profiles if it doesn't exist"""
    profiles_path = 'eatsmart_profiles.csv'
    
    if not os.path.exists(profiles_path):
        # Create empty dataframe with the correct columns
        columns = ['id', 'name', 'age', 'gender', 'height', 'weight', 
                  'exercise_days', 'activity_level', 'diabetes_status', 'date_created']
        df = pd.DataFrame(columns=columns)
        df.to_csv(profiles_path, index=False)
        
    return profiles_path

def get_all_users():
    """Retrieve all users from the CSV file"""
    profiles_path = init_profiles_csv()
    try:
        users = pd.read_csv(profiles_path)
        return users
    except Exception as e:
        st.error(f"Error loading user profiles: {e}")
        return pd.DataFrame()

def save_user_to_csv(profile_data):
    """Save a new user to the CSV file or update existing user"""
    profiles_path = init_profiles_csv()
    
    try:
        # Load existing profiles
        if os.path.exists(profiles_path) and os.path.getsize(profiles_path) > 0:
            users_df = pd.read_csv(profiles_path)
        else:
            # Create a new dataframe if file doesn't exist or is empty
            columns = ['id', 'name', 'age', 'gender', 'height', 'weight', 
                      'exercise_days', 'activity_level', 'diabetes_status', 'date_created']
            users_df = pd.DataFrame(columns=columns)
        
        # Check if user with same name exists
        existing_user = users_df[users_df['name'] == profile_data['name']]
        
        if not existing_user.empty:
            # Update existing user
            user_id = existing_user.iloc[0]['id']
            users_df.loc[users_df['name'] == profile_data['name'], [
                'age', 'gender', 'height', 'weight', 'exercise_days', 
                'activity_level', 'diabetes_status'
            ]] = [
                profile_data['age'], profile_data['gender'], profile_data['height'], 
                profile_data['weight'], profile_data['exercise_days'], 
                profile_data['activity_level'], profile_data['diabetes_status']
            ]
        else:
            # Create a new user
            user_id = 1 if users_df.empty else (users_df['id'].max() + 1)
            new_user = {
                'id': user_id,
                'name': profile_data['name'],
                'age': profile_data['age'],
                'gender': profile_data['gender'],
                'height': profile_data['height'],
                'weight': profile_data['weight'],
                'exercise_days': profile_data['exercise_days'],
                'activity_level': profile_data['activity_level'],
                'diabetes_status': profile_data['diabetes_status'],
                'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
        
        # Save the updated dataframe
        users_df.to_csv(profiles_path, index=False)
        return user_id
    
    except Exception as e:
        st.error(f"Error saving user profile: {e}")
        return None

def get_user_by_id(user_id):
    """Get a user's profile by ID from the CSV file"""
    profiles_path = init_profiles_csv()
    
    try:
        users_df = pd.read_csv(profiles_path)
        user = users_df[users_df['id'] == user_id]
        
        if not user.empty:
            return user.iloc[0].to_dict()
        return None
    
    except Exception as e:
        st.error(f"Error retrieving user profile: {e}")
        return None

# Initialize profiles CSV
init_profiles_csv()

# Your custom CSS (same as before)
st.markdown("""
    <style>
    /* CSS code remains the same */
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables (same as before)
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_user_id' not in st.session_state:
    st.session_state.selected_user_id = None
if 'food_image' not in st.session_state:
    st.session_state.food_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'api_url' not in st.session_state:
    st.session_state.api_url = ""

# Navigation functions (same as before)
def go_to_home():
    st.session_state.page = 'home'

def go_to_profile_list():
    st.session_state.page = 'profile_list'

def go_to_create_profile():
    st.session_state.page = 'create_profile'
    st.session_state.profile_data = {
        'name': '',
        'age': 30,
        'gender': 'Male',
        'height': 170,
        'weight': 70,
        'exercise_days': 3,
        'activity_level': 'Moderate (moderate exercise 3-5 days/week)',
        'diabetes_status': 'Normal'
    }

def go_to_edit_profile(user_id):
    st.session_state.page = 'create_profile'
    user_data = get_user_by_id(user_id)
    st.session_state.profile_data = user_data
    st.session_state.editing_user_id = user_id

def go_to_analyze():
    st.session_state.page = 'analyze'

def go_to_results():
    st.session_state.page = 'results'

def select_user(user_id):
    st.session_state.selected_user_id = user_id

def save_profile(profile_data):
    user_id = save_user_to_csv(profile_data)
    if user_id:
        st.session_state.selected_user_id = user_id
        st.session_state.page = 'profile_list'
        st.success(f"Profile for {profile_data['name']} saved successfully!")
    else:
        st.error("Failed to save profile. Please try again.")

def analyze_food(image=None, user_id=None):
    """Send the food image to the Gradio API for analysis"""
    if image is not None:
        st.session_state.food_image = image
    
    if user_id is not None:
        st.session_state.selected_user_id = user_id
    
    # Get the selected user's profile
    user_profile = get_user_by_id(st.session_state.selected_user_id)
    if not user_profile:
        st.error("Selected user profile not found")
        return None
    
    # Check if we have an API URL and an image
    if not st.session_state.api_url:
        st.error("Please enter the Gradio API URL first")
        return None
    
    if st.session_state.food_image is None:
        st.error("Please upload a food image first")
        return None
    
    # Add debug information
    st.info(f"Connecting to API: {st.session_state.api_url}")
    
    # Prepare the image for sending
    buffered = BytesIO()
    st.session_state.food_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare data for API
    data = {
        "image": f"data:image/png;base64,{image_base64}",
        "user_data": {
            "age": user_profile["age"],
            "gender": user_profile["gender"],
            "height": user_profile["height"],
            "weight": user_profile["weight"],
            "exercise_days": user_profile["exercise_days"],
            "activity_level": user_profile["activity_level"],
            "diabetes_status": user_profile["diabetes_status"]
        }
    }
    
    # Make API request
    try:
        # For Gradio API - use the exact URL format from your Colab output
        api_url = st.session_state.api_url.rstrip('/')
        
        # Add a debug message showing the full URL being used
        full_url = f"{api_url}/api/analyze"
        st.info(f"Making request to: {full_url}")
        
        response = requests.post(full_url, json=data)
        
        st.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.analysis_results = result
            st.session_state.page = 'results'
            return result
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
            # Fallback to mock results for demo purposes
            return use_mock_data(user_profile)
    except Exception as e:
        st.error(f"Error connecting to the API: {e}")
        # Fallback to mock results for demo purposes
        # return use_mock_data(user_profile)

def use_mock_data(user_profile):
    """Fallback to mock results for testing purposes"""
    mock_results = {
        "food_name": "Chocolate Cake",
        "category": "Dessert",
        "nutritional_info": {
            "calories": 350,
            "carbs": 45,
            "protein": 5,
            "fat": 15
        },
        "glucose_impact": {
            "level": "High",
            "score": 85,
            "curve": [80, 85, 95, 120, 155, 180, 170, 150, 130, 115, 100, 90, 85],
            "time_points": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        },
        "recommendations": [
            "This dessert has a high sugar content that may cause a significant glucose spike",
            "Consider having a smaller portion or sharing with someone",
            "Pairing with protein can help reduce the glucose impact"
        ],
        "user_name": user_profile["name"]
    }
    st.session_state.analysis_results = mock_results
    st.session_state.page = 'results'
    return mock_results

# Home Page
def render_home():
    st.markdown("<h1 style='color: #27ae60;'>Eat Health AI</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='header-text'>Your Personal Food Health Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subheader-text'>Make informed food choices with AI-powered analysis that predicts glucose spikes based on your unique health profile.</p>", unsafe_allow_html=True)
    
    # Setup API URL if not set
    if not st.session_state.api_url:
        st.text_input("Gradio API URL", key="api_url_input", 
                     placeholder="https://c0c10dd7ad9f427b84.gradio.livep")
        if st.session_state.api_url_input:
            st.session_state.api_url = st.session_state.api_url_input
            st.success("API URL set successfully!")
    
    # Main action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        if st.button("Manage Profiles 👤", key="manage_profiles_btn"):
            go_to_profile_list()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        if st.button("Analyze Food 📊", key="analyze_food_btn"):
            go_to_analyze()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div><span class='icon-green'>❤️</span> <b>Personalized Health Profile</b></div>", unsafe_allow_html=True)
    st.markdown("<p>Create your detailed health profile including age, weight, exercise habits, and lifestyle factors.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div><span class='icon-green'>🍽️</span> <b>Instant Food Analysis</b></div>", unsafe_allow_html=True)
    st.markdown("<p>Take a photo of your meal and get immediate identification of what you're about to eat.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div><span class='icon-green'>📊</span> <b>Glucose Impact Prediction</b></div>", unsafe_allow_html=True)
    st.markdown("<p>Understand how your food choices will affect your glucose levels based on your unique profile.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Profile List Page
def render_profile_list():
    # Navigation bar
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    st.markdown("<h2>User Profiles</h2>", unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<div class='secondary-btn' style='width: 120px;'>", unsafe_allow_html=True)
    if st.button("← Back", key="profile_list_back_btn"):
        go_to_home()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='subheader-text'>Select a profile to edit or create a new one.</p>", unsafe_allow_html=True)
    
    # Get all users
    users = get_all_users()
    
    if not users.empty:
        st.markdown("<h3>Existing Profiles</h3>", unsafe_allow_html=True)
        for index, user in users.iterrows():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                # User card
                is_selected = st.session_state.selected_user_id == user['id']
                card_class = "user-profile-card selected" if is_selected else "user-profile-card"
                st.markdown(f"<div class='{card_class}' id='user-{user['id']}'>", unsafe_allow_html=True)
                st.markdown(f"<b>{user['name']}</b> ({user['age']})", unsafe_allow_html=True)
                st.markdown(f"{user['gender']} | {user['height']} cm | {user['weight']} kg", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[1]:
                if st.button("Select", key=f"select_user_{user['id']}"):
                    select_user(user['id'])
                    st.success(f"Selected {user['name']}")
                    st.experimental_rerun()
            
            with cols[2]:
                if st.button("Edit", key=f"edit_user_{user['id']}"):
                    go_to_edit_profile(user['id'])
    else:
        st.info("No user profiles found. Create your first profile!")
    
    st.markdown("<div class='primary-btn' style='margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("+ Create New Profile", key="create_profile_btn"):
        go_to_create_profile()
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.selected_user_id:
        st.markdown("<div class='primary-btn' style='margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("Analyze Food with Selected Profile", key="analyze_with_profile_btn"):
            go_to_analyze()
        st.markdown("</div>", unsafe_allow_html=True)

# Profile Creation Page
def render_create_profile():
    # Check if we're editing an existing profile or creating a new one
    editing = 'editing_user_id' in st.session_state
    
    # Navigation bar
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    title = "Edit Profile" if editing else "Create New Profile"
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<div class='secondary-btn' style='width: 120px;'>", unsafe_allow_html=True)
    if st.button("← Back", key="create_profile_back_btn"):
        if 'editing_user_id' in st.session_state:
            del st.session_state.editing_user_id
        go_to_profile_list()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='subheader-text'>Enter your information to create a personalized profile for glucose predictions.</p>", unsafe_allow_html=True)
    
    # Load profile data
    profile_data = st.session_state.profile_data if 'profile_data' in st.session_state else {
        'name': '',
        'age': 30,
        'gender': 'Male',
        'height': 170,
        'weight': 70,
        'exercise_days': 3,
        'activity_level': 'Moderate (moderate exercise 3-5 days/week)',
        'diabetes_status': 'Normal'
    }
    
    # Basic Information
    profile_data['name'] = st.text_input("Full Name", value=profile_data.get('name', ''))
    
    col1, col2 = st.columns(2)
    with col1:
        profile_data['age'] = st.number_input("Age", value=int(profile_data.get('age', 30)), min_value=1, max_value=120)
    with col2:
        profile_data['gender'] = st.radio("Gender", options=["Male", "Female", "Other"], horizontal=True, index=["Male", "Female", "Other"].index(profile_data.get('gender', 'Male')))
    
    col1, col2 = st.columns(2)
    with col1:
        profile_data['height'] = st.number_input("Height (cm)", value=float(profile_data.get('height', 170)), min_value=50.0, max_value=250.0)
    with col2:
        profile_data['weight'] = st.number_input("Weight (kg)", value=float(profile_data.get('weight', 70)), min_value=20.0, max_value=300.0)
    
    # Lifestyle Information
    profile_data['exercise_days'] = st.slider("Exercise (days per week)", 0, 7, int(profile_data.get('exercise_days', 3)))
    
    activity_options = [
        "Sedentary (little or no exercise)",
        "Light (light exercise 1-3 days/week)",
        "Moderate (moderate exercise 3-5 days/week)",
        "Active (hard exercise 6-7 days/week)",
        "Very Active (twice daily training)"
    ]
    profile_data['activity_level'] = st.selectbox(
        "Activity Level",
        options=activity_options,
        index=activity_options.index(profile_data.get('activity_level', 'Moderate (moderate exercise 3-5 days/week)'))
    )
    
    # Health Information
    diabetes_options = ["Normal", "Prediabetes", "Type 1 Diabetes", "Type 2 Diabetes", "Gestational"]
    profile_data['diabetes_status'] = st.selectbox(
        "Diabetes Status",
        options=diabetes_options,
        index=diabetes_options.index(profile_data.get('diabetes_status', 'Normal'))
    )
    
    # Save and Cancel buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
        if st.button("Cancel", key="cancel_profile_btn"):
            if 'editing_user_id' in st.session_state:
                del st.session_state.editing_user_id
            go_to_profile_list()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        if st.button("Save Profile", key="save_profile_btn"):
            if not profile_data['name']:
                st.error("Please enter a name for the profile")
            else:
                save_profile(profile_data)
                if 'editing_user_id' in st.session_state:
                    del st.session_state.editing_user_id
        st.markdown("</div>", unsafe_allow_html=True)

# Food Analysis Page
def render_analyze():
    # Navigation bar
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    st.markdown("<h2>Analyze Your Food</h2>", unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<div class='secondary-btn' style='width: 120px;'>", unsafe_allow_html=True)
    if st.button("← Back", key="analyze_back_btn"):
        go_to_home()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<p class='subheader-text'>Take a photo of your food to get nutritional insights and glucose impact.</p>", unsafe_allow_html=True)
    
    # User Profile Selection
    st.markdown("<h3>Select User Profile</h3>", unsafe_allow_html=True)
    users = get_all_users()
    
    if users.empty:
        st.warning("No user profiles found. Please create a profile first.")
        st.markdown("<div class='primary-btn' style='margin-top: 20px;'>", unsafe_allow_html=True)
        if st.button("Create Profile", key="create_profile_from_analyze"):
            go_to_create_profile()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Create a dropdown of users
    user_options = users[['id', 'name']].set_index('id')['name'].to_dict()
    selected_user_id = st.selectbox(
        "Choose a profile",
        options=list(user_options.keys()),
        format_func=lambda x: user_options[x],
        index=0 if st.session_state.selected_user_id not in user_options else list(user_options.keys()).index(st.session_state.selected_user_id)
    )
    
    st.session_state.selected_user_id = selected_user_id
    selected_user = get_user_by_id(selected_user_id)
    
    # Display selected user info
    if selected_user:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div><b>{selected_user['name']}</b> ({selected_user['age']}, {selected_user['gender']})</div>", unsafe_allow_html=True)
        st.markdown(f"<div>Height: {selected_user['height']} cm | Weight: {selected_user['weight']} kg</div>", unsafe_allow_html=True)
        st.markdown(f"<div>Activity Level: {selected_user['activity_level']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Image upload area
    st.markdown("<h3>Food Image</h3>", unsafe_allow_html=True)
    
    # Image upload/capture options
    image_source = st.radio("Image Source", ["Upload Image", "Take Photo"], horizontal=True)
    
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
            st.session_state.food_image = image
    else:  # Take Photo
        # Streamlit doesn't natively support camera access, so we'll create a custom component
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 5px; padding: 20px; text-align: center;">
            <p>For camera access, we recommend using a mobile device or installing a webcam extension.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative using an experimental component
        try:
            camera_image = st.camera_input("Take a picture of your food")
            if camera_image is not None:
                image = Image.open(camera_image)
                st.session_state.food_image = image
        except:
            st.warning("Camera input not supported in this Streamlit version.")
    
    # Analysis button
    st.markdown("<div class='primary-btn' style='margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("Analyze Food", key="analyze_food_action_btn", disabled=st.session_state.food_image is None):
        analyze_food(st.session_state.food_image, st.session_state.selected_user_id)
    st.markdown("</div>", unsafe_allow_html=True)

# Results Page
def render_results():
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please analyze a food first.")
        if st.button("Go to Analyze", key="go_to_analyze_btn"):
            go_to_analyze()
        return
    
    results = st.session_state.analysis_results
    
    # Navigation bar
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    st.markdown("<h2>Food Analysis Results</h2>", unsafe_allow_html=True)
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("<div class='secondary-btn' style='width: 120px;'>", unsafe_allow_html=True)
    if st.button("← Back", key="results_back_btn"):
        go_to_analyze()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Get user name if available
    user_name = results.get("user_name", "")
    if not user_name and st.session_state.selected_user_id:
        user = get_user_by_id(st.session_state.selected_user_id)
        if user:
            user_name = user["name"]
    
    if user_name:
        st.markdown(f"<p class='subheader-text'>Here's how {results['food_name']} would affect {user_name}'s glucose levels.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='subheader-text'>Here's how {results['food_name']} would affect your glucose levels.</p>", unsafe_allow_html=True)
    
    # Display food image and name
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.session_state.food_image is not None:
            st.image(st.session_state.food_image, width=150)
        else:
            st.markdown(
                """
                <div style="
                    width: 100px;
                    height: 100px;
                    background-color: #eee;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <span style="color: #aaa;">No image</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    with col2:
        st.markdown(f"<h3>{results['food_name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>{results['category']}</p>", unsafe_allow_html=True)
    
    # Nutritional information
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666; margin-bottom: 5px;'>Calories</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0;'>{results['nutritional_info']['calories']}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666; margin-bottom: 5px;'>Carbs (g)</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0;'>{results['nutritional_info']['carbs']}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666; margin-bottom: 5px;'>Protein (g)</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-top: 0;'>{results['nutritional_info']['protein']}</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Glucose impact
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Glucose Impact</h3>", unsafe_allow_html=True)
    
    # Impact level badge
    impact_level = results['glucose_impact']['level']
    impact_color_class = ""
    if impact_level == "High":
        impact_color_class = "high-risk"
    elif impact_level == "Medium":
        impact_color_class = "medium-risk"
    else:
        impact_color_class = "low-risk"
    
    # Progress bar for impact score
    impact_score = results['glucose_impact']['score']
    st.progress(impact_score / 100)
    
    st.markdown(
        f"<div style='text-align: right;'><span class='{impact_color_class}'>{impact_level}</span></div>",
        unsafe_allow_html=True
    )
    
    # Glucose curve visualization
    if 'curve' in results['glucose_impact'] and 'time_points' in results['glucose_impact']:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 6))
        time_points = results['glucose_impact']['time_points']
        glucose_curve = results['glucose_impact']['curve']
        
        ax.plot(time_points, glucose_curve, 'b-', linewidth=2)
        ax.set_title('Predicted Blood Glucose Response', fontsize=16)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=140, color='r', linestyle='--', alpha=0.5, label='High threshold')
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Low threshold')
        
        # Fill areas
        ax.fill_between(time_points, glucose_curve, 80, 
                       where=(np.array(glucose_curve) > 140), 
                       color='red', alpha=0.3)
        ax.fill_between(time_points, glucose_curve, 80, 
                       where=(np.array(glucose_curve) <= 140) & (np.array(glucose_curve) >= 70), 
                       color='green', alpha=0.3)
        
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("<p>Based on your profile, this food may cause a glucose spike as shown above.</p>", unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("<div class='recommendations'>", unsafe_allow_html=True)
    st.markdown("<p><b>ℹ️ Recommendations</b></p>", unsafe_allow_html=True)
    for rec in results['recommendations']:
        st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
        if st.button("📷 Analyze Another", key="analyze_another_btn"):
            go_to_analyze()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        if st.button("Back to Home", key="back_to_home_btn"):
            go_to_home()
        st.markdown("</div>", unsafe_allow_html=True)

# Render the appropriate page based on session state
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'profile_list':
    render_profile_list()
elif st.session_state.page == 'create_profile':
    render_create_profile()
elif st.session_state.page == 'analyze':
    render_analyze()
elif st.session_state.page == 'results':
    render_results()