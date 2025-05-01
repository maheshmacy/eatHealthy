import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="EATSMART-AI",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint (change to your Flask backend URL)
API_ENDPOINT = "http://localhost:5000"

# Functions to interact with the API
def create_user(user_data):
    """Create a new user via API"""
    url = f"{API_ENDPOINT}/users"
    response = requests.post(url, json=user_data)
    return response.json() if response.status_code == 201 else None

def get_user(user_id):
    """Get user profile via API"""
    url = f"{API_ENDPOINT}/users/{user_id}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

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

# Helper functions
def create_glucose_graph(glucose_prediction):
    """Create interactive glucose prediction graph"""
    # Check if we have valid prediction data
    if not glucose_prediction:
        return None
    
    # Check if we have time series data specifically
    has_time_series = 'time_series' in glucose_prediction and glucose_prediction['time_series']
    
    # If no time series but we have predicted_glucose, create a simple single point visualization
    if not has_time_series and 'predicted_glucose' in glucose_prediction:
        # Create a simple point plot showing just the predicted glucose value
        fig = go.Figure()
        
        # Add single point marker
        predicted_glucose = glucose_prediction['predicted_glucose']
        fig.add_trace(go.Scatter(
            x=[0],  # Just showing the prediction at the start
            y=[predicted_glucose],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Predicted Glucose',
            hovertemplate='Predicted Peak Glucose: %{y:.1f} mg/dL'
        ))
        
        # Add category ranges as horizontal bands
        fig.add_shape(type="rect", x0=-10, x1=10, y0=180, y1=400,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=140, y1=180,
                      fillcolor="rgba(255,165,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=70, y1=140,
                      fillcolor="rgba(0,128,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=0, y1=70,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0), layer="below")
        
        # Add category text labels
        fig.add_annotation(x=8, y=190, text="High", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=8, y=160, text="Elevated", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=8, y=105, text="Normal", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=8, y=35, text="Low", showarrow=False, font=dict(color="red"))
        
        # Update layout
        fig.update_layout(
            title='Predicted Peak Glucose',
            yaxis_title='Blood Glucose (mg/dL)',
            xaxis_visible=False,
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Set y-axis range to show all categories
        fig.update_yaxes(range=[0, 250])
        
        return fig
    
    # If we have proper time series data, create the full graph
    elif has_time_series:
        time_series = glucose_prediction['time_series']
        
        # If the time series is a list of dictionaries
        if isinstance(time_series, list) and isinstance(time_series[0], dict):
            df = pd.DataFrame(time_series)
            x = df.get('minute', [])
            y = df.get('glucose', [])
        else:
            # If it's just a list of values
            x = list(range(0, len(time_series) * 5, 5))
            y = time_series
        
        # If we still don't have valid x,y data, return None
        if not x or not y:
            return None
            
        # Create glucose range areas
        fig = go.Figure()
        
        # Add glucose ranges (colored areas)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[180] * len(x) + [400] * len(x),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='High',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[140] * len(x) + [180] * len(x),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.1)',
            line=dict(color='rgba(255, 165, 0, 0)'),
            name='Elevated',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[70] * len(x) + [140] * len(x),
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.1)',
            line=dict(color='rgba(0, 128, 0, 0)'),
            name='Normal',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[0] * len(x) + [70] * len(x),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Low',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add predicted glucose curve
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            name='Predicted Glucose',
            hovertemplate='Time: %{x} min<br>Glucose: %{y:.1f} mg/dL'
        ))
        
        # Add peak marker if available
        max_glucose_time = glucose_prediction.get('max_glucose_time', 0)
        if max_glucose_time > 0 and max_glucose_time in x:
            max_index = x.index(max_glucose_time)
            max_glucose = y[max_index] if max_index < len(y) else None
            
            if max_glucose is not None:
                fig.add_trace(go.Scatter(
                    x=[max_glucose_time],
                    y=[max_glucose],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Peak Glucose',
                    hovertemplate='Peak Time: %{x} min<br>Peak Glucose: %{y:.1f} mg/dL'
                ))
        
        # Update layout
        fig.update_layout(
            title='Predicted Glucose Response',
            xaxis_title='Time (minutes after meal)',
            yaxis_title='Blood Glucose (mg/dL)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Add range lines
        max_x = max(x)
        fig.add_shape(type="line", x0=0, y0=70, x1=max_x, y1=70, 
                    line=dict(color="green", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=140, x1=max_x, y1=140, 
                    line=dict(color="orange", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=180, x1=max_x, y1=180, 
                    line=dict(color="red", width=1, dash="dash"))
        
        return fig
    
    # If neither condition is met, return None
    return None

def get_nutrient_chart(nutrients):
    """Create a bar chart for nutritional information"""
    if not nutrients:
        return None
    
    # Create data for bar chart
    nutrient_names = []
    nutrient_values = []
    
    for key, value in nutrients.items():
        if key not in ['calories']:  # Excluding calories as it's typically much higher than others
            nutrient_names.append(key.capitalize())
            nutrient_values.append(value)
    
    # Create bar chart
    fig = px.bar(
        x=nutrient_names,
        y=nutrient_values,
        title="Nutritional Content (g)",
        labels={"x": "Nutrient", "y": "Amount (g)"},
        color=nutrient_names,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Grams",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def image_to_base64(image_file):
    """Convert image to base64 for display"""
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

# Session state management
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# App title and description
st.title("🍏 EATSMART-AI")
st.markdown("### Personalized Glycemic Index Tracking")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "User Registration", "Food Analysis", "History"])

# Check API health
api_health = get_health_check()
if api_health and api_health.get('status') == 'healthy':
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Not Connected")
    st.error("Cannot connect to the backend API. Please check if the server is running.")

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to EATSMART-AI!
    
    This application helps you understand how different foods affect your blood glucose levels, 
    providing personalized recommendations based on your health profile.
    
    ### Features:
    
    - **Personalized Analysis**: Get insights tailored to your health profile
    - **Food Recognition**: Upload food images for automatic analysis
    - **Glucose Prediction**: See how your meal will likely affect your blood glucose
    - **Smart Recommendations**: Receive customized dietary suggestions
    
    ### Getting Started:
    
    1. Register your profile in the "User Registration" section
    2. Upload food images in the "Food Analysis" section
    3. View your personalized recommendations
    4. Track your meal history and patterns
    
    Let's start your journey to better health through smart eating!
    """)
    
    # If user is logged in, show quick access
    if st.session_state.user_id:
        st.success(f"Logged in as User ID: {st.session_state.user_id}")
        st.button("Go to Food Analysis", on_click=lambda: st.session_state.update({"page": "Food Analysis"}))

# User Registration page
elif page == "User Registration":
    st.header("User Registration")
    
    if st.session_state.user_id:
        st.success(f"You are registered with User ID: {st.session_state.user_id}")
        if st.button("Log Out"):
            st.session_state.user_id = None
            st.session_state.user_profile = None
            st.session_state.analysis_result = None
            st.experimental_rerun()
    else:
        st.markdown("Please fill in your details to create a personalized profile:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=18, max_value=120, value=35)
            gender = st.selectbox("Gender", ["male", "female", "other"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        
        with col2:
            activity_level = st.selectbox("Activity Level", [
                "sedentary", "lightly_active", "moderately_active", 
                "very_active", "extremely_active"
            ])
            diabetes_status = st.selectbox("Diabetes Status", [
                "none", "pre_diabetic", "type1_diabetes", "type2_diabetes"
            ])
            weight_goal = st.selectbox("Weight Goal", ["lose", "maintain", "gain"])
            
            # Optional fields
            hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=15.0, value=0.0, 
                                   help="Leave at 0 if unknown")
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=0, max_value=300, value=0,
                                             help="Leave at 0 if unknown")
        
        submit_button = st.button("Register")
        
        if submit_button:
            if not name or age < 18 or height <= 0 or weight <= 0:
                st.error("Please fill in all required fields correctly.")
            else:
                user_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "activity_level": activity_level,
                    "diabetes_status": diabetes_status,
                    "weight_goal": weight_goal
                }
                
                # Add optional fields if provided
                if hba1c > 0:
                    user_data["hba1c"] = hba1c
                if fasting_glucose > 0:
                    user_data["fasting_glucose"] = fasting_glucose
                
                with st.spinner("Creating your profile..."):
                    result = create_user(user_data)
                    
                    if result and 'user_id' in result:
                        user_id = result['user_id']
                        st.session_state.user_id = user_id
                        st.session_state.user_profile = user_data
                        st.success(f"Registration successful! Your User ID: {user_id}")
                        st.info("Please save your User ID for future logins.")
                    else:
                        st.error("Registration failed. Please try again or check the API connection.")

        # Alternative: Login with existing user ID
        st.markdown("---")
        st.subheader("Already registered?")
        existing_user_id = st.text_input("Enter your User ID")
        login_button = st.button("Login")
        
        if login_button and existing_user_id:
            with st.spinner("Logging in..."):
                user_profile = get_user(existing_user_id)
                
                if user_profile and 'profile' in user_profile:
                    st.session_state.user_id = existing_user_id
                    st.session_state.user_profile = user_profile['profile']
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("User not found. Please check your User ID.")

# Food Analysis page
elif page == "Food Analysis":
    st.header("Food Analysis")
    
    if not st.session_state.user_id:
        st.warning("Please register or login first to analyze food.")
        st.button("Go to Registration", on_click=lambda: st.session_state.update({"page": "User Registration"}))
    else:
        st.info(f"Logged in as User ID: {st.session_state.user_id}")
        
        # Food image upload
        st.subheader("Upload Food Image")
        uploaded_file = st.file_uploader("Choose an image of your meal", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded meal image", use_container_width=True)
                
                analyze_button = st.button("Analyze Food")
                
                if analyze_button or st.session_state.analysis_result:
                    # If we already have results or the button is clicked
                    if analyze_button:  # Only call API if button is clicked
                        with st.spinner("Analyzing your food..."):
                            # Reset file position
                            uploaded_file.seek(0)
                            result = analyze_food_image(st.session_state.user_id, uploaded_file)
                            
                            if result and 'food_analysis' in result:
                                st.session_state.analysis_result = result
                            else:
                                st.error("Failed to analyze food. Please try again.")
            
            # Display results if available
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                
                with col2:
                    st.subheader("Identified Foods")
                    
                    if 'food_analysis' in result and 'identified_foods' in result['food_analysis']:
                        foods = result['food_analysis']['identified_foods']
                        
                        for i, food in enumerate(foods):
                            st.markdown(f"**{i+1}. {food['food_name']}** (Confidence: {food['confidence']:.2f})")
                            
                            with st.expander(f"Details for {food['food_name']}"):
                                st.markdown(f"**Base Glycemic Index:** {food['base_gi']:.1f}")
                                
                                if 'personalized_gi' in food:
                                    personalized = food['personalized_gi']
                                    st.markdown(f"**Personalized GI Score:** {personalized['personalized_gi_score']:.1f}")
                                    st.markdown(f"**Impact Level:** {personalized['impact_level'].capitalize()}")
                                    st.markdown(f"**Warning Level:** {personalized['warning_level']}")
                                    st.markdown(f"**Recommended Portion:** {personalized['recommended_portion']}")
                                    
                                    if 'recommendations' in personalized and personalized['recommendations']:
                                        st.markdown("**Recommendations:**")
                                        for rec in personalized['recommendations']:
                                            st.markdown(f"- {rec}")
                
                # Overall meal analysis in the next row
                st.markdown("---")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("Nutritional Content")
                    if 'food_analysis' in result and 'total_nutrients' in result['food_analysis']:
                        nutrients = result['food_analysis']['total_nutrients']
                        
                        # Display calories separately
                        if 'calories' in nutrients:
                            st.metric("Calories", f"{nutrients['calories']:.1f} kcal")
                        
                        # Display nutrient chart
                        fig = get_nutrient_chart(nutrients)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display macronutrient ratio
                        if all(key in nutrients for key in ['carbs', 'protein', 'fat']):
                            st.subheader("Macronutrient Ratio")
                            
                            carbs = nutrients['carbs']
                            protein = nutrients['protein']
                            fat = nutrients['fat']
                            total = carbs + protein + fat
                            
                            if total > 0:
                                carbs_pct = (carbs / total) * 100
                                protein_pct = (protein / total) * 100
                                fat_pct = (fat / total) * 100
                                
                                macro_df = pd.DataFrame({
                                    'Macronutrient': ['Carbs', 'Protein', 'Fat'],
                                    'Percentage': [carbs_pct, protein_pct, fat_pct],
                                    'Amount (g)': [carbs, protein, fat]
                                })
                                
                                fig = px.pie(
                                    macro_df, 
                                    values='Percentage', 
                                    names='Macronutrient',
                                    color='Macronutrient',
                                    color_discrete_sequence=['#4CAF50', '#2196F3', '#FFC107'],
                                    hover_data=['Amount (g)']
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
                                st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    st.subheader("Glucose Prediction")
                    if 'glucose_prediction' in result:
                        prediction = result['glucose_prediction']
                        
                        # Display glucose prediction metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            glucose_value = prediction.get('predicted_glucose', 0)
                            st.metric("Peak Glucose", f"{glucose_value:.1f} mg/dL")
                        
                        with col_b:
                            category = prediction.get('glucose_category', 'Unknown')
                            category_color = {
                                'Low': 'red',
                                'Normal': 'green',
                                'Elevated': 'orange',
                                'High': 'red'
                            }.get(category, 'blue')
                            st.markdown(f"**Category:** <span style='color:{category_color};'>{category}</span>", unsafe_allow_html=True)
                        
                        with col_c:
                            max_time = prediction.get('max_glucose_time', 0)
                            if max_time > 0:
                                st.metric("Time to Peak", f"{max_time} min")
                        
                        # Display glucose graph if we can create one
                        fig = create_glucose_graph(prediction)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display recommendations
                        if 'guidelines' in prediction or 'recommendations' in prediction:
                            st.markdown("---")
                            st.subheader("Personalized Recommendations")
                            
                            if 'guidelines' in prediction and prediction['guidelines']:
                                st.markdown("**Guidelines:**")
                                for guideline in prediction['guidelines']:
                                    st.markdown(f"- {guideline}")
                            
                            if 'recommendations' in prediction and prediction['recommendations']:
                                st.markdown("**Specific Recommendations:**")
                                for rec in prediction['recommendations']:
                                    st.markdown(f"- {rec}")
                        elif not prediction.get('time_series') and prediction.get('predicted_glucose'):
                            # If we have a prediction but no recommendations, show a generic message
                            st.info("The system provided a glucose prediction but no specific recommendations for this meal.")
                    else:
                        st.warning("No glucose prediction data available for this meal.")
# History page
elif page == "History":
    st.header("Meal History")
    
    if not st.session_state.user_id:
        st.warning("Please register or login first to view your history.")
        st.button("Go to Registration", on_click=lambda: st.session_state.update({"page": "User Registration"}))
    else:
        st.info("History view is not implemented in this demo as it would require additional API endpoints to retrieve meal history.")
        st.markdown("""
        In a complete implementation, this page would:
        
        1. Fetch meal history data for the user from the backend API
        2. Display a chronological list of analyzed meals
        3. Show trends in glucose responses over time
        4. Provide insights on which foods consistently cause glucose spikes
        5. Recommend dietary adjustments based on historical data
        
        You can create these additional API endpoints in your Flask backend to support these features.
        """)

# Add footer
st.markdown("---")
st.markdown("#### EATSMART-AI: Personalized Glycemic Index Tracking")
st.markdown("Powered by AI for better health decisions")

# CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .css-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)
