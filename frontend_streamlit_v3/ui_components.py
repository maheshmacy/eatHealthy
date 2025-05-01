import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import logging
from api_client import *
from helpers import *

# Configure logging
logger = logging.getLogger(__name__)

def render_home_page():
    """Render the home page content"""
    st.markdown("""
    ## Welcome to EATSMART-AI!
    
    This application helps you understand how different foods affect your blood glucose levels, 
    providing personalized recommendations based on your health profile.
    
    ### Features:
    
    - **Personalized Analysis**: Get insights tailored to your health profile
    - **Food Recognition**: Upload food images for automatic analysis
    - **Glucose Prediction**: See how your meal will likely affect your blood glucose
    - **Smart Recommendations**: Receive customized dietary suggestions
    - **Meal History Tracking**: Track your meals and glucose responses over time
    
    ### Getting Started:
    
    1. Register your profile in the "User Registration" section
    2. Upload food images in the "Food Analysis" section
    3. View your personalized recommendations
    4. Track your meal history and patterns in the "History" section
    
    Let's start your journey to better health through smart eating!
    """)

def render_registration_form():
    """Render the user registration form"""
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
    
    login_col1, login_col2 = st.columns([3, 1])
    
    with login_col1:
        existing_user_id = st.text_input("Enter your User ID", key="login_user_id")
    
    with login_col2:
        login_button = st.button("Login")
    
    if login_button:
        if not existing_user_id:
            st.error("Please enter a User ID")
        else:
            with st.spinner("Logging in..."):
                user_profile = get_user(existing_user_id)
                
                if user_profile and 'profile' in user_profile:
                    st.session_state.user_id = existing_user_id
                    st.session_state.user_profile = user_profile['profile']
                    st.success("Login successful!")
                    # Use a separate key for forcing rerun to avoid the warning
                    st.session_state.force_rerun = True
                    st.rerun()
                else:
                    st.error("User not found. Please check your User ID and make sure the backend server is running.")

def render_profile_update_form():
    """Render the profile update form for logged-in users"""
    user_profile = st.session_state.user_profile
    
    st.success(f"You are registered with User ID: {st.session_state.user_id}")
    
    with st.expander("View/Update Profile", expanded=True):
        st.markdown("You can update your profile information below:")
        
        update_col1, update_col2 = st.columns(2)
        
        with update_col1:
            name = st.text_input("Full Name", value=user_profile.get('name', ''))
            age = st.number_input("Age", min_value=18, max_value=120, value=user_profile.get('age', 35))
            gender = st.selectbox("Gender", ["male", "female", "other"], 
                                  index=["male", "female", "other"].index(user_profile.get('gender', 'male')))
            height = st.number_input("Height (cm)", min_value=100, max_value=250, 
                                     value=user_profile.get('height', 170))
            weight = st.number_input("Weight (kg)", min_value=30, max_value=300, 
                                     value=user_profile.get('weight', 70))
        
        with update_col2:
            activity_level = st.selectbox("Activity Level", 
                ["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"],
                index=["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"].index(
                    user_profile.get('activity_level', 'moderately_active'))
            )
            diabetes_status = st.selectbox("Diabetes Status", 
                ["none", "pre_diabetic", "type1_diabetes", "type2_diabetes"],
                index=["none", "pre_diabetic", "type1_diabetes", "type2_diabetes"].index(
                    user_profile.get('diabetes_status', 'none'))
            )
            weight_goal = st.selectbox("Weight Goal", ["lose", "maintain", "gain"],
                                      index=["lose", "maintain", "gain"].index(
                                          user_profile.get('weight_goal', 'maintain'))
                                      )
            
            # Optional fields
            hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=15.0, 
                                   value=user_profile.get('hba1c', 0.0),
                                   help="Leave at 0 if unknown")
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=0, max_value=300, 
                                             value=user_profile.get('fasting_glucose', 0),
                                             help="Leave at 0 if unknown")
        
        update_button = st.button("Update Profile")
        
        if update_button:
            with st.spinner("Updating your profile..."):
                updated_data = {
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
                    updated_data["hba1c"] = hba1c
                if fasting_glucose > 0:
                    updated_data["fasting_glucose"] = fasting_glucose
                
                result = update_user(st.session_state.user_id, updated_data)
                
                if result is not None:
                    # Update the session state with the new profile data
                    st.session_state.user_profile = updated_data
                    st.success("Profile updated successfully!")
                else:
                    st.error("Failed to update profile. Please try again or check the API connection.")
    
    if st.button("Log Out"):
        st.session_state.user_id = None
        st.session_state.user_profile = None
        st.session_state.analysis_result = None
        st.session_state.meal_history_data = None
        st.session_state.meal_stats_data = None
        st.session_state.selected_meal = None
        st.session_state.history_view = "list"
        st.session_state.feedback_submitted = False
        st.rerun()

def render_food_analysis_page():
    """Render the food analysis page"""
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

def render_meal_card(meal):
    """Render a meal card with improved styling and image handling."""
    try:
        with st.container():
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Display meal image with improved handling
                if 'image_path' in meal and meal['image_path']:
                    # Extract the filename from the path
                    filename = os.path.basename(meal['image_path'])
                    # Get the image URL
                    image_url = get_image_url(filename)
                    
                    # Log for debugging
                    logger.info(f"Loading meal image from: {image_url}")
                    
                    # Create a container with border styling
                    with st.container():
                        # Display the image using st.image
                        try:
                            st.image(image_url, width=150)
                        except Exception as e:
                            logger.error(f"Error displaying image {filename}: {e}")
                            # Fall back to placeholder
                            display_meal_image(meal)
                else:
                    # Use placeholder if no image path
                    display_meal_image(meal)
            
            with col2:
                # Display date and foods with better formatting
                st.markdown(f"**{format_timestamp(meal['timestamp'])}**")
                
                # Format foods list nicely
                foods_list = meal.get('foods', ['Unknown'])
                st.markdown(f"**Foods:** {', '.join(foods_list)}")
                
                # Display nutrients if available
                if 'total_nutrients' in meal and meal['total_nutrients']:
                    nutrients = meal['total_nutrients']
                    if 'calories' in nutrients:
                        st.markdown(f"**Calories:** {nutrients['calories']:.1f} kcal")
                    
                    # Show macronutrients if available
                    macros = []
                    if 'carbs' in nutrients:
                        macros.append(f"Carbs: {nutrients['carbs']:.1f}g")
                    if 'protein' in nutrients:
                        macros.append(f"Protein: {nutrients['protein']:.1f}g")
                    if 'fat' in nutrients:
                        macros.append(f"Fat: {nutrients['fat']:.1f}g")
                    
                    if macros:
                        st.markdown(f"**Macros:** {' | '.join(macros)}")
            
            with col3:
                # Display glucose prediction with improved color-coding
                if 'predicted_glucose' in meal and meal['predicted_glucose']:
                    glucose_value = meal['predicted_glucose']
                    category = meal.get('glucose_category', 'Unknown')
                    
                    # Define colors for different categories
                    category_color = {
                        'Low': 'red',
                        'Normal': 'green',
                        'Elevated': 'orange',
                        'High': 'red'
                    }.get(category, 'blue')
                    
                    # Display glucose info with color
                    st.markdown(f"**Glucose:** <span style='color:{category_color};font-weight:bold;'>{glucose_value:.1f} mg/dL</span>", unsafe_allow_html=True)
                    st.markdown(f"**Category:** <span style='color:{category_color};font-weight:bold;'>{category}</span>", unsafe_allow_html=True)
                
                # View details button with improved styling
                if st.button(f"View Details", key=f"view_meal_{meal.get('meal_id')}", 
                          help="Click to see detailed information about this meal"):
                    st.session_state.selected_meal = meal.get('meal_id')
                    st.session_state.history_view = "detail"
                    st.session_state.feedback_submitted = False
                    st.rerun()
    except Exception as e:
        # Catch any errors in meal card rendering
        logger.error(f"Error rendering meal card: {e}")
        st.error(f"Error displaying meal: {e}")

def display_meal_detail(meal_detail, user_id):
    """Display an enhanced meal detail view with better styling and image handling."""
    if not meal_detail or 'meal' not in meal_detail:
        st.error("Failed to load meal details.")
        
        # Back button
        if st.button("← Back to Meal List"):
            st.session_state.history_view = "list"
            st.session_state.selected_meal = None
            st.rerun()
        return
    
    meal = meal_detail['meal']
    
    # Back button at the top
    if st.button("← Back to Meal List", key="back_button_top"):
        st.session_state.history_view = "list"
        st.session_state.selected_meal = None
        st.rerun()
    
    # Header with meal timestamp
    st.header(f"Meal Details: {format_timestamp(meal['timestamp'])}")
    st.divider()
    
    # Meal detail layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display meal image with better handling
        st.subheader("Meal Image")
        
        try:
            if 'image_path' in meal and meal['image_path']:
                # Extract the filename from the path
                filename = os.path.basename(meal['image_path'])
                # Get the image URL
                image_url = get_image_url(filename)
                
                # Log for debugging
                logger.info(f"Loading meal detail image from: {image_url}")
                
                # Display the image
                try:
                    st.image(image_url, width=300)
                except Exception as e:
                    logger.error(f"Error displaying detail image {filename}: {e}")
                    # Fall back to placeholder
                    display_meal_image(meal, width=300)
            else:
                # Use placeholder if no image path
                display_meal_image(meal, width=300)
        except Exception as e:
            logger.error(f"Error with meal detail image: {e}")
            # Final fallback
            st.error("Could not load meal image")
            display_meal_image(meal, width=300)
        
        # Display food items
        st.subheader("Food Items")
        for food in meal.get('food_items', []):
            # Create an expandable section for each food
            with st.expander(f"**{food['food_name']}**"):
                # Display GI information
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Base GI", f"{food.get('base_gi', 'N/A')}")
                with cols[1]:
                    if 'personalized_gi' in food:
                        pers_gi = food['personalized_gi']
                        st.metric("Personalized GI", f"{pers_gi.get('personalized_gi_score', 'N/A')}")
                
                # Display impact level with color
                if 'personalized_gi' in food:
                    pers_gi = food['personalized_gi']
                    impact_level = pers_gi.get('impact_level', '').capitalize()
                    impact_color = "green"
                    if impact_level == "Medium":
                        impact_color = "orange"
                    elif impact_level == "High":
                        impact_color = "red"
                    
                    st.markdown(f"**Impact Level:** <span style='color:{impact_color};'>{impact_level}</span>", 
                                unsafe_allow_html=True)
                    st.markdown(f"**Warning Level:** {pers_gi.get('warning_level', 'N/A')}")
                    st.markdown(f"**Recommended Portion:** {pers_gi.get('recommended_portion', 'N/A')}")
                
                # Display nutrients
                if 'nutrients' in food:
                    st.subheader("Nutrients")
                    nutrient_data = []
                    for nutrient, value in food['nutrients'].items():
                        # Convert nutrient name to proper case and format value
                        nutrient_name = nutrient.capitalize()
                        if isinstance(value, (int, float)):
                            nutrient_value = f"{value:.1f}g"
                        else:
                            nutrient_value = str(value)
                        nutrient_data.append({"Nutrient": nutrient_name, "Amount": nutrient_value})
                    
                    # Display as a nicely formatted table
                    if nutrient_data:
                        st.table(pd.DataFrame(nutrient_data))
    
    with col2:
        # Display glucose prediction with enhanced visualization
        st.subheader("Glucose Impact")
        if 'glucose_prediction' in meal:
            prediction = meal['glucose_prediction']
            
            # Display metrics in a row
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                glucose_value = prediction.get('predicted_glucose', 0)
                st.metric("Peak Glucose", f"{glucose_value:.1f} mg/dL")
            
            with metric_cols[1]:
                category = prediction.get('glucose_category', 'Unknown')
                category_color = {
                    'Low': 'red',
                    'Normal': 'green',
                    'Elevated': 'orange',
                    'High': 'red'
                }.get(category, 'blue')
                
                # Use HTML for colored text
                st.markdown(f"**Category:**  \n<span style='color:{category_color};font-size:1.2em;font-weight:bold;'>{category}</span>", 
                            unsafe_allow_html=True)
            
            with metric_cols[2]:
                max_time = prediction.get('max_glucose_time', 0)
                if max_time > 0:
                    st.metric("Time to Peak", f"{max_time} min")
            
            # Display glucose graph if we can create one
            fig = create_glucose_graph(prediction)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display guidelines and recommendations
            if 'guidelines' in prediction or 'recommendations' in prediction:
                st.subheader("Personalized Recommendations")
                
                if 'guidelines' in prediction and prediction['guidelines']:
                    st.markdown("**Guidelines:**")
                    for guideline in prediction['guidelines']:
                        st.markdown(f"- {guideline}")
                
                if 'recommendations' in prediction and prediction['recommendations']:
                    st.markdown("**Specific Recommendations:**")
                    for rec in prediction['recommendations']:
                        st.markdown(f"- {rec}")
        
        # Display total nutritional content
        st.subheader("Total Nutritional Content")
        if 'total_nutrients' in meal:
            nutrients = meal['total_nutrients']
            
            # Display calories as a metric
            if 'calories' in nutrients:
                st.metric("Total Calories", f"{nutrients['calories']:.1f} kcal")
            
            # Display macronutrient breakdown
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
            
            # Display nutrient chart
            fig = get_nutrient_chart(nutrients)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Display feedback section
        st.subheader("Meal Feedback")
        
        # Show existing feedback if available
        if 'feedback' in meal:
            feedback = meal['feedback']
            st.success(f"Feedback submitted on {format_timestamp(feedback['timestamp'])}")
            
            if 'user_response' in feedback and feedback['user_response']:
                response_map = {
                    'less_than_expected': 'Lower than expected',
                    'as_expected': 'As expected',
                    'more_than_expected': 'Higher than expected'
                }
                response_text = response_map.get(feedback['user_response'], feedback['user_response'])
                
                # Show with appropriate color
                response_color = 'green'
                if feedback['user_response'] == 'more_than_expected':
                    response_color = 'red'
                elif feedback['user_response'] == 'less_than_expected':
                    response_color = 'orange'
                
                st.markdown(f"**Glucose Impact:** <span style='color:{response_color};'>{response_text}</span>", 
                            unsafe_allow_html=True)
            
            if 'notes' in feedback and feedback['notes']:
                st.markdown(f"**Notes:** {feedback['notes']}")


def display_meal_detail(meal_detail, user_id):
    """Display an enhanced meal detail view with better styling and image handling."""
    try:
        if not meal_detail or 'meal' not in meal_detail:
            st.error("Failed to load meal details.")

            # Back button
            if st.button("← Back to Meal List"):
                st.session_state.history_view = "list"
                st.session_state.selected_meal = None
                st.rerun()
            return

        meal = meal_detail['meal']

        # Back button at the top
        if st.button("← Back to Meal List", key="back_button_top"):
            st.session_state.history_view = "list"
            st.session_state.selected_meal = None
            st.rerun()

        # Header with meal timestamp
        st.header(f"Meal Details: {format_timestamp(meal['timestamp'])}")
        st.divider()

        # Meal detail layout
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display meal image with better handling
            st.subheader("Meal Image")

            try:
                if 'image_path' in meal and meal['image_path']:
                    # Extract the filename from the path
                    filename = os.path.basename(meal['image_path'])
                    # Get the image URL
                    image_url = get_image_url(filename)

                    # Log for debugging
                    logger.info(f"Loading meal detail image from: {image_url}")

                    # Display the image
                    st.image(image_url, width=300)
                else:
                    # Use placeholder if no image path
                    display_meal_image(meal, width=300)
            except Exception as e:
                logger.error(f"Error with meal detail image: {e}")
                # Final fallback
                st.error("Could not load meal image")
                display_meal_image(meal, width=300)

            # Display food items
            st.subheader("Food Items")
            for food in meal.get('food_items', []):
                # Create an expandable section for each food
                with st.expander(f"{food['food_name']}"):
                    # Display GI information
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Base GI", f"{food.get('base_gi', 'N/A')}")
                    with cols[1]:
                        if 'personalized_gi' in food:
                            pers_gi = food['personalized_gi']
                            st.metric("Personalized GI", f"{pers_gi.get('personalized_gi_score', 'N/A')}")

                    # Display impact level with color
                    if 'personalized_gi' in food:
                        pers_gi = food['personalized_gi']
                        impact_level = pers_gi.get('impact_level', '').capitalize()
                        impact_color = "green"
                        if impact_level == "Medium":
                            impact_color = "orange"
                        elif impact_level == "High":
                            impact_color = "red"

                        st.markdown(f"**Impact Level:** <span style='color:{impact_color};'>{impact_level}</span>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**Warning Level:** {pers_gi.get('warning_level', 'N/A')}")
                        st.markdown(f"**Recommended Portion:** {pers_gi.get('recommended_portion', 'N/A')}")

                    # Display nutrients
                    if 'nutrients' in food:
                        st.subheader("Nutrients")
                        nutrient_data = []
                        for nutrient, value in food['nutrients'].items():
                            # Convert nutrient name to proper case and format value
                            nutrient_name = nutrient.capitalize()
                            if isinstance(value, (int, float)):
                                nutrient_value = f"{value:.1f}g"
                            else:
                                nutrient_value = str(value)
                            nutrient_data.append({"Nutrient": nutrient_name, "Amount": nutrient_value})

                        # Display as a nicely formatted table
                        if nutrient_data:
                            st.table(pd.DataFrame(nutrient_data))

        with col2:
            # Rest of the detail view code
            # [rest of the function content...]

            # Display glucose prediction with enhanced visualization
            st.subheader("Glucose Impact")
            if 'glucose_prediction' in meal:
                prediction = meal['glucose_prediction']

                # Display metrics in a row
                metric_cols = st.columns(3)

                with metric_cols[0]:
                    glucose_value = prediction.get('predicted_glucose', 0)
                    st.metric("Peak Glucose", f"{glucose_value:.1f} mg/dL")

                with metric_cols[1]:
                    category = prediction.get('glucose_category', 'Unknown')
                    category_color = {
                        'Low': 'red',
                        'Normal': 'green',
                        'Elevated': 'orange',
                        'High': 'red'
                    }.get(category, 'blue')

                    # Use HTML for colored text
                    st.markdown(f"**Category:**  \n<span style='color:{category_color};font-size:1.2em;font-weight:bold;'>{category}</span>",
                                unsafe_allow_html=True)

                with metric_cols[2]:
                    max_time = prediction.get('max_glucose_time', 0)
                    if max_time > 0:
                        st.metric("Time to Peak", f"{max_time} min")

                # Display glucose graph if we can create one
                fig = create_glucose_graph(prediction)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Display guidelines and recommendations
                if 'guidelines' in prediction or 'recommendations' in prediction:
                    st.subheader("Personalized Recommendations")

                    if 'guidelines' in prediction and prediction['guidelines']:
                        st.markdown("**Guidelines:**")
                        for guideline in prediction['guidelines']:
                            st.markdown(f"- {guideline}")

                    if 'recommendations' in prediction and prediction['recommendations']:
                        st.markdown("**Specific Recommendations:**")
                        for rec in prediction['recommendations']:
                            st.markdown(f"- {rec}")

            # Display nutritional content
            # [rest of the nutritional content code...]

    except Exception as e:
        st.error(f"Error displaying meal detail: {e}")
        import traceback
        st.text(traceback.format_exc())
        logger.error(f"Error in display_meal_detail: {e}")
        logger.error(traceback.format_exc())

        # Back button for error recovery
        if st.button("← Return to Meal List", key="error_back_button"):
            st.session_state.history_view = "list"
            st.session_state.selected_meal = None
            st.rerun()

def render_meal_history_tab():
    """Render the meal history tab with filters and meal list/detail views"""
    try:
        # Add filter controls in a form
        with st.form(key="meal_history_filters"):
            st.subheader("Filter Meals")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                start_date = st.date_input("Start Date", value=None)
            with col2:
                end_date = st.date_input("End Date", value=None)
            with col3:
                food_type = st.text_input("Food Type (optional)")
            
            col4, col5 = st.columns(2)
            with col4:
                sort_by = st.selectbox("Sort By", ["date", "glucose_impact"], index=0)
            with col5:
                sort_order = st.selectbox("Sort Order", ["desc", "asc"], index=0)
            
            filter_submit = st.form_submit_button("Apply Filters")
        
        if filter_submit or st.session_state.meal_history_data is None:
            # Format dates for API call
            start_date_str = start_date.isoformat() if start_date else None
            end_date_str = end_date.isoformat() if end_date else None
            
            # Fetch meal history with filters
            with st.spinner("Loading meal history..."):
                history_data = get_meal_history(
                    st.session_state.user_id,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    food_type=food_type if food_type else None,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
                
                if history_data:
                    st.session_state.meal_history_data = history_data
                    logger.info(f"Loaded {len(history_data.get('meals', []))} meals for user {st.session_state.user_id}")
                else:
                    st.error("Failed to load meal history. Please check the API connection.")
        
        # Display meal history
        if st.session_state.meal_history_data:
            meals = st.session_state.meal_history_data.get('meals', [])
            
            if not meals:
                st.info("No meals found with the selected filters.")
            else:
                st.subheader(f"Found {len(meals)} meals")
                
                # Detail view for a specific meal
                if st.session_state.history_view == "detail" and st.session_state.selected_meal:
                    meal_detail = get_meal_details(st.session_state.user_id, st.session_state.selected_meal)
                    if meal_detail:
                        display_meal_detail(meal_detail, st.session_state.user_id)
                    else:
                        st.error("Failed to load meal details")
                        st.session_state.history_view = "list"
                        st.session_state.selected_meal = None
                else:
                    # Add a search box for quick filtering
                    search_term = st.text_input("Search meals by food name", "")
                    
                    # Create a table for meal history
                    for i, meal in enumerate(meals):
                        # Apply search filter if provided
                        if search_term and not any(search_term.lower() in food.lower() for food in meal.get('foods', [])):
                            continue
                            
                        # Render the meal card
                        render_meal_card(meal)
                    
                    # Pagination (simplified)
                    if len(meals) >= 50:  # If we have the maximum number of results
                        st.markdown("Note: Showing up to 50 meals. Use filters to narrow down results.")
        else:
            st.info("No meal history data found. Try analyzing some meals first.")
            
            # Demo button for first-time users
            if st.button("Analyze a Food Image"):
                st.session_state.page = "Food Analysis"
                st.rerun()
    except Exception as e:
        st.error(f"Error in meal history tab: {e}")
        import traceback
        st.text(traceback.format_exc())
        logger.error(f"Error in render_meal_history_tab: {e}")
        logger.error(traceback.format_exc())

def render_meal_history_page():
    """Render the meal history page with tabs for history and statistics"""
    try:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Meal History", "Statistics"])
        
        with tab1:
            render_meal_history_tab()
        
        # Statistics tab
        with tab2:
            render_statistics_tab()
    except Exception as e:
        st.error(f"Error rendering meal history page: {e}")
        import traceback
        st.text(traceback.format_exc())
        logger.error(f"Error in render_meal_history_page: {e}")
        logger.error(traceback.format_exc())
def render_statistics_tab():
    """Render the statistics tab with charts and insights"""
    # Fetch meal statistics if not already loaded
    if st.session_state.meal_stats_data is None:
        with st.spinner("Loading meal statistics..."):
            stats_data = get_meal_stats(st.session_state.user_id)
            if stats_data:
                st.session_state.meal_stats_data = stats_data
    
    # Display statistics
    if st.session_state.meal_stats_data:
        stats = st.session_state.meal_stats_data
        
        if stats.get('total_meals', 0) == 0:
            st.info("No meal data available for statistics.")
        else:
            # Display summary metrics
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Meals", stats.get('total_meals', 0))
            
            with col2:
                avg_glucose = stats.get('stats', {}).get('avg_glucose', 0)
                st.metric("Average Glucose Impact", f"{avg_glucose:.1f} mg/dL")
            
            with col3:
                # Calculate percentage of normal glucose responses
                categories = stats.get('stats', {}).get('glucose_categories', {})
                normal_count = categories.get('Normal', 0)
                total = sum(categories.values()) if categories else 0
                
                if total > 0:
                    normal_pct = (normal_count / total) * 100
                    st.metric("Normal Glucose Responses", f"{normal_pct:.1f}%")
            
            # Display glucose category distribution
            st.subheader("Glucose Response Categories")
            categories = stats.get('stats', {}).get('glucose_categories', {})
            
            if categories:
                # Create data for plotting
                category_names = list(categories.keys())
                category_counts = list(categories.values())
                
                # Create pie chart
                fig = px.pie(
                    values=category_counts,
                    names=category_names,
                    color=category_names,
                    color_discrete_map={
                        'Low': 'red',
                        'Normal': 'green',
                        'Elevated': 'orange',
                        'High': 'darkred'
                    },
                    title="Glucose Response Distribution"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Display top foods
            st.subheader("Most Frequently Consumed Foods")
            top_foods = stats.get('stats', {}).get('top_foods', [])
            
            if top_foods:
                # Create bar chart for top foods
                food_names = [food['name'].capitalize() for food in top_foods]
                food_counts = [food['count'] for food in top_foods]
                
                fig = px.bar(
                    x=food_names,
                    y=food_counts,
                    labels={"x": "Food", "y": "Count"},
                    title="Most Common Foods",
                    color=food_names,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display problem foods
            st.subheader("Foods with Highest Glucose Impact")
            problem_foods = stats.get('stats', {}).get('problem_foods', [])
            
            if problem_foods:
                # Create bar chart for problem foods
                problem_food_names = [food['name'].capitalize() for food in problem_foods]
                problem_food_glucose = [food['avg_glucose'] for food in problem_foods]
                
                fig = px.bar(
                    x=problem_food_names,
                    y=problem_food_glucose,
                    labels={"x": "Food", "y": "Average Glucose (mg/dL)"},
                    title="Foods with Highest Glucose Impact",
                    color=problem_food_names,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                # Add a line for average glucose
                avg_glucose = stats.get('stats', {}).get('avg_glucose', 0)
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(problem_food_names) - 0.5,
                    y0=avg_glucose,
                    y1=avg_glucose,
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=len(problem_food_names) - 1,
                    y=avg_glucose + 5,
                    text=f"Avg: {avg_glucose:.1f}",
                    showarrow=False,
                    font=dict(color="red")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display weekly trends
            st.subheader("Glucose Trends Over Time")
            weekly_trends = stats.get('stats', {}).get('weekly_trends', [])
            
            if weekly_trends:
                # Create line chart for weekly trends
                weeks = [format_date(trend['week']) for trend in weekly_trends]
                avg_glucose_values = [trend['avg_glucose'] for trend in weekly_trends]
                meal_counts = [trend['meal_count'] for trend in weekly_trends]
                
                # Create figure with two y-axes
                fig = go.Figure()
                
                # Add traces
                fig.add_trace(
                    go.Scatter(
                        x=weeks,
                        y=avg_glucose_values,
                        name="Average Glucose",
                        line=dict(color='blue', width=3)
                    )
                )
                
                fig.add_trace(
                    go.Bar(
                        x=weeks,
                        y=meal_counts,
                        name="Meal Count",
                        marker_color='lightblue',
                        opacity=0.5,
                        yaxis="y2"
                    )
                )
                
                # Set up layout with two y-axes
                fig.update_layout(
                    title="Weekly Glucose Trends",
                    xaxis_title="Week",
                    yaxis=dict(
                        title="Average Glucose (mg/dL)",
                        tickfont=dict(color="blue")
                    ),
                    yaxis2=dict(
                        title="Meal Count",
                        tickfont=dict(color="lightblue"),
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on statistics
            st.subheader("Personalized Recommendations")
            
            recommendations = []
            
            # Add recommendations based on problem foods
            if problem_foods:
                recommendations.append(f"Consider limiting consumption of **{problem_foods[0]['name'].capitalize()}** as it has the highest glucose impact.")
            
            # Add recommendations based on glucose categories
            if categories:
                high_count = categories.get('High', 0) + categories.get('Elevated', 0)
                total = sum(categories.values())
                
                if total > 0 and (high_count / total) > 0.3:
                    recommendations.append("Over 30% of your meals cause elevated glucose responses. Consider balancing your meals with more protein and fiber.")
            
            # Add recommendations based on weekly trends
            if weekly_trends and len(weekly_trends) >= 2:
                latest_week = weekly_trends[-1]
                previous_week = weekly_trends[-2]
                
                if latest_week['avg_glucose'] > previous_week['avg_glucose'] * 1.1:
                    recommendations.append("Your glucose responses have increased in the last week. Review recent dietary changes.")
                elif latest_week['avg_glucose'] < previous_week['avg_glucose'] * 0.9:
                    recommendations.append("Great job! Your average glucose response has improved compared to the previous week.")
            
            # Display recommendations
            if recommendations:
                for i, rec in enumerate(recommendations):
                    st.markdown(f"{i+1}. {rec}")
            else:
                st.info("Not enough data to generate personalized recommendations yet. Continue tracking your meals.")
            
            # Add refresh button
            if st.button("Refresh Statistics"):
                st.session_state.meal_stats_data = None
                st.rerun()
    else:
        st.info("No meal statistics available. Try analyzing some meals first.")
        
        # Demo button for first-time users
        if st.button("Analyze a Food Image"):
            st.session_state.page = "Food Analysis"
            st.rerun()
