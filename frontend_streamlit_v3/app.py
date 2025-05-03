import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional, Tuple

from api_client import NutritionAPI, UserData
from helpers import calculate_nutritional_needs, format_nutrition_data, generate_meal_plan
from ui_components import display_header, create_sidebar, render_nutrition_chart, display_food_card

API_ENDPOINT = os.getenv("API_ENDPOINT", "https://nutrition-api.example.com/v1")
API_KEY = os.getenv("API_KEY", "")
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "https://user-service.example.com/api")

def initialize_session_state():
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
    
    if "nutrition_api" not in st.session_state:
        st.session_state.nutrition_api = NutritionAPI(API_ENDPOINT, API_KEY)
    
    if "user_service" not in st.session_state:
        st.session_state.user_service = UserData(USER_SERVICE_URL)
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    
    if "meal_history" not in st.session_state:
        st.session_state.meal_history = []
    
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Dashboard"

def login_page():
    st.title("Nutrition Tracker - Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            auth_result = st.session_state.user_service.authenticate(username, password)
            
            if "error" in auth_result:
                st.error(f"Authentication failed: {auth_result['error']}")
            else:
                st.session_state.user_authenticated = True
                st.session_state.user_profile = st.session_state.user_service.get_user_profile()
                st.session_state.meal_history = st.session_state.user_service.get_meal_history()
                st.success("Login successful!")
                st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("Don't have an account? Sign up feature coming soon!")

def dashboard_page():
    display_header("Nutrition Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.meal_history:
            recent_meals = st.session_state.meal_history[:5]
            
            st.subheader("Recent Meals")
            for meal in recent_meals:
                with st.expander(f"{meal['name']} - {meal['date']}"):
                    st.write(f"Total Calories: {meal['nutrition']['calories']} kcal")
                    st.write(f"Protein: {meal['nutrition']['protein']}g")
                    st.write(f"Carbs: {meal['nutrition']['carbohydrates']}g")
                    st.write(f"Fat: {meal['nutrition']['fat']}g")
        else:
            st.info("No meal history available. Start logging your meals!")
    
    with col2:
        st.subheader("Nutritional Goals")
        user_needs = calculate_nutritional_needs(
            st.session_state.user_profile.get("weight", 70),
            st.session_state.user_profile.get("height", 170),
            st.session_state.user_profile.get("age", 30),
            st.session_state.user_profile.get("gender", "male"),
            st.session_state.user_profile.get("activity_level", "moderate")
        )
        
        st.metric("Daily Calorie Goal", f"{user_needs['calories']} kcal")
        st.metric("Protein Goal", f"{user_needs['protein']}g")
        st.metric("Carbs Goal", f"{user_needs['carbohydrates']}g")
        st.metric("Fat Goal", f"{user_needs['fat']}g")
    
    st.markdown("---")
    
    st.subheader("Nutrition Trends")
    if len(st.session_state.meal_history) > 3:
        dates = []
        calories = []
        proteins = []
        carbs = []
        fats = []
        
        for meal in st.session_state.meal_history:
            meal_date = datetime.strptime(meal['date'], "%Y-%m-%d").date()
            
            if meal_date not in dates:
                dates.append(meal_date)
                calories.append(meal['nutrition']['calories'])
                proteins.append(meal['nutrition']['protein'])
                carbs.append(meal['nutrition']['carbohydrates'])
                fats.append(meal['nutrition']['fat'])
            else:
                idx = dates.index(meal_date)
                calories[idx] += meal['nutrition']['calories']
                proteins[idx] += meal['nutrition']['protein']
                carbs[idx] += meal['nutrition']['carbohydrates']
                fats[idx] += meal['nutrition']['fat']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(dates, calories, marker='o')
        ax1.set_title("Calorie Intake Over Time")
        ax1.set_ylabel("Calories (kcal)")
        ax1.grid(True)
        
        ax2.plot(dates, proteins, marker='o', label='Protein')
        ax2.plot(dates, carbs, marker='o', label='Carbs')
        ax2.plot(dates, fats, marker='o', label='Fat')
        ax2.set_title("Macronutrient Intake Over Time")
        ax2.set_ylabel("Amount (g)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough meal data to show trends. Log more meals to see your nutrition trends.")

def food_search_page():
    display_header("Food Search")
    
    search_query = st.text_input("Search for food", "")
    
    if search_query:
        with st.spinner("Searching for food..."):
            search_results = st.session_state.nutrition_api.search_food_items(search_query)
        
        if search_results and not (len(search_results) == 1 and "error" in search_results[0]):
            st.subheader("Search Results")
            
            for result in search_results:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{result['name']}**")
                    st.write(f"Calories: {result['calories']} kcal | Protein: {result['protein']}g | Carbs: {result['carbohydrates']}g | Fat: {result['fat']}g")
                
                with col2:
                    if st.button("View Details", key=f"details_{result['id']}"):
                        food_details = st.session_state.nutrition_api.get_food_details(result['id'])
                        st.session_state.food_details = food_details
                        st.experimental_rerun()
                
                with col3:
                    if st.button("Add to Meal", key=f"add_{result['id']}"):
                        st.session_state.selected_food = result
                        st.session_state.current_tab = "Meal Planner"
                        st.experimental_rerun()
                
                st.markdown("---")
        elif search_results and "error" in search_results[0]:
            st.error(f"Error searching for food: {search_results[0]['error']}")
        else:
            st.info(f"No results found for '{search_query}'")
    
    if "food_details" in st.session_state:
        st.subheader(f"Details for {st.session_state.food_details['name']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Nutrition Facts**")
            st.write(f"Serving Size: {st.session_state.food_details['serving_size']} {st.session_state.food_details['serving_unit']}")
            st.write(f"Calories: {st.session_state.food_details['calories']} kcal")
            st.write(f"Protein: {st.session_state.food_details['protein']}g")
            st.write(f"Carbohydrates: {st.session_state.food_details['carbohydrates']}g")
            st.write(f"Fat: {st.session_state.food_details['fat']}g")
            st.write(f"Fiber: {st.session_state.food_details.get('fiber', 0)}g")
            st.write(f"Sugar: {st.session_state.food_details.get('sugar', 0)}g")
        
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            
            nutrition = [
                st.session_state.food_details['protein'],
                st.session_state.food_details['carbohydrates'],
                st.session_state.food_details['fat']
            ]
            
            labels = ['Protein', 'Carbs', 'Fat']
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            ax.pie(nutrition, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title("Macronutrient Distribution")
            
            st.pyplot(fig)
        
        if st.button("Add to Meal"):
            st.session_state.selected_food = st.session_state.food_details
            st.session_state.current_tab = "Meal Planner"
            st.experimental_rerun()

def meal_planner_page():
    display_header("Meal Planner")
    
    if "meal_items" not in st.session_state:
        st.session_state.meal_items = []
    
    if "selected_food" in st.session_state:
        food = st.session_state.selected_food
        
        st.subheader("Add to Meal")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{food['name']}**")
        
        with col2:
            serving_size = st.number_input("Serving Size", min_value=0.1, value=1.0, step=0.1, key="serving_size")
        
        with col3:
            if st.button("Add"):
                meal_item = {
                    "id": food['id'],
                    "name": food['name'],
                    "serving_size": serving_size,
                    "calories": food['calories'] * serving_size,
                    "protein": food['protein'] * serving_size,
                    "carbohydrates": food['carbohydrates'] * serving_size,
                    "fat": food['fat'] * serving_size
                }
                
                st.session_state.meal_items.append(meal_item)
                st.session_state.pop("selected_food", None)
                st.experimental_rerun()
    
    if st.session_state.meal_items:
        st.subheader("Current Meal")
        
        total_calories = sum(item['calories'] for item in st.session_state.meal_items)
        total_protein = sum(item['protein'] for item in st.session_state.meal_items)
        total_carbs = sum(item['carbohydrates'] for item in st.session_state.meal_items)
        total_fat = sum(item['fat'] for item in st.session_state.meal_items)
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Calories", f"{total_calories:.1f} kcal")
        col2.metric("Protein", f"{total_protein:.1f}g")
        col3.metric("Carbs", f"{total_carbs:.1f}g")
        col4.metric("Fat", f"{total_fat:.1f}g")
        
        for i, item in enumerate(st.session_state.meal_items):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{item['name']}** ({item['serving_size']} servings)")
                st.write(f"Calories: {item['calories']:.1f} kcal | Protein: {item['protein']:.1f}g | Carbs: {item['carbohydrates']:.1f}g | Fat: {item['fat']:.1f}g")
            
            with col3:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.meal_items.pop(i)
                    st.experimental_rerun()
            
            st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            meal_name = st.text_input("Meal Name", "My Meal")
            meal_date = st.date_input("Date", datetime.now())
            meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
        
        with col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            
            nutrition = [total_protein, total_carbs, total_fat]
            labels = ['Protein', 'Carbs', 'Fat']
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            ax.pie(nutrition, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title("Macronutrient Distribution")
            
            st.pyplot(fig)
        
        if st.button("Save Meal"):
            meal_data = {
                "name": meal_name,
                "date": meal_date.strftime("%Y-%m-%d"),
                "type": meal_type,
                "items": st.session_state.meal_items,
                "nutrition": {
                    "calories": total_calories,
                    "protein": total_protein,
                    "carbohydrates": total_carbs,
                    "fat": total_fat
                }
            }
            
            result = st.session_state.user_service.save_meal_log(meal_data)
            
            if "error" in result:
                st.error(f"Error saving meal: {result['error']}")
            else:
                st.success("Meal saved successfully!")
                st.session_state.meal_items = []
                
                # Update meal history
                st.session_state.meal_history = st.session_state.user_service.get_meal_history()
                
                # Redirect to dashboard
                st.session_state.current_tab = "Dashboard"
                st.experimental_rerun()
    else:
        st.info("No items in your meal yet. Search for food to add to your meal.")
        
        if st.button("Search for Food"):
            st.session_state.current_tab = "Food Search"
            st.experimental_rerun()

def profile_page():
    display_header("User Profile")
    
    user = st.session_state.user_profile
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        with st.form("profile_form"):
            name = st.text_input("Name", user.get("name", ""))
            age = st.number_input("Age", min_value=1, max_value=120, value=user.get("age", 30))
            gender = st.selectbox("Gender", ["male", "female"], index=0 if user.get("gender", "male") == "male" else 1)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=user.get("weight", 70.0), step=0.1)
            height = st.number_input("Height (cm)", min_value=1.0, max_value=300.0, value=user.get("height", 170.0), step=0.1)
            activity_level = st.selectbox(
                "Activity Level",
                ["sedentary", "light", "moderate", "active", "very active"],
                index=["sedentary", "light", "moderate", "active", "very active"].index(user.get("activity_level", "moderate"))
            )
            
            dietary_preferences = st.multiselect(
                "Dietary Preferences",
                ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Paleo"],
                default=user.get("dietary_preferences", [])
            )
            
            food_allergies = st.multiselect(
                "Food Allergies",
                ["Nuts", "Shellfish", "Dairy", "Eggs", "Wheat", "Soy"],
                default=user.get("allergies", [])
            )
            
            save_button = st.form_submit_button("Save Profile")
            
            if save_button:
                updated_profile = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "weight": weight,
                    "height": height,
                    "activity_level": activity_level,
                    "dietary_preferences": dietary_preferences,
                    "allergies": food_allergies
                }
                
                result = st.session_state.user_service.update_user_profile(updated_profile)
                
                if "error" in result:
                    st.error(f"Error updating profile: {result['error']}")
                else:
                    st.session_state.user_profile = result
                    st.success("Profile updated successfully!")
    
    with col2:
        st.subheader("Nutritional Needs")
        
        user_needs = calculate_nutritional_needs(
            user.get("weight", 70),
            user.get("height", 170),
            user.get("age", 30),
            user.get("gender", "male"),
            user.get("activity_level", "moderate")
        )
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        nutrition_values = [
            user_needs["protein"],
            user_needs["carbohydrates"],
            user_needs["fat"]
        ]
        
        labels = ["Protein", "Carbohydrates", "Fat"]
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        ax.pie(nutrition_values, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title("Recommended Macronutrient Distribution")
        
        st.pyplot(fig)
        
        st.write(f"**Daily Calorie Goal:** {user_needs['calories']} kcal")
        st.write(f"**Protein:** {user_needs['protein']}g ({user_needs['protein'] * 4} kcal)")
        st.write(f"**Carbohydrates:** {user_needs['carbohydrates']}g ({user_needs['carbohydrates'] * 4} kcal)")
        st.write(f"**Fat:** {user_needs['fat']}g ({user_needs['fat'] * 9} kcal)")

def main():
    initialize_session_state()
    
    if not st.session_state.user_authenticated:
        login_page()
    else:
        create_sidebar()
        
        if st.session_state.current_tab == "Dashboard":
            dashboard_page()
        elif st.session_state.current_tab == "Food Search":
            food_search_page()
        elif st.session_state.current_tab == "Meal Planner":
            meal_planner_page()
        elif st.session_state.current_tab == "Profile":
            profile_page()
        elif st.session_state.current_tab == "Logout":
            st.session_state.user_authenticated = False
            st.session_state.user_profile = {}
            st.session_state.meal_history = []
            st.session_state.current_tab = "Dashboard"
            st.experimental_rerun()

if __name__ == "__main__":
    main()
