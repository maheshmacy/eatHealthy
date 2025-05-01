"""
EATSMART-AI Model Testing Script

This script allows you to test a pickled glucose prediction model with new meal data.
It supports testing both standard and user-specific models.

Usage:
    python test_model.py --model-path /path/to/model.pkl --user-id USER_ID
"""

import os
import sys
import json
import argparse
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Configure argument parser
parser = argparse.ArgumentParser(description='Test EATSMART-AI glucose prediction model')
parser.add_argument('--model-path', type=str, help='Path to the model pickle file')
parser.add_argument('--user-id', type=str, help='User ID for user-specific model')
parser.add_argument('--meal-json', type=str, help='JSON file containing test meal data')
parser.add_argument('--user-profile-json', type=str, help='JSON file containing user profile data')
parser.add_argument('--generate-samples', action='store_true', help='Generate sample meal data')
parser.add_argument('--output', type=str, default='prediction_results.json', help='Output file for results')

# Helper functions
def map_activity_level(activity_level):
    """Map activity level string to numerical value"""
    mapping = {
        'sedentary': 1,
        'lightly_active': 2,
        'moderately_active': 3,
        'very_active': 4,
        'extremely_active': 5
    }
    return mapping.get(activity_level, 3)

def get_diabetes_category(diabetes_status):
    """Map diabetes status to numerical category for model"""
    mapping = {
        'none': 0,
        'pre_diabetic': 1,
        'type1_diabetes': 2,
        'type2_diabetes': 2
    }
    return mapping.get(diabetes_status, 0)

def generate_sample_meals(count=5):
    """Generate sample meal data for testing"""
    meals = []
    
    # Common foods with estimated GI values
    foods = [
        {"name": "White Bread", "base_gi": 75, "portion_size": "1 slice", 
         "nutrients": {"calories": 80, "carbs": 15, "protein": 3, "fat": 1, "fiber": 1, "sugar": 2}},
        {"name": "Brown Rice", "base_gi": 50, "portion_size": "1 cup", 
         "nutrients": {"calories": 220, "carbs": 45, "protein": 5, "fat": 2, "fiber": 4, "sugar": 0}},
        {"name": "Apple", "base_gi": 35, "portion_size": "1 medium", 
         "nutrients": {"calories": 95, "carbs": 25, "protein": 0.5, "fat": 0.3, "fiber": 4, "sugar": 19}},
        {"name": "Chicken Breast", "base_gi": 0, "portion_size": "3 oz", 
         "nutrients": {"calories": 165, "carbs": 0, "protein": 31, "fat": 3.6, "fiber": 0, "sugar": 0}},
        {"name": "Sweet Potato", "base_gi": 63, "portion_size": "1 medium", 
         "nutrients": {"calories": 180, "carbs": 41, "protein": 4, "fat": 0.1, "fiber": 6, "sugar": 9}},
        {"name": "Salmon", "base_gi": 0, "portion_size": "4 oz", 
         "nutrients": {"calories": 233, "carbs": 0, "protein": 25, "fat": 15, "fiber": 0, "sugar": 0}},
        {"name": "Pasta", "base_gi": 65, "portion_size": "1 cup", 
         "nutrients": {"calories": 220, "carbs": 43, "protein": 8, "fat": 1.3, "fiber": 2.5, "sugar": 0.8}},
        {"name": "Broccoli", "base_gi": 15, "portion_size": "1 cup", 
         "nutrients": {"calories": 55, "carbs": 11, "protein": 3.7, "fat": 0.6, "fiber": 5, "sugar": 2.6}},
        {"name": "Orange Juice", "base_gi": 50, "portion_size": "1 cup", 
         "nutrients": {"calories": 112, "carbs": 26, "protein": 2, "fat": 0.5, "fiber": 0.5, "sugar": 21}},
        {"name": "Chocolate Cake", "base_gi": 38, "portion_size": "1 slice", 
         "nutrients": {"calories": 352, "carbs": 50, "protein": 5, "fat": 15, "fiber": 2, "sugar": 32}}
    ]
    
    # Generate sample meals by combining 2-4 foods
    for i in range(count):
        # Select random number of foods (2-4)
        num_foods = np.random.randint(2, 5)
        selected_foods = np.random.choice(foods, num_foods, replace=False)
        
        food_items = []
        total_nutrients = {
            "calories": 0,
            "carbs": 0,
            "fat": 0,
            "protein": 0,
            "fiber": 0,
            "sugar": 0
        }
        
        # Add selected foods to the meal
        for food in selected_foods:
            food_items.append(food)
            
            # Add to total nutrients
            for key in total_nutrients:
                if key in food["nutrients"]:
                    total_nutrients[key] += food["nutrients"][key]
        
        # Calculate meal GI (weighted average based on carb content)
        if total_nutrients["carbs"] > 0:
            avg_gi = sum(food["base_gi"] * food["nutrients"]["carbs"] 
                          for food in food_items) / total_nutrients["carbs"]
        else:
            avg_gi = 0
        
        # Calculate glycemic load
        gl = (avg_gi * total_nutrients["carbs"]) / 100
        
        # Generate random meal time
        hour = np.random.randint(6, 22)
        minute = np.random.choice([0, 15, 30, 45])
        timestamp = f"2025-04-30T{hour:02d}:{minute:02d}:00"
        
        # Create meal features
        meal_features = {
            "Age": 40,  # Will be overridden by user profile
            "Sex": 1,   # Will be overridden by user profile
            "BMI": 25,  # Will be overridden by user profile
            "Category": 0,  # Will be overridden by user profile
            "Carbs": total_nutrients["carbs"],
            "Fat": total_nutrients["fat"],
            "Fiber": total_nutrients["fiber"],
            "Protein": total_nutrients["protein"],
            "GI": avg_gi,
            "Glycemic_Load": gl,
            "Activity_Level": 3,  # Will be overridden by user profile
            "Activity_Timing": np.random.randint(0, 3),
            "Stress_Level": np.random.uniform(3.0, 8.0),
            "Sleep_Quality": np.random.uniform(4.0, 9.0),
            "Time_Since_Last_Meal": np.random.uniform(2.0, 6.0)
        }
        
        # Add Minutes_After_Meal for time series (will be used if model supports it)
        meal_features["Minutes_After_Meal"] = 0
        
        # Create meal data
        meal = {
            "meal_id": f"sample_meal_{i+1}",
            "timestamp": timestamp,
            "food_items": food_items,
            "total_nutrients": total_nutrients,
            "meal_features": meal_features
        }
        
        meals.append(meal)
    
    return meals

def default_user_profile():
    """Create a default user profile for testing"""
    return {
        "name": "Test User",
        "age": 40,
        "gender": "male",
        "height": 175,
        "weight": 75,
        "bmi": 24.5,
        "activity_level": "moderately_active",
        "diabetes_status": "none",
        "weight_goal": "maintain",
        "hba1c": None,
        "fasting_glucose": None
    }

def prepare_prediction_features(meal_data, user_profile, required_features=None):
    """
    Prepare features for prediction based on meal data and user profile.
    """
    # Extract nutrients from meal data
    nutrients = meal_data.get("total_nutrients", {})
    carbs = nutrients.get("carbs", 0)
    protein = nutrients.get("protein", 0)
    fat = nutrients.get("fat", 0)
    fiber = nutrients.get("fiber", 0)
    sugar = nutrients.get("sugar", 0)
    
    # Calculate GI and GL
    gi = meal_data.get("GI", None)
    if gi is None:
        # Calculate from food items
        food_items = meal_data.get("food_items", [])
        gi_values = []
        for food in food_items:
            gi_values.append(food.get("base_gi", 55))
        gi = sum(gi_values) / len(gi_values) if gi_values else 55
    
    gl = meal_data.get("Glycemic_Load", None)
    if gl is None:
        gl = (gi * carbs) / 100
    
    # Time of day
    try:
        timestamp = meal_data.get("timestamp", datetime.now().isoformat())
        hour = int(timestamp.split("T")[1].split(":")[0])
    except:
        hour = 12  # Default to noon
    
    is_breakfast = 1 if 6 <= hour < 10 else 0
    is_lunch = 1 if 11 <= hour < 14 else 0
    is_dinner = 1 if 17 <= hour < 21 else 0
    
    # User profile features
    if user_profile:
        age = user_profile.get("age", 40)
        is_male = 1 if user_profile.get("gender", "male") == "male" else 0
        bmi = user_profile.get("bmi", 25)
        
        # Diabetes status
        diabetes_status = user_profile.get("diabetes_status", "none")
        is_diabetic = 1 if diabetes_status in ["type1_diabetes", "type2_diabetes"] else 0
        is_prediabetic = 1 if diabetes_status == "pre_diabetic" else 0
        
        # Activity level
        activity_level = map_activity_level(user_profile.get("activity_level", "moderate"))
    else:
        # Default values if profile not provided
        age = 40
        is_male = 1
        bmi = 25
        is_diabetic = 0
        is_prediabetic = 0
        activity_level = 3
    
    # Other variables from meal data
    meal_features = meal_data.get("meal_features", {})
    minutes_after_meal = meal_features.get("Minutes_After_Meal", 0)
    stress_level = meal_features.get("Stress_Level", 5.0)
    sleep_quality = meal_features.get("Sleep_Quality", 7.0)
    activity_timing = meal_features.get("Activity_Timing", 2)  # Default to 'none'
    time_since_last_meal = meal_features.get("Time_Since_Last_Meal", 4.0)
    
    # Combine all features
    all_features = {
        "Age": age,
        "Sex": is_male,
        "BMI": bmi,
        "Category": get_diabetes_category(diabetes_status) if 'diabetes_status' in locals() else 0,
        "Carbs": carbs,
        "Fat": fat,
        "Fiber": fiber,
        "Protein": protein,
        "GI": gi,
        "Glycemic_Load": gl,
        "Sugar": sugar,
        "Carb_To_Fiber_Ratio": carbs / fiber if fiber > 0 else carbs,
        "Hour_Of_Day": hour,
        "Is_Breakfast": is_breakfast,
        "Is_Lunch": is_lunch,
        "Is_Dinner": is_dinner,
        "Activity_Level": activity_level,
        "Activity_Timing": activity_timing,
        "Stress_Level": stress_level,
        "Sleep_Quality": sleep_quality,
        "Time_Since_Last_Meal": time_since_last_meal,
        "Minutes_After_Meal": minutes_after_meal
    }
    
    # Create feature array based on required features
    if required_features:
        # Use only the features required by the model
        feature_array = []
        for feature in required_features:
            if isinstance(feature, int) and 0 <= feature < len(all_features):
                # If feature is specified as an index
                feature_array.append(list(all_features.values())[feature])
            elif feature in all_features:
                # If feature is specified by name
                feature_array.append(all_features[feature])
            else:
                # Default value if feature not found
                feature_array.append(0)
    else:
        # Use all features if no specific ones are required
        feature_array = list(all_features.values())
    
    return np.array([feature_array]), all_features

def predict_with_model(model_results, X):
    """Make prediction using model"""
    if not model_results or 'model' not in model_results:
        return None
    
    # Extract model
    model = model_results['model']
    
    # Make prediction
    try:
        prediction = model.predict(X)[0]
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def generate_time_series(model_results, meal_data, user_profile=None):
    """Generate time series predictions if model supports it"""
    time_series = []
    
    # Check if model has feature list
    if 'features' not in model_results:
        return time_series
    
    # Generate predictions at different time points
    for minute in range(0, 121, 5):  # 0 to 120 minutes, 5-minute intervals
        # Update time feature
        meal_copy = meal_data.copy()
        if 'meal_features' in meal_copy:
            meal_copy['meal_features']['Minutes_After_Meal'] = minute
        
        # Prepare features
        X, _ = prepare_prediction_features(meal_copy, user_profile, model_results['features'])
        
        # Make prediction
        glucose = predict_with_model(model_results, X)
        
        if glucose is not None:
            time_series.append({
                "minute": minute,
                "glucose": glucose
            })
    
    return time_series

def test_model(model_path, meals, user_profile=None):
    """Test model with provided meals"""
    results = []
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model_results = joblib.load(model_path)
        print("Model loaded successfully")
        
        # Print model info
        if 'training_date' in model_results:
            print(f"Model training date: {model_results['training_date']}")
        if 'num_samples' in model_results:
            print(f"Model trained on {model_results['num_samples']} samples")
        if 'train_score' in model_results:
            print(f"Model training score: {model_results['train_score']:.4f}")
        if 'val_score' in model_results:
            print(f"Model validation score: {model_results['val_score']:.4f}")
        
        # Process each meal
        for i, meal in enumerate(meals):
            print(f"\nTesting meal {i+1} of {len(meals)}...")
            print(f"Meal: {', '.join(food['name'] for food in meal['food_items'])}")
            
            # Prepare features
            X, feature_dict = prepare_prediction_features(
                meal, user_profile, model_results.get('features')
            )
            
            # Make prediction
            glucose = predict_with_model(model_results, X)
            
            if glucose is not None:
                # Determine glucose category
                if glucose < 70:
                    category = "Low"
                elif glucose < 140:
                    category = "Normal"
                elif glucose < 180:
                    category = "Elevated"
                else:
                    category = "High"
                
                # Generate time series if possible
                time_series = generate_time_series(model_results, meal, user_profile)
                
                # Find peak value if time series is available
                max_glucose = glucose
                max_time = 0
                
                if time_series:
                    glucose_values = [point['glucose'] for point in time_series]
                    if glucose_values:
                        max_index = glucose_values.index(max(glucose_values))
                        max_glucose = glucose_values[max_index]
                        max_time = time_series[max_index]['minute']
                
                # Add result
                result = {
                    "meal_id": meal.get("meal_id", f"meal_{i+1}"),
                    "foods": [food['name'] for food in meal['food_items']],
                    "nutrients": meal['total_nutrients'],
                    "features_used": feature_dict,
                    "prediction": {
                        "glucose": float(glucose),
                        "max_glucose": float(max_glucose),
                        "max_glucose_time": max_time,
                        "category": category,
                        "time_series": time_series
                    }
                }
                
                results.append(result)
                
                # Print summary
                print(f"Predicted glucose: {glucose:.1f} mg/dL ({category})")
                if time_series:
                    print(f"Peak glucose: {max_glucose:.1f} mg/dL at {max_time} minutes")
            else:
                print("Failed to make prediction for this meal")
                results.append({
                    "meal_id": meal.get("meal_id", f"meal_{i+1}"),
                    "foods": [food['name'] for food in meal['food_items']],
                    "error": "Failed to make prediction"
                })
        
        return results
    
    except Exception as e:
        print(f"Error testing model: {e}")
        return []

def main():
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.user_id:
        parser.error("Either --model-path or --user-id must be provided")
    
    # Determine model path
    model_path = args.model_path
    if not model_path and args.user_id:
        # Try to find user-specific model
        user_model_path = os.path.join('user_data', 'models', f"{args.user_id}_glucose_model.pkl")
        if os.path.exists(user_model_path):
            model_path = user_model_path
        else:
            # Fall back to standard model
            model_path = os.path.join('models', 'standard_glucose_model.pkl')
            print(f"User-specific model not found for {args.user_id}, using standard model")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return 1
    
    # Load or generate meals
    meals = []
    if args.meal_json:
        try:
            with open(args.meal_json, 'r') as f:
                meals = json.load(f)
            print(f"Loaded {len(meals)} meals from {args.meal_json}")
        except Exception as e:
            print(f"Error loading meal data: {e}")
            return 1
    elif args.generate_samples:
        print("Generating sample meals...")
        meals = generate_sample_meals(5)
        
        # Save sample meals
        with open('sample_meals.json', 'w') as f:
            json.dump(meals, f, indent=2)
        print(f"Saved sample meals to sample_meals.json")
    else:
        print("No meal data provided. Use --meal-json or --generate-samples")
        return 1
    
    # Load or create user profile
    user_profile = None
    if args.user_profile_json:
        try:
            with open(args.user_profile_json, 'r') as f:
                user_profile = json.load(f)
            print(f"Loaded user profile from {args.user_profile_json}")
        except Exception as e:
            print(f"Error loading user profile: {e}")
            return 1
    else:
        print("Using default user profile")
        user_profile = default_user_profile()
        
        # Save default profile
        with open('default_user_profile.json', 'w') as f:
            json.dump(user_profile, f, indent=2)
        print("Saved default user profile to default_user_profile.json")
    
    # Test model
    results = test_model(model_path, meals, user_profile)
    
    # Save results
    if results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved prediction results to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
