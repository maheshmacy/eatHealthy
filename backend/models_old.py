"""
Core machine learning models and personalization algorithms for GI Personalize app.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from config import Config

# Load GI database
gi_database = pd.read_csv(os.path.join('data', 'gi_database.csv'))

# Load reference values
with open(os.path.join('data', 'reference_values.json'), 'r') as f:
    reference_values = json.load(f)

def calculate_baseline_risk_factor(user_profile):
    """
    Calculate a baseline risk factor for a user based on their profile.
    
    Args:
        user_profile (dict): User's health profile information
        
    Returns:
        float: Risk factor (higher indicates more sensitivity to high GI foods)
    """
    # Start with base factor of 1.0
    risk_factor = 1.0
    
    # Adjust based on diabetes status
    diabetes_status = user_profile.get('diabetes_status', 'none')
    if diabetes_status == 'pre_diabetic':
        risk_factor *= 1.5
    elif diabetes_status == 'type2_diabetes':
        risk_factor *= 2.0
    elif diabetes_status == 'type1_diabetes':
        risk_factor *= 2.5
    
    # Adjust based on BMI
    bmi = user_profile.get('bmi', 25)
    if bmi > 30:
        risk_factor *= 1.3
    elif bmi > 25:
        risk_factor *= 1.1
    
    # Adjust based on activity level (more active = lower risk)
    activity_level = user_profile.get('activity_level', 'moderate')
    if activity_level == 'very_active':
        risk_factor *= 0.7
    elif activity_level == 'moderately_active':
        risk_factor *= 0.8
    elif activity_level == 'lightly_active':
        risk_factor *= 0.9
    elif activity_level == 'extremely_active':
        risk_factor *= 0.6
    
    # Adjust based on age (risk increases with age)
    age = user_profile.get('age', 40)
    if age > 60:
        risk_factor *= 1.2
    elif age > 40:
        risk_factor *= 1.1
    
    # Adjust based on HbA1c if available
    hba1c = user_profile.get('hba1c', None)
    if hba1c is not None:
        if hba1c > 6.5:
            risk_factor *= 1.5
        elif hba1c > 5.7:
            risk_factor *= 1.2
    
    # Adjust based on fasting glucose if available
    fasting_glucose = user_profile.get('fasting_glucose', None)
    if fasting_glucose is not None:
        if fasting_glucose > 126:  # mg/dL
            risk_factor *= 1.5
        elif fasting_glucose > 100:
            risk_factor *= 1.2
    
    # Adjust based on weight goal
    weight_goal = user_profile.get('weight_goal', 'maintain')
    if weight_goal == 'lose':
        risk_factor *= 1.1  # Slightly higher concern for high GI foods when trying to lose weight
    
    # Adjust based on sex (women typically have higher insulin sensitivity)
    gender = user_profile.get('gender', None)
    if gender == 'female':
        risk_factor *= 0.95
    
    return risk_factor

def personalize_gi_impact(base_gi, user_profile):
    """
    Personalize the glycemic impact of a food based on user profile.
    
    Args:
        base_gi (float): Base glycemic index of the food
        user_profile (dict): User's health profile
        
    Returns:
        dict: Personalized GI impact information
    """
    # Calculate personalized GI impact
    risk_factor = calculate_baseline_risk_factor(user_profile)
    
    # Calculate personalized GI score
    personalized_gi = min(100, base_gi * risk_factor)
    
    # Determine impact level
    if personalized_gi >= 70:
        impact_level = "high"
        warning_level = "High Risk" if risk_factor > 1.5 else "Moderate Risk"
        recommended_portion = "Small" if risk_factor > 1.5 else "Medium"
    elif personalized_gi >= 55:
        impact_level = "medium"
        warning_level = "Moderate Risk" if risk_factor > 1.8 else "Low Risk"
        recommended_portion = "Medium" if risk_factor > 1.8 else "Standard"
    else:
        impact_level = "low"
        warning_level = "Low Risk"
        recommended_portion = "Standard"
    
    # Generate personalized recommendations
    recommendations = []
    
    if impact_level == "high":
        recommendations.append("Consider pairing with protein or healthy fat to reduce glycemic impact")
        recommendations.append("Consume earlier in the day when insulin sensitivity is typically higher")
    
    if user_profile.get('diabetes_status', 'none') != 'none':
        recommendations.append("Monitor your blood glucose response if you consume this food")
    
    if impact_level == "medium" and user_profile.get('weight_goal') == 'lose':
        recommendations.append("Moderate portion sizes to help with weight management goals")
    
    return {
        "personalized_gi_score": personalized_gi,
        "impact_level": impact_level,
        "warning_level": warning_level,
        "recommended_portion": recommended_portion,
        "risk_factor": risk_factor,
        "recommendations": recommendations
    }

def lookup_gi(food_name):
    """
    Look up the glycemic index of a food in the database.
    
    Args:
        food_name (str): Name of the food
        
    Returns:
        float: Glycemic index value
    """
    # Search in database
    try:
        result = gi_database[gi_database['food_name'].str.contains(food_name, case=False)]
        if len(result) > 0:
            return float(result.iloc[0]['glycemic_index'])
        else:
            # Fallback to similar food names
            result = gi_database[gi_database['food_name'].str.contains(food_name.split()[0], case=False)]
            if len(result) > 0:
                return float(result.iloc[0]['glycemic_index'])
            # Return an estimated value if not found
            return 50.0  # Medium GI as default
    except Exception as e:
        print(f"Error looking up GI: {e}")
        return 50.0

def calculate_incremental_auc(glucose_readings):
    """
    Calculate the incremental area under the curve for glucose readings.
    
    Args:
        glucose_readings (list): List of glucose readings at different time points
        
    Returns:
        float: Incremental area under the curve
    """
    if len(glucose_readings) < 2:
        return 0
    
    # Assuming readings are at regular intervals
    baseline = glucose_readings[0]
    
    # Calculate iAUC using trapezoidal method
    total_auc = 0
    for i in range(1, len(glucose_readings)):
        # Only count area above baseline
        prev_height = max(0, glucose_readings[i-1] - baseline)
        curr_height = max(0, glucose_readings[i] - baseline)
        
        # Trapezoid area = average height × width (assuming equal time intervals of 1 unit)
        area = 0.5 * (prev_height + curr_height) * 1
        total_auc += area
    
    return total_auc

def process_calibration_meal(glucose_readings):
    """
    Process calibration meal data to calculate user's personal response factor.
    
    Args:
        glucose_readings (list): List of glucose readings at different time points
        
    Returns:
        float: Personal response factor compared to population average
    """
    # Standard reference response (population average) - [0, 30, 60, 90, 120min]
    reference_response = reference_values["standard_glucose_response"]
    
    # Calculate area under curve for this user
    user_auc = calculate_incremental_auc(glucose_readings)
    reference_auc = calculate_incremental_auc(reference_response)
    
    # Calculate personal response factor
    if reference_auc == 0:
        return 1.0
    
    response_factor = user_auc / reference_auc
    
    # Normalize extreme values
    if response_factor < 0.5:
        response_factor = 0.5
    elif response_factor > 2.0:
        response_factor = 2.0
    
    return response_factor

def prepare_training_data(meal_history):
    """
    Prepare training data from user's meal history for personalized model.
    
    Args:
        meal_history (list): List of meal data with user responses
        
    Returns:
        tuple: (X, y) features and target values for training
    """
    X = []
    y = []
    
    for meal in meal_history:
        if 'user_response' not in meal:
            continue
        
        # Extract features from meal
        try:
            # Basic meal features
            features = []
            
            # Food composition features
            carbs = 0
            protein = 0
            fat = 0
            fiber = 0
            gi_sum = 0
            
            # Process each food item
            for food in meal['food_items']:
                gi_sum += food['base_gi']
                
                # Extract nutritional info from GI database
                food_data = gi_database[gi_database['food_name'].str.contains(food['food_name'], case=False)]
                if len(food_data) > 0:
                    carbs += food_data.iloc[0].get('carbs_per_serving', 15)
                    protein += food_data.iloc[0].get('protein_per_serving', 5)
                    fat += food_data.iloc[0].get('fat_per_serving', 3)
                    fiber += food_data.iloc[0].get('fiber_per_serving', 1)
                else:
                    # Default values if not found
                    carbs += 15
                    protein += 5
                    fat += 3
                    fiber += 1
            
            # Calculate meal averages
            avg_gi = gi_sum / len(meal['food_items']) if len(meal['food_items']) > 0 else 0
            total_carbs = carbs
            carb_to_fiber_ratio = carbs / fiber if fiber > 0 else carbs
            
            # Extract meal time (hour of day)
            try:
                timestamp = meal.get('timestamp', datetime.now().isoformat())
                hour = int(timestamp.split('T')[1].split(':')[0])
                meal_time_category = 0  # breakfast
                if 11 <= hour < 15:
                    meal_time_category = 1  # lunch
                elif 15 <= hour < 22:
                    meal_time_category = 2  # dinner
            except:
                meal_time_category = 0  # default to breakfast
            
            # Create feature vector
            features = [
                avg_gi,
                total_carbs,
                protein,
                fat,
                fiber,
                carb_to_fiber_ratio,
                meal_time_category,
            ]
            
            # Get target value (response level)
            response_value = meal['user_response'].get('response', 'as_expected')
            if response_value == 'less_than_expected':
                target = 0
            elif response_value == 'as_expected':
                target = 1
            else:  # 'more_than_expected'
                target = 2
            
            X.append(features)
            y.append(target)
            
        except Exception as e:
            print(f"Error processing meal for training: {str(e)}")
            continue
    
    return np.array(X), np.array(y)

def update_user_model(user_id, X, y):
    """
    Update user's personalized model with new training data.
    
    Args:
        user_id (str): User ID
        X (array): Feature matrix
        y (array): Target values
        
    Returns:
        bool: True if model was updated, False otherwise
    """
    try:
        # Train a Random Forest model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save the model
        os.makedirs(os.path.join(Config.USER_DATA_FOLDER, 'models'), exist_ok=True)
        model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return True
    except Exception as e:
        print(f"Error updating user model: {str(e)}")
        return False

def predict_response(user_id, features):
    """
    Use personalized model to predict response to a meal.
    
    Args:
        user_id (str): User ID
        features (array): Feature vector
        
    Returns:
        float: Predicted response (0=less than expected, 1=as expected, 2=more than expected)
    """
    try:
        model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_model.pkl")
        
        if not os.path.exists(model_path):
            return 1  # Default to "as expected"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        prediction = model.predict([features])[0]
        return prediction
    except Exception as e:
        print(f"Error predicting response: {str(e)}")
        return 1  # Default to "as expected"
