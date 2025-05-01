"""
Core machine learning models and personalization algorithms for GI Personalize app.
"""
import os
import json
import pickle
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'models.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load GI database
gi_database = None
try:
    gi_database = pd.read_csv(os.path.join('data', 'gi_database.csv'))
    logger.info("GI database loaded successfully")
except Exception as e:
    logger.error(f"Error loading GI database: {str(e)}")
    # Initialize with empty DataFrame as fallback
    gi_database = pd.DataFrame(columns=['food_name', 'glycemic_index', 'carbs_per_serving', 
                                        'protein_per_serving', 'fat_per_serving', 'fiber_per_serving'])

# Load reference values
reference_values = {}
try:
    with open(os.path.join('data', 'reference_values.json'), 'r') as f:
        reference_values = json.load(f)
    logger.info("Reference values loaded successfully")
except Exception as e:
    logger.error(f"Error loading reference values: {str(e)}")
    # Initialize with default values
    reference_values = {
        "standard_glucose_response": [100, 140, 130, 115, 105]
    }

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
        if gi_database is None or len(gi_database) == 0:
            return 50.0  # Return default if database not available
            
        result = gi_database[gi_database['food_name'].str.contains(food_name, case=False)]
        if len(result) > 0:
            return float(result.iloc[0]['glycemic_index'])
        else:
            # Fallback to similar food names
            words = food_name.split()
            if len(words) > 0:
                result = gi_database[gi_database['food_name'].str.contains(words[0], case=False)]
                if len(result) > 0:
                    return float(result.iloc[0]['glycemic_index'])
            # Return an estimated value if not found
            return 50.0  # Medium GI as default
    except Exception as e:
        logger.error(f"Error looking up GI for {food_name}: {str(e)}")
        return 50.0

def get_nutrients(food_name, portion_size="1 serving"):
    """
    Get nutritional information for a food.
    
    Args:
        food_name (str): Name of the food
        portion_size (str): Portion size description
        
    Returns:
        dict: Nutritional information
    """
    try:
        if gi_database is None or len(gi_database) == 0:
            # Return default values if database not available
            return {
                "calories": 200,
                "carbs": 15,
                "protein": 5,
                "fat": 3,
                "fiber": 2,
                "sugar": 3
            }
            
        # Look up in database
        result = gi_database[gi_database['food_name'].str.contains(food_name, case=False)]
        
        if len(result) > 0:
            food_data = result.iloc[0]
            
            # Extract portion multiplier if available (e.g., "2 servings")
            multiplier = 1.0
            try:
                parts = portion_size.split()
                if len(parts) >= 2 and parts[0].replace('.', '', 1).isdigit():
                    multiplier = float(parts[0])
            except:
                multiplier = 1.0
            
            return {
                "calories": food_data.get('calories_per_serving', 200) * multiplier,
                "carbs": food_data.get('carbs_per_serving', 15) * multiplier,
                "protein": food_data.get('protein_per_serving', 5) * multiplier,
                "fat": food_data.get('fat_per_serving', 3) * multiplier,
                "fiber": food_data.get('fiber_per_serving', 2) * multiplier,
                "sugar": food_data.get('sugar_per_serving', 3) * multiplier
            }
        else:
            # Fallback to similar food names
            words = food_name.split()
            if len(words) > 0:
                result = gi_database[gi_database['food_name'].str.contains(words[0], case=False)]
                if len(result) > 0:
                    food_data = result.iloc[0]
                    
                    # Extract portion multiplier
                    multiplier = 1.0
                    try:
                        parts = portion_size.split()
                        if len(parts) >= 2 and parts[0].replace('.', '', 1).isdigit():
                            multiplier = float(parts[0])
                    except:
                        multiplier = 1.0
                    
                    return {
                        "calories": food_data.get('calories_per_serving', 200) * multiplier,
                        "carbs": food_data.get('carbs_per_serving', 15) * multiplier,
                        "protein": food_data.get('protein_per_serving', 5) * multiplier,
                        "fat": food_data.get('fat_per_serving', 3) * multiplier,
                        "fiber": food_data.get('fiber_per_serving', 2) * multiplier,
                        "sugar": food_data.get('sugar_per_serving', 3) * multiplier
                    }
            
            # Default values if not found
            return {
                "calories": 200,
                "carbs": 15,
                "protein": 5,
                "fat": 3,
                "fiber": 2,
                "sugar": 3
            }
    except Exception as e:
        logger.error(f"Error getting nutrients for {food_name}: {str(e)}")
        # Return default values in case of error
        return {
            "calories": 200,
            "carbs": 15,
            "protein": 5,
            "fat": 3,
            "fiber": 2,
            "sugar": 3
        }

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
    reference_response = reference_values.get("standard_glucose_response", [100, 140, 130, 115, 105])
    
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

def prepare_training_data(meal_history, user_profile):
    """
    Prepare training data from user's meal history for personalized model.
    
    Args:
        meal_history (list): List of meal data with user responses
        user_profile (dict): User profile data
        
    Returns:
        tuple: (X, y) features and target values for training
    """
    X = []
    y = []
    
    for meal in meal_history:
        if 'glucose_readings' not in meal and 'user_response' not in meal:
            continue
        
        # Extract features from meal
        try:
            # Basic meal features
            features = []
            
            # Food composition features
            carbs = meal.get('total_nutrients', {}).get('carbs', 0)
            protein = meal.get('total_nutrients', {}).get('protein', 0)
            fat = meal.get('total_nutrients', {}).get('fat', 0)
            fiber = meal.get('total_nutrients', {}).get('fiber', 0)
            sugar = meal.get('total_nutrients', {}).get('sugar', 0)
            
            # Get base GI values
            gi_values = []
            for food in meal.get('food_items', []):
                gi_values.append(food.get('base_gi', 55))
            
            # Calculate meal averages
            avg_gi = sum(gi_values) / len(gi_values) if gi_values else 55
            gl = (avg_gi * carbs) / 100  # Glycemic load
            carb_to_fiber_ratio = carbs / fiber if fiber > 0 else carbs
            
            # Extract meal time (hour of day)
            try:
                timestamp = meal.get('timestamp', datetime.now().isoformat())
                hour = int(timestamp.split('T')[1].split(':')[0])
                # Encode time of day (0-23 hours)
                meal_time_hour = hour
                # Is it a main meal time?
                is_breakfast = 1 if 6 <= hour < 10 else 0
                is_lunch = 1 if 11 <= hour < 14 else 0
                is_dinner = 1 if 17 <= hour < 21 else 0
            except:
                meal_time_hour = 12  # Default to noon
                is_breakfast = 0
                is_lunch = 1
                is_dinner = 0
            
            # User profile features
            age = user_profile.get('age', 40)
            is_male = 1 if user_profile.get('gender', 'male') == 'male' else 0
            bmi = user_profile.get('bmi', 25)
            
            # Diabetes status encoding
            diabetes_status = user_profile.get('diabetes_status', 'none')
            is_diabetic = 1 if diabetes_status in ['type1_diabetes', 'type2_diabetes'] else 0
            is_prediabetic = 1 if diabetes_status == 'pre_diabetic' else 0
            
            # Activity level encoding
            activity_level = map_activity_level(user_profile.get('activity_level', 'moderate'))
            
            # Create feature vector
            features = [
                avg_gi,
                gl,
                carbs,
                protein,
                fat,
                fiber,
                sugar,
                carb_to_fiber_ratio,
                meal_time_hour,
                is_breakfast,
                is_lunch,
                is_dinner,
                age,
                is_male,
                bmi,
                is_diabetic,
                is_prediabetic,
                activity_level
            ]
            
            # Get target value - glucose reading or user response
            if 'glucose_readings' in meal and meal['glucose_readings']:
                # Use maximum glucose reading as target
                readings = meal['glucose_readings']
                target = max(readings) if readings else 0
            elif 'user_response' in meal:
                # Map user response to numerical target
                response_value = meal['user_response'].get('response', 'as_expected')
                if response_value == 'less_than_expected':
                    target = 0
                elif response_value == 'as_expected':
                    target = 1
                else:  # 'more_than_expected'
                    target = 2
            else:
                # Skip if no target value available
                continue
            
            X.append(features)
            y.append(target)
            
        except Exception as e:
            logger.error(f"Error processing meal for training: {str(e)}")
            continue
    
    return np.array(X), np.array(y)

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

def load_user_model(user_id):
    """
    Load a user's personalized glucose prediction model.
    
    Args:
        user_id (str): User ID
        
    Returns:
        dict: Model results containing the model and feature list, or None if not found
    """
    try:
        # Check for user-specific model
        user_model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_glucose_model.pkl")
        if os.path.exists(user_model_path):
            model_results = joblib.load(user_model_path)
            logger.info(f"Loaded personalized glucose model for user {user_id}")
            return model_results
        else:
            # Fall back to standard model if user model doesn't exist
            standard_model_path = os.path.join(Config.MODEL_FOLDER, 'standard_glucose_model.pkl')
            if os.path.exists(standard_model_path):
                model_results = joblib.load(standard_model_path)
                logger.info(f"Using standard glucose model for user {user_id} (no personalized model found)")
                return model_results
        
        logger.warning(f"No glucose prediction model found for user {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error loading glucose model for user {user_id}: {str(e)}")
        return None

def update_user_model(user_id, feature_data, response_data, user_profile=None):
    """
    Update user's personalized model with new training data.
    
    Args:
        user_id (str): User ID
        feature_data (list or array): Feature data for training
        response_data (list or array): Response data for training
        user_profile (dict, optional): User profile data
        
    Returns:
        bool: True if model was updated successfully, False otherwise
    """
    try:
        # Convert to numpy arrays if not already
        X = np.array(feature_data)
        y = np.array(response_data)
        
        # Check if we have enough data
        if len(X) < 5:
            logger.warning(f"Not enough training data for user {user_id} (only {len(X)} samples)")
            return False
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a gradient boosting model (better for small datasets than Random Forest)
        model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        logger.info(f"User {user_id} model - Training R²: {train_score:.4f}, Validation R²: {val_score:.4f}")
        
        # Extract feature importance
        feature_importance = model.feature_importances_
        
        # Prepare data to save
        model_results = {
            'model': model,
            'features': list(range(X.shape[1])),  # List of feature indices
            'feature_importance': feature_importance.tolist(),
            'train_score': train_score,
            'val_score': val_score,
            'training_date': datetime.now().isoformat(),
            'num_samples': len(X)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(Config.USER_DATA_FOLDER, 'models'), exist_ok=True)
        
        # Save the model
        model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_glucose_model.pkl")
        joblib.dump(model_results, model_path)
        
        logger.info(f"Personalized glucose model updated for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating model for user {user_id}: {str(e)}")
        return False

def predict_glucose_response(user_id, meal_data, user_profile=None):
    """
    Predict glucose response to a meal using the user's personalized model.
    
    Args:
        user_id (str): User ID
        meal_data (dict): Meal data including nutrients and GI values
        user_profile (dict, optional): User profile data
        
    Returns:
        dict: Predicted glucose response
    """
    try:
        # Load the user's model
        model_results = load_user_model(user_id)
        if not model_results or 'model' not in model_results:
            logger.warning(f"No model available for user {user_id}, using baseline prediction")
            return baseline_glucose_prediction(meal_data, user_profile)
        
        # Extract the model and feature list
        model = model_results['model']
        features = model_results.get('features', [])
        
        # Prepare input features
        X = prepare_prediction_features(meal_data, user_profile, features)
        
        # Make prediction
        try:
            predicted_glucose = model.predict(X)[0]
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return baseline_glucose_prediction(meal_data, user_profile)
        
        # Calculate glucose category based on predicted value
        if predicted_glucose < 70:
            category = "Low"
        elif predicted_glucose < 140:
            category = "Normal"
        elif predicted_glucose < 180:
            category = "Elevated"
        else:
            category = "High"
        
        # Generate time series if possible
        time_series = []
        max_glucose_time = 0
        
        if model_results.get('time_series_capable', False):
            # Generate time series predictions
            time_points = range(0, 120, 5)  # Every 5 minutes for 2 hours
            glucose_values = []
            
            for minute in time_points:
                time_data = meal_data.copy()
                time_data['Minutes_After_Meal'] = minute
                
                # Prepare features for this time point
                X_time = prepare_prediction_features(time_data, user_profile, features)
                
                # Predict glucose at this time point
                try:
                    glucose_at_time = model.predict(X_time)[0]
                    time_series.append({
                        'minute': minute,
                        'glucose': glucose_at_time
                    })
                    glucose_values.append(glucose_at_time)
                except Exception as e:
                    logger.error(f"Error predicting time series at minute {minute}: {str(e)}")
            
            # Find peak glucose value and time
            if glucose_values:
                max_glucose = max(glucose_values)
                max_index = glucose_values.index(max_glucose)
                max_glucose_time = time_points[max_index]
                
                # Use peak glucose as the prediction if higher
                if max_glucose > predicted_glucose:
                    predicted_glucose = max_glucose
                    
                    # Recalculate category
                    if predicted_glucose < 70:
                        category = "Low"
                    elif predicted_glucose < 140:
                        category = "Normal"
                    elif predicted_glucose < 180:
                        category = "Elevated"
                    else:
                        category = "High"
        
        return {
            'predicted_glucose': predicted_glucose,
            'glucose_category': category,
            'time_series': time_series,
            'max_glucose_time': max_glucose_time,
            'model_type': 'personalized' if user_id in model_results.get('model_path', '') else 'standard'
        }
    except Exception as e:
        logger.error(f"Error in glucose prediction: {str(e)}")
        return baseline_glucose_prediction(meal_data, user_profile)

def prepare_prediction_features(meal_data, user_profile, required_features=None):
    """
    Prepare features for prediction based on meal data and user profile.
    
    Args:
        meal_data (dict): Meal data
        user_profile (dict): User profile data
        required_features (list): List of required feature names
        
    Returns:
        numpy.ndarray: Feature array for prediction
    """
    # Extract nutrients from meal data
    nutrients = meal_data.get('total_nutrients', {})
    carbs = nutrients.get('carbs', 0)
    protein = nutrients.get('protein', 0)
    fat = nutrients.get('fat', 0)
    fiber = nutrients.get('fiber', 0)
    sugar = nutrients.get('sugar', 0)
    
    # Calculate GI and GL
    gi = meal_data.get('GI', None)
    if gi is None:
        # Calculate from food items
        food_items = meal_data.get('food_items', [])
        gi_values = []
        for food in food_items:
            gi_values.append(food.get('base_gi', 55))
        gi = sum(gi_values) / len(gi_values) if gi_values else 55
    
    gl = meal_data.get('Glycemic_Load', None)
    if gl is None:
        gl = (gi * carbs) / 100
    
    # Time of day
    try:
        timestamp = meal_data.get('timestamp', datetime.now().isoformat())
        hour = int(timestamp.split('T')[1].split(':')[0])
    except:
        hour = 12  # Default to noon
    
    is_breakfast = 1 if 6 <= hour < 10 else 0
    is_lunch = 1 if 11 <= hour < 14 else 0
    is_dinner = 1 if 17 <= hour < 21 else 0
    
    # User profile features
    if user_profile:
        age = user_profile.get('age', 40)
        is_male = 1 if user_profile.get('gender', 'male') == 'male' else 0
        bmi = user_profile.get('bmi', 25)
        
        # Diabetes status
        diabetes_status = user_profile.get('diabetes_status', 'none')
        is_diabetic = 1 if diabetes_status in ['type1_diabetes', 'type2_diabetes'] else 0
        is_prediabetic = 1 if diabetes_status == 'pre_diabetic' else 0
        
        # Activity level
        activity_level = map_activity_level(user_profile.get('activity_level', 'moderate'))
    else:
        # Default values if profile not provided
        age = 40
        is_male = 1
        bmi = 25
        is_diabetic = 0
        is_prediabetic = 0
        activity_level = 3
    
    # Other variables from meal_data
    minutes_after_meal = meal_data.get('Minutes_After_Meal', 0)
    stress_level = meal_data.get('Stress_Level', 5.0)
    sleep_quality = meal_data.get('Sleep_Quality', 7.0)
    activity_timing = meal_data.get('Activity_Timing', 2)  # Default to 'none'
    time_since_last_meal = meal_data.get('Time_Since_Last_Meal', 4.0)
    
    # Combine all features
    all_features = {
        'Age': age,
        'Sex': is_male,
        'BMI': bmi,
        'Category': 2 if is_diabetic else (1 if is_prediabetic else 0),
        'Carbs': carbs,
        'Fat': fat,
        'Fiber': fiber,
        'Protein': protein,
        'GI': gi,
        'Glycemic_Load': gl,
        'Sugar': sugar,
        'Carb_To_Fiber_Ratio': carbs / fiber if fiber > 0 else carbs,
        'Hour_Of_Day': hour,
        'Is_Breakfast': is_breakfast,
        'Is_Lunch': is_lunch,
        'Is_Dinner': is_dinner,
        'Activity_Level': activity_level,
        'Activity_Timing': activity_timing,
        'Stress_Level': stress_level,
        'Sleep_Quality': sleep_quality,
        'Time_Since_Last_Meal': time_since_last_meal,
        'Minutes_After_Meal': minutes_after_meal
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
    
    return np.array([feature_array])

def baseline_glucose_prediction(meal_data, user_profile=None):
    """
    Make a baseline glucose prediction without using ML model.
    This serves as a fallback when no model is available.
    
    Args:
        meal_data (dict): Meal data including nutrients and GI
        user_profile (dict, optional): User profile data
        
    Returns:
        dict: Baseline prediction
    """
    # Extract key features
    nutrients = meal_data.get('total_nutrients', {})
    carbs = nutrients.get('carbs', 0)
    fat = nutrients.get('fat', 0)
    protein = nutrients.get('protein', 0)
    
    # Calculate GI
    gi = meal_data.get('GI', None)
    if gi is None:
        # Calculate from food items
        food_items = meal_data.get('food_items', [])
        gi_values = []
        for food in food_items:
            gi_values.append(food.get('base_gi', 55))
        gi = sum(gi_values) / len(gi_values) if gi_values else 55
    
    # Baseline formula based on carbs and GI
    # Starting with fasting glucose (baseline)
    fasting_glucose = 85  # Average fasting glucose for healthy individual
    
    # Adjust based on user profile if available
    if user_profile:
        diabetes_status = user_profile.get('diabetes_status', 'none')
        if diabetes_status == 'pre_diabetic':
            fasting_glucose = 100
        elif diabetes_status in ['type1_diabetes', 'type2_diabetes']:
            fasting_glucose = 120
    
    # Simple formula: higher carbs and higher GI lead to higher glucose response
    # GL = GI * carbs / 100
    gl = (gi * carbs) / 100
    
    # Estimated glucose rise based on GL
    glucose_rise = gl * 4  # Approximate factor
    
    # Fat and protein can moderate glucose rise
    glucose_modifier = 1.0
    if fat + protein > 0:
        # Higher fat and protein can lower glucose rise by up to 30%
        ratio = min(1.0, (fat + protein) / (carbs if carbs > 0 else 1))
        glucose_modifier = max(0.7, 1.0 - ratio * 0.3)
    
    # Calculate predicted peak glucose
    predicted_glucose = fasting_glucose + (glucose_rise * glucose_modifier)
    
    # Determine category
    if predicted_glucose < 70:
        category = "Low"
    elif predicted_glucose < 140:
        category = "Normal"
    elif predicted_glucose < 180:
        category = "Elevated"
    else:
        category = "High"
    
    # Create time series - simple bell curve
    time_series = []
    glucose_values = []
    for minute in range(0, 120, 5):
        # Simple model: rise until 30-45 mins then fall
        if minute <= 45:
            percentage = min(1.0, minute / 45.0)
            glucose = fasting_glucose + percentage * (predicted_glucose - fasting_glucose)
        else:
            percentage = min(1.0, (minute - 45) / 75.0)
            glucose = predicted_glucose - percentage * (predicted_glucose - fasting_glucose)
        
        time_series.append({
            'minute': minute,
            'glucose': glucose
        })
        glucose_values.append(glucose)
    
    # Max glucose time - default to 45 minutes
    max_glucose_time = 45
    
    return {
        'predicted_glucose': predicted_glucose,
        'glucose_category': category,
        'time_series': time_series,
        'max_glucose_time': max_glucose_time,
        'model_type': 'baseline'
    }

def train_initial_model(user_id, user_profile, meal_count=10):
    """
    Train an initial personalized model for a new user based on their profile.
    
    This function simulates training data based on typical responses for
    the user's demographic and health status.
    
    Args:
        user_id (str): User ID
        user_profile (dict): User profile data
        meal_count (int): Number of simulated meals to generate
        
    Returns:
        bool: True if model was created successfully, False otherwise
    """
    try:
        # Extract key profile information
        age = user_profile.get('age', 40)
        gender = user_profile.get('gender', 'male')
        is_male = 1 if gender == 'male' else 0
        bmi = user_profile.get('bmi', 25)
        diabetes_status = user_profile.get('diabetes_status', 'none')
        is_diabetic = 1 if diabetes_status in ['type1_diabetes', 'type2_diabetes'] else 0
        is_prediabetic = 1 if diabetes_status == 'pre_diabetic' else 0
        activity_level = map_activity_level(user_profile.get('activity_level', 'moderate'))
        
        # Generate simulated meal data
        X = []
        y = []
        
        # Base glucose levels by diabetes status
        if diabetes_status in ['type1_diabetes', 'type2_diabetes']:
            base_glucose = 130
            glucose_variance = 50
        elif diabetes_status == 'pre_diabetic':
            base_glucose = 110
            glucose_variance = 35
        else:
            base_glucose = 90
            glucose_variance = 25
        
        # Age adjustment for base glucose
        age_factor = min(1.5, max(1.0, age / 50))
        base_glucose *= age_factor
        
        # BMI adjustment
        bmi_factor = min(1.3, max(0.9, bmi / 25))
        base_glucose *= bmi_factor
        
        # Activity level adjustment
        activity_factor = max(0.7, min(1.2, 1.1 - (activity_level - 3) * 0.1))
        base_glucose *= activity_factor
        
        # Generate random meal scenarios
        for _ in range(meal_count):
            # Random meal compositions
            avg_gi = np.random.uniform(30, 80)
            carbs = np.random.uniform(10, 100)
            protein = np.random.uniform(5, 30)
            fat = np.random.uniform(2, 25)
            fiber = np.random.uniform(1, 15)
            sugar = np.random.uniform(0, carbs * 0.4)  # Sugar up to 40% of carbs
            gl = (avg_gi * carbs) / 100
            carb_to_fiber_ratio = carbs / fiber if fiber > 0 else carbs
            
            # Random meal time
            meal_time_hour = np.random.randint(6, 22)
            is_breakfast = 1 if 6 <= meal_time_hour < 10 else 0
            is_lunch = 1 if 11 <= meal_time_hour < 14 else 0
            is_dinner = 1 if 17 <= meal_time_hour < 21 else 0
            
            # Create feature vector
            features = [
                avg_gi,
                gl,
                carbs,
                protein,
                fat,
                fiber,
                sugar,
                carb_to_fiber_ratio,
                meal_time_hour,
                is_breakfast,
                is_lunch,
                is_dinner,
                age,
                is_male,
                bmi,
                is_diabetic,
                is_prediabetic,
                activity_level
            ]
            
            # Simulate glucose response based on profile and meal
            # Higher GI, carbs, and GL lead to higher glucose
            glucose_impact = gl * 4  # Base impact from glycemic load
            
            # Fat and protein moderating effect
            moderating_ratio = min(1.0, (fat + protein) / (carbs if carbs > 0 else 1))
            glucose_modifier = max(0.7, 1.0 - moderating_ratio * 0.3)
            
            # Generate target glucose level with some randomness
            target_glucose = base_glucose + (glucose_impact * glucose_modifier)
            
            # Add noise
            target_glucose += np.random.normal(0, glucose_variance * 0.2)
            
            # Ensure glucose is positive
            target_glucose = max(70, target_glucose)
            
            X.append(features)
            y.append(target_glucose)
        
        # Train and save the model
        success = update_user_model(user_id, X, y, user_profile)
        
        if success:
            logger.info(f"Initial model created for user {user_id} with {meal_count} simulated meals")
        else:
            logger.warning(f"Failed to create initial model for user {user_id}")
        
        return success
    except Exception as e:
        logger.error(f"Error creating initial model for user {user_id}: {str(e)}")
        return False

def update_model_with_feedback(user_id, meal_id, glucose_readings=None, user_feedback=None):
    """
    Update user's model with feedback on a specific meal.
    
    Args:
        user_id (str): User ID
        meal_id (str): Meal ID to update
        glucose_readings (list, optional): Actual glucose readings
        user_feedback (dict, optional): User feedback on meal
        
    Returns:
        bool: True if model was updated successfully, False otherwise
    """
    try:
        from utils.database import get_user_data, save_user_data
        
        # Get user data
        user_data = get_user_data(user_id)
        if not user_data:
            logger.error(f"User {user_id} not found")
            return False
        
        # Find the meal
        meal_found = False
        for i, meal in enumerate(user_data.get('meals', [])):
            if meal.get('meal_id') == meal_id:
                meal_found = True
                
                # Update meal with feedback
                if glucose_readings:
                    user_data['meals'][i]['glucose_readings'] = glucose_readings
                
                if user_feedback:
                    user_data['meals'][i]['user_feedback'] = user_feedback
                
                # Save user data
                save_user_data(user_id, user_data)
                break
        
        if not meal_found:
            logger.warning(f"Meal {meal_id} not found for user {user_id}")
            return False
        
        # Check if we have enough data to retrain the model
        if len(user_data.get('meals', [])) >= 5:
            # Prepare training data
            X, y = prepare_training_data(user_data.get('meals', []), user_data.get('profile', {}))
            
            if len(X) >= 5:
                # Update the model
                success = update_user_model(user_id, X, y, user_data.get('profile', {}))
                
                if success:
                    logger.info(f"Model updated for user {user_id} with new feedback")
                    return True
                else:
                    logger.warning(f"Failed to update model for user {user_id}")
                    return False
            else:
                logger.info(f"Not enough valid meals with feedback for user {user_id} to update model")
                return False
        else:
            logger.info(f"Not enough meals for user {user_id} to update model (minimum 5 required)")
            return False
    except Exception as e:
        logger.error(f"Error updating model with feedback for user {user_id}: {str(e)}")
        return False

def get_model_info(user_id):
    """
    Get information about the user's glucose prediction model.
    
    Args:
        user_id (str): User ID
        
    Returns:
        dict: Model information
    """
    try:
        # Check for user-specific model
        user_model_path = os.path.join(Config.USER_DATA_FOLDER, 'models', f"{user_id}_glucose_model.pkl")
        if os.path.exists(user_model_path):
            model_results = joblib.load(user_model_path)
            
            info = {
                'model_type': 'personalized',
                'training_date': model_results.get('training_date', 'unknown'),
                'num_samples': model_results.get('num_samples', 0),
                'accuracy': model_results.get('val_score', 0) * 100,
                'feature_importance': model_results.get('feature_importance', []),
                'available': True
            }
            
            return info
        else:
            # Check if standard model exists
            standard_model_path = os.path.join(Config.MODEL_FOLDER, 'standard_glucose_model.pkl')
            if os.path.exists(standard_model_path):
                return {
                    'model_type': 'standard',
                    'available': True,
                    'message': 'Using standard model. Personalized model will be created after collecting sufficient meal data.'
                }
            else:
                return {
                    'model_type': 'none',
                    'available': False,
                    'message': 'No glucose prediction model available.'
                }
    except Exception as e:
        logger.error(f"Error getting model info for user {user_id}: {str(e)}")
        return {
            'model_type': 'error',
            'available': False,
            'message': f"Error accessing model: {str(e)}"
        }
