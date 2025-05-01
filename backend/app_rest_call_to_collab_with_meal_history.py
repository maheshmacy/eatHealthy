"""
Optimized Flask application for EATSMART-AI with focus on core functionality.
Includes enhanced food analysis with ML-based glucose spike prediction.
"""
from flask import Flask, request, jsonify
import os
import uuid
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage

# Import modules

from utils.database import get_user_data, save_user_data, initialize_database
from utils.food_recognition import identify_food_in_image
from utils.validators import validate_user_data, validate_glucose_readings
import models
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask-RESTX API with Swagger documentation
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}
api = Api(app, 
          version='1.0', 
          title='EATSMART-AI',
          description='API for Personalized Glycemic Index Tracking',
          doc='/swagger',
          authorizations=authorizations,
          security='apikey')

# Define API namespaces
health_ns = api.namespace('health', description='Health Check Endpoints')
users_ns = api.namespace('users', description='User Profile Management')
food_ns = api.namespace('food', description='Food Analysis Endpoints')
training_ns = api.namespace('training', description='Model Training Endpoints')
meals_ns = api.namespace('meals', description='Meal History Management')

# Define API models
user_profile_model = api.model('UserProfile', {
    'name': fields.String(required=True, description='User\'s full name'),
    'age': fields.Integer(required=True, description='User\'s age'),
    'gender': fields.String(required=True, description='User\'s gender'),
    'height': fields.Float(required=True, description='User height in cm'),
    'weight': fields.Float(required=True, description='User weight in kg'),
    'activity_level': fields.String(description='Activity level'),
    'diabetes_status': fields.String(description='Diabetes status'),
    'weight_goal': fields.String(description='Weight management goal'),
    'hba1c': fields.Float(description='HbA1c percentage'),
    'fasting_glucose': fields.Float(description='Fasting glucose level')
})

food_analysis_model = api.model('FoodAnalysis', {
    'food_name': fields.String(required=True, description='Identified food name'),
    'confidence': fields.Float(description='Confidence of food identification'),
    'base_gi': fields.Float(description='Base Glycemic Index'),
    'personalized_gi': fields.Raw(description='Personalized GI impact'),
    'nutrients': fields.Raw(description='Nutritional content')
})

glucose_prediction_model = api.model('GlucosePrediction', {
    'predicted_glucose': fields.Float(description='Predicted glucose level (mg/dL)'),
    'glucose_category': fields.String(description='Risk level (Low, Normal, Elevated, High)'),
    'time_series': fields.List(fields.Float, description='Predicted glucose values over time'),
    'max_glucose_time': fields.Integer(description='Time to peak glucose (minutes)'),
    'guidelines': fields.List(fields.String, description='Dietary guidelines'),
    'recommendations': fields.List(fields.String, description='Specific recommendations')
})

meal_filter_model = api.model('MealFilter', {
    'start_date': fields.String(description='Start date (ISO format)'),
    'end_date': fields.String(description='End date (ISO format)'),
    'food_type': fields.String(description='Filter by food type'),
    'sort_by': fields.String(description='Sort field', enum=['date', 'glucose_impact']),
    'sort_order': fields.String(description='Sort order', enum=['asc', 'desc']),
    'limit': fields.Integer(description='Maximum number of meals to return')
})

training_model = api.model('ModelTraining', {
    'feature_data': fields.List(fields.Raw(), description='Feature data for training'),
    'response_data': fields.List(fields.Raw(), description='Response data for training')
})

# Create parser for food analysis endpoint
food_image_parser = api.parser()
food_image_parser.add_argument('food_image', location='files', type=FileStorage, required=True, help='Food image file')
food_image_parser.add_argument('user_id', location='form', type=str, required=True, help='User ID')
food_image_parser.add_argument('person_info', location='form', type=str, required=False, help='JSON string with person info')



# Ensure required directories exist
for directory in [app.config['UPLOAD_FOLDER'], 
                 app.config['USER_DATA_FOLDER'],
                 app.config.get('MODEL_FOLDER', 'models'),
                 'logs']:
    os.makedirs(directory, exist_ok=True)

# Initialize database on startup
initialize_database()

# Load glucose prediction model globally
glucose_model_results = None
try:
    model_path = os.path.join(app.config.get('MODEL_FOLDER', 'models'), 'standard_glucose_model.pkl')
    if os.path.exists(model_path):
        glucose_model_results = joblib.load(model_path)
        logger.info("Glucose prediction model loaded successfully")
    else:
        logger.warning("Glucose prediction model not found at: " + model_path)
except Exception as e:
    logger.error(f"Error loading glucose prediction model: {str(e)}", exc_info=True)

# Define glucose prediction function using ML model
def predict_with_standard_model(model_results, meal_data):
    """
    Make a prediction using the standard glucose model

    Args:
        model_results: Dictionary containing the trained model and feature list
        meal_data: Dictionary with meal and patient features

    Returns:
        Dictionary with prediction results
    """
    if not model_results or 'model' not in model_results:
        return None
    
    # Extract components from model results
    model = model_results['model']
    features = model_results['features']
    
    # Convert meal data to DataFrame (single row)
    meal_df = pd.DataFrame([meal_data])
    
    # Ensure all required features are present
    for feature in features:
        if feature not in meal_df.columns:
            meal_df[feature] = 0  # Default value
    
    # Select only the features used by the model
    X_meal = meal_df[features]
    
    # Make prediction
    glucose_prediction = model.predict(X_meal)[0]
    
    # Calculate glucose category based on predicted value
    if glucose_prediction < 70:
        category = "Low"
    elif glucose_prediction < 140:
        category = "Normal"
    elif glucose_prediction < 180:
        category = "Elevated"
    else:
        category = "High"
    
    # If this is a time-series model with Minutes_After_Meal feature, generate time series
    time_series = []
    max_glucose_time = 0
    
    if 'Minutes_After_Meal' in features:
        # Predict glucose over a 2-hour period (5-minute intervals)
        time_series = []
        glucose_values = []
        
        for minute in range(0, 120, 5):
            time_data = meal_data.copy()
            time_data['Minutes_After_Meal'] = minute
            
            # Make prediction for this time point
            time_prediction = predict_with_standard_model_at_time(model_results, time_data)
            if time_prediction:
                time_series.append(minute)
                glucose_values.append(time_prediction['predicted_glucose'])
        
        if glucose_values:
            # Find the peak glucose value and time
            max_glucose = max(glucose_values)
            max_glucose_time = time_series[glucose_values.index(max_glucose)]
    
    return {
        'predicted_glucose': glucose_prediction,
        'glucose_category': category,
        'time_series': time_series,
        'max_glucose_time': max_glucose_time
    }

def predict_with_standard_model_at_time(model_results, meal_data):
    """Helper function to predict glucose at a specific time point"""
    if not model_results or 'model' not in model_results:
        return None
    
    # Extract components from model results
    model = model_results['model']
    features = model_results['features']
    
    # Convert meal data to DataFrame (single row)
    meal_df = pd.DataFrame([meal_data])
    
    # Ensure all required features are present
    for feature in features:
        if feature not in meal_df.columns:
            meal_df[feature] = 0  # Default value
    
    # Select only the features used by the model
    X_meal = meal_df[features]
    
    # Make prediction
    glucose_prediction = model.predict(X_meal)[0]
    
    return {
        'predicted_glucose': glucose_prediction
    }

# Generate dietary recommendations based on glucose prediction
def generate_recommendations(prediction, meal_nutrients, user_profile):
    """
    Generate personalized recommendations based on glucose prediction,
    meal nutrients, and user profile
    """
    if not prediction:
        return None
    
    guidelines = []
    recommendations = []
    
    # Guidelines based on glucose category
    category = prediction['glucose_category']
    glucose_level = prediction['predicted_glucose']
    
    # Basic guidelines based on glucose category
    if category == "Low":
        guidelines.append("This meal may cause your blood glucose to drop too low.")
        guidelines.append("Consider adding more complex carbohydrates to maintain stable blood glucose.")
    elif category == "Normal":
        guidelines.append("This meal should result in a healthy glucose response.")
        guidelines.append("This type of meal helps maintain good metabolic health.")
    elif category == "Elevated":
        guidelines.append("This meal may cause a moderate glucose spike.")
        guidelines.append("Consider reducing portion size or balancing with protein and healthy fats.")
    else:  # High
        guidelines.append("This meal may cause a significant glucose spike.")
        guidelines.append("Consider substituting high glycemic foods with lower glycemic alternatives.")
    
    # Specific recommendations based on meal content
    carbs = meal_nutrients.get('carbs', 0)
    fiber = meal_nutrients.get('fiber', 0)
    fat = meal_nutrients.get('fat', 0)
    protein = meal_nutrients.get('protein', 0)
    sugar = meal_nutrients.get('sugar', 0)
    
    # Carb to fiber ratio recommendations
    if carbs > 30 and fiber < 5:
        recommendations.append("This meal has a poor carb-to-fiber ratio. Adding fiber-rich vegetables or whole grains could improve the glucose response.")
    
    # Sugar recommendations
    if sugar > 15:
        recommendations.append("The sugar content in this meal is high, which can lead to rapid glucose spikes. Consider reducing added sugars.")
    
    # Macronutrient balance
    if carbs > 45 and fat < 10 and protein < 15:
        recommendations.append("This meal is high in carbs but low in protein and fat. Adding lean protein or healthy fats can slow glucose absorption.")
    
    # Meal timing recommendations
    if glucose_level > 160:
        recommendations.append("Consider consuming this meal earlier in the day when insulin sensitivity is typically higher.")
        
    # Exercise recommendations
    if glucose_level > 140:
        activity_level = user_profile.get('activity_level', 'moderate')
        if activity_level == 'sedentary' or activity_level == 'lightly_active':
            recommendations.append("A 15-20 minute walk after this meal could significantly reduce the glucose impact.")
        else:
            recommendations.append("Even for active individuals, light activity after this meal can help moderate glucose response.")
    
    # Diabetes-specific recommendations
    diabetes_status = user_profile.get('diabetes_status', 'none')
    if diabetes_status in ['pre_diabetic', 'type1_diabetes', 'type2_diabetes'] and glucose_level > 140:
        recommendations.append("With your diabetes status, this meal may require closer glucose monitoring or medication adjustment.")
    
    # Stress and sleep recommendations
    if category in ["Elevated", "High"]:
        recommendations.append("Stress and poor sleep can amplify glucose responses. Eating this meal when well-rested and relaxed is recommended.")
    
    return {
        "guidelines": guidelines,
        "recommendations": recommendations
    }

# API Routes

@health_ns.route('')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'glucose_model_loaded': glucose_model_results is not None}

@users_ns.route('')
class UserCreation(Resource):
    @api.expect(user_profile_model)
    def post(self):
        """Create a new user profile"""
        try:
            data = request.json
            logger.info("Received user creation request")
            
            # Validate user data
            validation_error = validate_user_data(data)
            if validation_error:
                logger.warning(f"Validation error: {validation_error}")
                return {'error': validation_error}, 400
            
            # Generate unique user ID
            user_id = str(uuid.uuid4())
            
            # Calculate BMI
            data['bmi'] = float(data['weight']) / ((float(data['height'])/100) ** 2)
            
            # Create user profile
            user_profile = {
                "user_id": user_id,
                "profile": data,
                "meals": [],
                "created_at": datetime.now().isoformat()
            }
            
            # Save user profile
            save_user_data(user_id, user_profile)

            # Train initial model based on user profile
            models.train_initial_model(user_id, data)
            
            return {"user_id": user_id}, 201
        
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}", exc_info=True)
            return {"error": f"Failed to create user: {str(e)}"}, 500

@users_ns.route('/<string:user_id>')
class UserManagement(Resource):
    def get(self, user_id):
        """Get user profile"""
        try:
            user_data = get_user_data(user_id)
            
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Remove sensitive info for response
            response_data = {
                "user_id": user_data["user_id"],
                "profile": user_data["profile"],
                "created_at": user_data["created_at"]
            }
            
            return response_data, 200
        
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get user profile"}, 500
            
    @api.expect(user_profile_model)
    def put(self, user_id):
        """Update user profile"""
        try:
            user_data = get_user_data(user_id)
            
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Update profile with new data
            data = request.json
            
            # Validate user data
            validation_error = validate_user_data(data, required_fields=False)
            if validation_error:
                return {"error": validation_error}, 400
            
            user_data['profile'].update(data)
            
            # Recalculate BMI if weight or height was updated
            if 'weight' in data or 'height' in data:
                weight = float(user_data['profile']['weight'])
                height = float(user_data['profile']['height'])
                user_data['profile']['bmi'] = weight / ((height/100) ** 2)
            
            # Save updated profile
            save_user_data(user_id, user_data)
            
            return {"message": "User profile updated successfully"}, 200
        
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to update user profile"}, 500


@food_ns.route('/analyze-food')
class FoodAnalysis(Resource):
    @api.expect(food_image_parser)
    def post(self):
        """Analyze food image with ML-based glucose prediction"""
        try:
            args = food_image_parser.parse_args()
            
            if 'food_image' not in request.files:
                return {"error": "No file part"}, 400
            
            file = request.files['food_image']
            user_id = args['user_id']
            
            # If user doesn't submit a file
            if file.filename == '':
                return {"error": "No selected file"}, 400
            
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Save the file if valid
            if file and allowed_file(file.filename):
                # Save the file
                filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Food image saved: {filepath}")

                # Identify food in the image using the REST API
                food_items = identify_food_in_image(filepath)
                
                if not food_items:
                    return {"error": "Could not identify food in the image"}, 400
                
                # Process each food item
                results = []
                total_nutrients = {
                    "calories": 0,
                    "carbs": 0,
                    "fat": 0,
                    "protein": 0,
                    "fiber": 0,
                    "sugar": 0
                }
                
                for food in food_items:
                    # The GI value is now directly from the API response
                    base_gi = food.get('gi', 55)  # Default to medium GI if not provided
                    
                    # Personalize GI impact
                    personalized = models.personalize_gi_impact(base_gi, user_data['profile'])
                    
                    # Use nutrients from API response
                    nutrients = food.get('nutrients', {})
                    
                    # Make sure all required nutrients are present
                    if 'fiber' not in nutrients:
                        nutrients['fiber'] = 2  # Default fiber value
                    if 'sugar' not in nutrients:
                        nutrients['sugar'] = nutrients.get('carbs', 0) * 0.2  # Estimate 20% of carbs as sugar
                    
                    # Add to total nutrients
                    for key in total_nutrients:
                        if key in nutrients:
                            total_nutrients[key] += nutrients[key]
                    
                    # Add to results
                    food_result = {
                        "food_name": food['name'],
                        "confidence": food['confidence'],
                        "portion_size": food.get('portion_size', '100g'),
                        "base_gi": base_gi,
                        "personalized_gi": personalized,
                        "nutrients": nutrients
                    }
                    results.append(food_result)
                
                # Calculate average GI and total GL for the meal
                total_carbs = total_nutrients['carbs']
                if total_carbs > 0:
                    # Use weighted average for GI calculation
                    meal_gi = sum(item['base_gi'] * item['nutrients'].get('carbs', 0) for item in results) / total_carbs
                    # GL = GI * carbs / 100
                    meal_gl = (meal_gi * total_carbs) / 100
                else:
                    # Default values if no carbs
                    meal_gi = 0
                    meal_gl = 0
                
                # Prepare data for glucose prediction
                meal_features = {
                    'Age': user_data['profile']['age'],
                    'Sex': 1 if user_data['profile']['gender'] == 'male' else 0,
                    'BMI': user_data['profile']['bmi'],
                    'Category': get_diabetes_category(user_data['profile'].get('diabetes_status', 'none')),
                    'Carbs': total_nutrients['carbs'],
                    'Fat': total_nutrients['fat'],
                    'Fiber': total_nutrients.get('fiber', 2),  # Default to 2g if not provided
                    'Protein': total_nutrients['protein'],
                    'GI': meal_gi,
                    'Glycemic_Load': meal_gl,
                    'Activity_Level': get_activity_level(user_data['profile'].get('activity_level', 'moderate')),
                    'Activity_Timing': 2,  # Default to 'none'
                    'Stress_Level': 5.0,  # Default to moderate stress
                    'Sleep_Quality': 7.0,  # Default to average sleep
                    'Takes_Medication': 1 if user_data['profile'].get('diabetes_status') in ['type1_diabetes', 'type2_diabetes'] else 0,
                    'Family_History': 1 if user_data['profile'].get('family_history') else 0,
                    'Time_Since_Last_Meal': 4.0  # Default to 4 hours
                }
                
                """
                # Check for custom person info from request to override defaults
                if args['person_info']:
                    try:
                        custom_info = json.loads(args['person_info'])
                        # Map custom fields to model features
                        field_mapping = {
                            'age': 'Age',
                            'gender': 'Sex',  # Convert in the next step
                            'bmi': 'BMI',
                            'diabetes_status': 'Category',  # Convert in the next step
                            'carbs': 'Carbs',
                            'fat': 'Fat',
                            'fiber': 'Fiber',
                            'protein': 'Protein',
                            'gi': 'GI',
                            'glycemic_load': 'Glycemic_Load',
                            'activity_level': 'Activity_Level',
                            'activity_timing': 'Activity_Timing',
                            'stress_level': 'Stress_Level',
                            'sleep_quality': 'Sleep_Quality',
                            'takes_medication': 'Takes_Medication',
                            'family_history': 'Family_History',
                            'time_since_last_meal': 'Time_Since_Last_Meal'
                        }
                        
                        for source_key, target_key in field_mapping.items():
                            if source_key in custom_info:
                                value = custom_info[source_key]
                                
                                # Special handling for gender
                                if source_key == 'gender':
                                    if isinstance(value, str):
                                        value = 1 if value.lower() == 'male' else 0
                                
                                # Special handling for diabetes status
                                if source_key == 'diabetes_status' and isinstance(value, str):
                                    value = get_diabetes_category(value)
                                
                                # Special handling for activity level
                                if source_key == 'activity_level' and isinstance(value, str):
                                    value = get_activity_level(value)
                                
                                meal_features[target_key] = value
                        
                    except Exception as e:
                        logger.warning(f"Error parsing person_info: {str(e)}")
                """
                # Make glucose prediction using ML model
                glucose_prediction = None
                if glucose_model_results and 'model' in glucose_model_results:
                    # Add 'Minutes_After_Meal' for time-series predictions if the model supports it
                    if 'Minutes_After_Meal' in glucose_model_results['features']:
                        # Generate time series predictions
                        time_series_predictions = []
                        for minute in range(0, 120, 5):  # Every 5 minutes for 2 hours
                            time_features = meal_features.copy()
                            time_features['Minutes_After_Meal'] = minute
                            
                            # Make prediction for this time point
                            prediction = predict_with_standard_model_at_time(glucose_model_results, time_features)
                            if prediction:
                                time_series_predictions.append({
                                    'minute': minute,
                                    'glucose': prediction['predicted_glucose']
                                })
                        
                        # Find peak glucose and time
                        if time_series_predictions:
                            glucose_values = [p['glucose'] for p in time_series_predictions]
                            max_glucose = max(glucose_values)
                            max_glucose_time = time_series_predictions[glucose_values.index(max_glucose)]['minute']
                            
                            # Determine glucose category
                            if max_glucose < 70:
                                category = "Low"
                            elif max_glucose < 140:
                                category = "Normal"
                            elif max_glucose < 180:
                                category = "Elevated"
                            else:
                                category = "High"
                            
                            glucose_prediction = {
                                'predicted_glucose': max_glucose,
                                'glucose_category': category,
                                'time_series': time_series_predictions,
                                'max_glucose_time': max_glucose_time
                            }
                    else:
                        # Single point prediction
                        glucose_prediction = predict_with_standard_model(glucose_model_results, meal_features)
                    
                    # Generate personalized recommendations
                    if glucose_prediction  and 'guidelines' not in glucose_prediction:
                        recommendations = generate_recommendations(
                            glucose_prediction, 
                            total_nutrients, 
                            user_data['profile']
                        )
                        
                        # Add recommendations to prediction
                        if recommendations:
                            glucose_prediction.update(recommendations)
                
                # Save this analysis to user history
                meal_id = str(uuid.uuid4())
                meal_data = {
                    "meal_id": meal_id,
                    "timestamp": datetime.now().isoformat(),
                    "food_items": results,
                    "image_path": filepath,
                    "total_nutrients": total_nutrients,
                    "glucose_prediction": glucose_prediction,
                    "meal_features": meal_features
                }
                
                if 'meals' not in user_data:
                    user_data['meals'] = []
                
                user_data['meals'].append(meal_data)
                
                # Save updated user data
                save_user_data(user_id, user_data)
                
                return {
                    "meal_id": meal_id,
                    "food_analysis": {
                        "identified_foods": results,
                        "total_nutrients": total_nutrients
                    },
                    "glucose_prediction": glucose_prediction
                }, 200
            
            return {"error": "Invalid file format"}, 400
        
        except Exception as e:
            logger.error(f"Error analyzing food: {str(e)}", exc_info=True)
            return {"error": f"Failed to analyze food: {str(e)}"}, 500

# These utility functions should be placed alongside the FoodAnalysis class
def get_diabetes_category(diabetes_status):
    """Map diabetes status to numerical category for model"""
    mapping = {
        'none': 0,
        'pre_diabetic': 1,
        'type1_diabetes': 2,
        'type2_diabetes': 2
    }
    return mapping.get(diabetes_status, 0)

def get_activity_level(activity_level):
    """Map activity level to numerical scale for model"""
    mapping = {
        'sedentary': 1.5,
        'lightly_active': 3.0,
        'moderately_active': 5.0,
        'very_active': 7.0,
        'extremely_active': 9.0
    }
    return mapping.get(activity_level, 5.0)

@food_ns.route('/analyze')
class FoodAnalysis(Resource):
    @api.expect(food_image_parser)
    def post(self):
        """Analyze food image with ML-based glucose prediction"""
        try:
            args = food_image_parser.parse_args()
            
            if 'food_image' not in request.files:
                return {"error": "No file part"}, 400
            
            file = request.files['food_image']
            user_id = args['user_id']
            
            # If user doesn't submit a file
            if file.filename == '':
                return {"error": "No selected file"}, 400
            
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Save the file if valid
            if file and allowed_file(file.filename):
                # Save the file
                filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Food image saved: {filepath}")

                # Identify food in the image
                food_items = identify_food_in_image(filepath)
                
                # Process each food item
                results = []
                total_nutrients = {
                    "calories": 0,
                    "carbs": 0,
                    "fat": 0,
                    "protein": 0,
                    "fiber": 0,
                    "sugar": 0
                }
                
                for food in food_items:
                    # Look up base GI value
                    base_gi = models.lookup_gi(food['name'])
                    
                    # Personalize GI impact
                    personalized = models.personalize_gi_impact(base_gi, user_data['profile'])
                    
                    # Get nutrition information
                    nutrients = models.get_nutrients(food['name'], food.get('portion_size', '1 serving'))
                    
                    # Add to total nutrients
                    for key in total_nutrients:
                        if key in nutrients:
                            total_nutrients[key] += nutrients[key]
                    
                    # Add to results
                    food_result = {
                        "food_name": food['name'],
                        "confidence": food['confidence'],
                        "base_gi": base_gi,
                        "personalized_gi": personalized,
                        "nutrients": nutrients
                    }
                    results.append(food_result)
                
                # Prepare data for glucose prediction
                meal_features = {
                    'Age': user_data['profile']['age'],
                    'Sex': 1 if user_data['profile']['gender'] == 'male' else 0,
                    'BMI': user_data['profile']['bmi'],
                    'Category': get_diabetes_category(user_data['profile'].get('diabetes_status', 'none')),
                    'Carbs': total_nutrients['carbs'],
                    'Fat': total_nutrients['fat'],
                    'Fiber': total_nutrients['fiber'],
                    'Protein': total_nutrients['protein'],
                    'GI': get_meal_gi(results),
                    'Glycemic_Load': (get_meal_gi(results) * total_nutrients['carbs']) / 100,
                    'Activity_Level': get_activity_level(user_data['profile'].get('activity_level', 'moderate')),
                    'Activity_Timing': 2,  # Default to 'none'
                    'Stress_Level': 5.0,  # Default to moderate stress
                    'Sleep_Quality': 7.0,  # Default to average sleep
                    'Takes_Medication': 1 if user_data['profile'].get('diabetes_status') in ['type1_diabetes', 'type2_diabetes'] else 0,
                    'Family_History': 1 if user_data['profile'].get('family_history') else 0,
                    'Time_Since_Last_Meal': 4.0  # Default to 4 hours
                }
                
                # Check for custom person info
                if args['person_info']:
                    try:
                        custom_info = json.loads(args['person_info'])
                        # Update meal features with custom info
                        for key, value in custom_info.items():
                            if key in meal_features:
                                meal_features[key] = value
                    except Exception as e:
                        logger.warning(f"Error parsing person_info: {str(e)}")
                
                # Make glucose prediction using ML model
                glucose_prediction = None
                if glucose_model_results:
                    glucose_prediction = predict_with_standard_model(glucose_model_results, meal_features)
                    
                    # Generate personalized recommendations
                    if glucose_prediction:
                        recommendations = generate_recommendations(
                            glucose_prediction, 
                            total_nutrients, 
                            user_data['profile']
                        )
                        
                        # Add recommendations to prediction
                        if recommendations:
                            glucose_prediction.update(recommendations)
                
                # Save this analysis to user history
                meal_id = str(uuid.uuid4())
                meal_data = {
                    "meal_id": meal_id,
                    "timestamp": datetime.now().isoformat(),
                    "food_items": results,
                    "image_path": filepath,
                    "total_nutrients": total_nutrients,
                    "glucose_prediction": glucose_prediction,
                    "meal_features": meal_features
                }
                
                if 'meals' not in user_data:
                    user_data['meals'] = []
                
                user_data['meals'].append(meal_data)
                
                # Save updated user data
                save_user_data(user_id, user_data)
                
                return {
                    "meal_id": meal_id,
                    "food_analysis": {
                        "identified_foods": results,
                        "total_nutrients": total_nutrients
                    },
                    "glucose_prediction": glucose_prediction
                }, 200
            
            return {"error": "Invalid file format"}, 400
        
        except Exception as e:
            logger.error(f"Error analyzing food: {str(e)}", exc_info=True)
            return {"error": f"Failed to analyze food: {str(e)}"}, 500

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_diabetes_category(diabetes_status):
    """Map diabetes status to numerical category for model"""
    mapping = {
        'none': 0,
        'pre_diabetic': 1,
        'type1_diabetes': 2,
        'type2_diabetes': 2
    }
    return mapping.get(diabetes_status, 0)

def get_activity_level(activity_level):
    """Map activity level to numerical scale for model"""
    mapping = {
        'sedentary': 1.5,
        'lightly_active': 3.0,
        'moderately_active': 5.0,
        'very_active': 7.0,
        'extremely_active': 9.0
    }
    return mapping.get(activity_level, 5.0)

def get_meal_gi(food_items):
    """Calculate weighted GI for an entire meal"""
    if not food_items:
        return 55  # Default to medium GI
    
    total_carbs = sum(item.get('nutrients', {}).get('carbs', 0) for item in food_items)
    if total_carbs == 0:
        return sum(item.get('base_gi', 55) for item in food_items) / len(food_items)
    
    # Calculate weighted GI based on carbohydrate contribution
    weighted_gi = sum(
        item.get('base_gi', 55) * item.get('nutrients', {}).get('carbs', 0) / total_carbs 
        for item in food_items
    )
    
    return weighted_gi

@food_ns.route('/<string:meal_id>/feedback')
class MealFeedback(Resource):
    def post(self, meal_id):
        """Add glucose readings or feedback for a meal"""
        try:
            data = request.json
            user_id = data.get('user_id')
            
            if not user_id:
                return {"error": "User ID is required"}, 400
            
            # Get glucose readings if provided
            glucose_readings = data.get('glucose_readings')
            
            # Get user feedback if provided
            user_feedback = data.get('feedback')
            
            # Update model with feedback
            success = models.update_model_with_feedback(
                user_id, 
                meal_id, 
                glucose_readings=glucose_readings, 
                user_feedback=user_feedback
            )
            
            if success:
                return {"message": "Feedback recorded and model updated"}, 200
            else:
                return {"message": "Feedback recorded, but not enough data to update model"}, 200
                
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {"error": f"Failed to process feedback: {str(e)}"}, 500

@training_ns.route('/info/<string:user_id>')
class ModelInfo(Resource):
    def get(self, user_id):
        """Get information about the user's model"""
        try:
            model_info = models.get_model_info(user_id)
            return model_info, 200
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": f"Failed to get model info: {str(e)}"}, 500


@training_ns.route('/<string:user_id>')
class ModelTraining(Resource):
    @api.expect(training_model)
    def post(self, user_id):
        """Train model for a specific user"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            data = request.json
            feature_data = data.get('feature_data', [])
            response_data = data.get('response_data', [])
            
            if not feature_data or not response_data:
                return {"error": "Missing training data"}, 400
            
            if len(feature_data) != len(response_data):
                return {"error": "Feature and response data must have the same length"}, 400
            
            # Train the model
            model_updated = models.update_user_model(user_id, feature_data, response_data)
            
            if model_updated:
                # Record model update in user data
                user_data['model_updated_at'] = datetime.now().isoformat()
                user_data['model_version'] = user_data.get('model_version', 0) + 1
                save_user_data(user_id, user_data)
                
                return {
                    "message": "Model trained successfully",
                    "model_version": user_data['model_version']
                }, 200
            else:
                return {"error": "Failed to train model"}, 500
            
        except Exception as e:
            logger.error(f"Error training model for user {user_id}: {str(e)}", exc_info=True)
            return {"error": f"Failed to train model: {str(e)}"}, 500

@meals_ns.route('/<string:user_id>')
class UserMeals(Resource):
    def get(self, user_id):
        """Get user's meal history"""
        try:
            # Get query parameters
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            food_type = request.args.get('food_type')
            sort_by = request.args.get('sort_by', 'date')
            sort_order = request.args.get('sort_order', 'desc')
            limit = request.args.get('limit', 50)
            
            if limit:
                try:
                    limit = int(limit)
                except ValueError:
                    limit = 50
            
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Get all meals
            meals = user_data.get('meals', [])
            
            # Apply filters
            filtered_meals = meals
            
            # Filter by date range
            if start_date:
                try:
                    start_datetime = datetime.fromisoformat(start_date)
                    filtered_meals = [
                        meal for meal in filtered_meals 
                        if datetime.fromisoformat(meal['timestamp']) >= start_datetime
                    ]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid start_date format: {start_date}")
            
            if end_date:
                try:
                    end_datetime = datetime.fromisoformat(end_date)
                    filtered_meals = [
                        meal for meal in filtered_meals 
                        if datetime.fromisoformat(meal['timestamp']) <= end_datetime
                    ]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid end_date format: {end_date}")
            
            # Filter by food type
            if food_type:
                filtered_meals = [
                    meal for meal in filtered_meals 
                    if any(food_type.lower() in food['food_name'].lower() for food in meal.get('food_items', []))
                ]
            
            # Sort results
            if sort_by == 'date':
                filtered_meals.sort(
                    key=lambda x: datetime.fromisoformat(x['timestamp']), 
                    reverse=(sort_order == 'desc')
                )
            elif sort_by == 'glucose_impact':
                # Sort by peak glucose prediction if available
                filtered_meals.sort(
                    key=lambda x: x.get('glucose_prediction', {}).get('predicted_glucose', 0), 
                    reverse=(sort_order == 'desc')
                )
            
            # Apply limit
            if limit and limit > 0:
                filtered_meals = filtered_meals[:limit]
            
            # Format response
            result = []
            for meal in filtered_meals:
                # Create a summary of the meal for the list view
                food_names = [food['food_name'] for food in meal.get('food_items', [])]
                glucose_prediction = meal.get('glucose_prediction', {})
                
                meal_summary = {
                    "meal_id": meal.get('meal_id'),
                    "timestamp": meal.get('timestamp'),
                    "foods": food_names,
                    "image_path": meal.get('image_path'),
                    "total_nutrients": meal.get('total_nutrients'),
                    "predicted_glucose": glucose_prediction.get('predicted_glucose'),
                    "glucose_category": glucose_prediction.get('glucose_category'),
                    "max_glucose_time": glucose_prediction.get('max_glucose_time')
                }
                
                result.append(meal_summary)
            
            return {"meals": result}, 200
        
        except Exception as e:
            logger.error(f"Error getting meals for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get meal history"}, 500

@meals_ns.route('/<string:user_id>/<string:meal_id>')
class MealDetail(Resource):
    def get(self, user_id, meal_id):
        """Get detailed information about a specific meal"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Find the meal
            meal = None
            for m in user_data.get('meals', []):
                if m.get('meal_id') == meal_id:
                    meal = m
                    break
            
            if not meal:
                return {"error": "Meal not found"}, 404
            
            # Return full meal details
            return {"meal": meal}, 200
        
        except Exception as e:
            logger.error(f"Error getting meal {meal_id} for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get meal details"}, 500

@meals_ns.route('/<string:user_id>/<string:meal_id>/feedback')
class MealFeedback(Resource):
    @api.expect(api.model('MealFeedbackSubmission', {
        'glucose_readings': fields.List(fields.Nested(api.model('GlucoseReading', {
            'timestamp': fields.DateTime(required=True, description='Timestamp of glucose reading'),
            'value': fields.Float(required=True, description='Glucose level')
        })), required=False),
        'user_response': fields.String(required=False, description='User feedback on meal',
                                      enum=['less_than_expected', 'as_expected', 'more_than_expected']),
        'notes': fields.String(required=False, description='Additional notes')
    }))
    def post(self, user_id, meal_id):
        """Submit feedback for a meal"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Find the meal
            meal_index = None
            for i, m in enumerate(user_data.get('meals', [])):
                if m.get('meal_id') == meal_id:
                    meal_index = i
                    break
            
            if meal_index is None:
                return {"error": "Meal not found"}, 404
            
            # Add feedback to meal
            data = request.json
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "glucose_readings": data.get('glucose_readings', []),
                "user_response": data.get('user_response'),
                "notes": data.get('notes', '')
            }
            
            # Update meal with feedback
            user_data['meals'][meal_index]['feedback'] = feedback
            
            # Update user's personal model if enough data points
            # This would call into your existing model update functions
            model_updated = False
            if 'glucose_readings' in data and data['glucose_readings']:
                # Process glucose readings for model update
                model_updated = models.update_model_with_feedback(
                    user_id,
                    meal_id,
                    glucose_readings=data['glucose_readings'],
                    user_feedback=data.get('user_response')
                )
            
            # Save updated user data
            save_user_data(user_id, user_data)
            
            response = {"message": "Feedback recorded successfully"}
            if model_updated:
                response["model_updated"] = True
                response["message"] = "Feedback recorded and personal model updated"
            
            return response, 200
        
        except Exception as e:
            logger.error(f"Error adding feedback for meal {meal_id}, user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to record feedback"}, 500

@meals_ns.route('/<string:user_id>/stats')
class MealStats(Resource):
    def get(self, user_id):
        """Get meal statistics for a user"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Get all meals
            meals = user_data.get('meals', [])
            
            if not meals:
                return {
                    "total_meals": 0,
                    "stats": {},
                    "message": "No meal data available"
                }, 200
            
            # Calculate statistics
            total_meals = len(meals)
            
            # Count meals by glucose category
            glucose_categories = {
                "Low": 0,
                "Normal": 0,
                "Elevated": 0,
                "High": 0
            }
            
            for meal in meals:
                prediction = meal.get('glucose_prediction', {})
                category = prediction.get('glucose_category')
                if category in glucose_categories:
                    glucose_categories[category] += 1
            
            # Find most common foods
            food_counter = {}
            for meal in meals:
                for food in meal.get('food_items', []):
                    food_name = food.get('food_name', '').lower()
                    if food_name:
                        food_counter[food_name] = food_counter.get(food_name, 0) + 1
            
            # Sort foods by frequency
            sorted_foods = sorted(food_counter.items(), key=lambda x: x[1], reverse=True)
            top_foods = [{"name": food[0], "count": food[1]} for food in sorted_foods[:10]]
            
            # Calculate average glucose impact
            glucose_values = [
                meal.get('glucose_prediction', {}).get('predicted_glucose', 0)
                for meal in meals
                if meal.get('glucose_prediction', {}).get('predicted_glucose') is not None
            ]
            
            avg_glucose = sum(glucose_values) / len(glucose_values) if glucose_values else 0
            
            # Identify problem foods (foods with high glucose impact)
            problem_foods = []
            for food_name in food_counter.keys():
                # Find meals containing this food
                food_meals = [
                    meal for meal in meals
                    if any(food.get('food_name', '').lower() == food_name for food in meal.get('food_items', []))
                ]
                
                # If this food appears in multiple meals, calculate average glucose impact
                if len(food_meals) >= 3:
                    food_glucose_values = [
                        meal.get('glucose_prediction', {}).get('predicted_glucose', 0)
                        for meal in food_meals
                        if meal.get('glucose_prediction', {}).get('predicted_glucose') is not None
                    ]
                    
                    if food_glucose_values:
                        avg_food_glucose = sum(food_glucose_values) / len(food_glucose_values)
                        
                        # If average glucose is higher than overall average + 20, consider it a problem food
                        if avg_food_glucose > avg_glucose + 20:
                            problem_foods.append({
                                "name": food_name,
                                "avg_glucose": avg_food_glucose,
                                "occurrence": len(food_meals)
                            })
            
            # Sort problem foods by average glucose impact
            problem_foods.sort(key=lambda x: x['avg_glucose'], reverse=True)
            
            # Calculate trend over time (weekly averages)
            weekly_trends = []
            if meals:
                # Convert timestamps to datetime objects
                for meal in meals:
                    meal['datetime'] = datetime.fromisoformat(meal['timestamp'])
                
                # Sort meals by date
                sorted_meals = sorted(meals, key=lambda x: x['datetime'])
                
                # Calculate start and end dates
                start_date = sorted_meals[0]['datetime']
                end_date = sorted_meals[-1]['datetime']
                
                # Calculate weekly averages
                current_week = start_date
                while current_week <= end_date:
                    next_week = current_week + timedelta(days=7)
                    
                    # Find meals in this week
                    week_meals = [
                        meal for meal in sorted_meals
                        if current_week <= meal['datetime'] < next_week
                    ]
                    
                    if week_meals:
                        # Calculate average glucose for this week
                        week_glucose_values = [
                            meal.get('glucose_prediction', {}).get('predicted_glucose', 0)
                            for meal in week_meals
                            if meal.get('glucose_prediction', {}).get('predicted_glucose') is not None
                        ]
                        
                        week_avg_glucose = sum(week_glucose_values) / len(week_glucose_values) if week_glucose_values else 0
                        
                        weekly_trends.append({
                            "week": current_week.strftime("%Y-%m-%d"),
                            "avg_glucose": week_avg_glucose,
                            "meal_count": len(week_meals)
                        })
                    
                    current_week = next_week
            
            return {
                "total_meals": total_meals,
                "stats": {
                    "glucose_categories": glucose_categories,
                    "avg_glucose": avg_glucose,
                    "top_foods": top_foods,
                    "problem_foods": problem_foods[:5],  # Limit to top 5
                    "weekly_trends": weekly_trends
                }
            }, 200
        
        except Exception as e:
            logger.error(f"Error getting meal stats for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get meal statistics"}, 500


# Application entry point
def create_app(config_class=Config):
    return app

# Run the application when executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

# Add this at the module level
app = create_app()
