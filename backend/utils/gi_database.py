"""
Main Flask application entry point for GI Personalize app with Swagger API documentation.
"""
from flask import Flask, request, jsonify
import os
import uuid
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from flask_restx import Api, Resource, fields

# Import modules
from utils.database import get_user_data, save_user_data, initialize_database
from utils.food_recognition import identify_food_in_image, get_nutritional_info
from utils.validators import validate_user_data, validate_glucose_readings
import ml_models
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
calibration_ns = api.namespace('calibration', description='Glucose Calibration Endpoints')
meals_ns = api.namespace('meals', description='Meal Management')

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

glucose_reading_model = api.model('GlucoseReading', {
    'timestamp': fields.DateTime(required=True, description='Timestamp of glucose reading'),
    'value': fields.Float(required=True, description='Glucose level')
})

food_analysis_model = api.model('FoodAnalysis', {
    'food_name': fields.String(required=True, description='Identified food name'),
    'confidence': fields.Float(description='Confidence of food identification'),
    'base_gi': fields.Float(description='Base Glycemic Index'),
    'personalized_gi': fields.Raw(description='Personalized GI impact')
})

meal_response_model = api.model('MealResponse', {
    'response': fields.String(required=True, description='User response to meal',
                             enum=['less_than_expected', 'as_expected', 'more_than_expected'])
})

# Ensure required directories exist
for directory in [app.config['UPLOAD_FOLDER'], 
                 app.config['USER_DATA_FOLDER'], 
                 'logs']:
    os.makedirs(directory, exist_ok=True)

# Initialize database on startup
initialize_database()

# API Routes
@health_ns.route('')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy'}

@users_ns.route('')
class UserCreation(Resource):
    @api.expect(user_profile_model)
    def post(self):
        """Create a new user profile"""
        try:
            data = request.json
            print("===============Inside Create User Method =================== ")
            print (f"Received Request Payload: {data}")
            
            # Validate user data
            validation_error = validate_user_data(data)
            if validation_error:
                print(f"Validation Errr:---> {validation_error}")
                return {'error': validation_error}, 400
            
            # Generate unique user ID
            user_id = str(uuid.uuid4())
            
            # Calculate BMI
            data['bmi'] = float(data['weight']) / ((float(data['height'])/100) ** 2)
            print(f"BMI Calculation Results: ---> {data['bmi']}")
            
            # Create user profile
            user_profile = {
                "user_id": user_id,
                "profile": data,
                "meals": [],
                "calibration": {
                    # Add default calibration based on profile
                    "calibration_factor": 1.0,
                    "timestamp": datetime.now().isoformat()
                },
                "created_at": datetime.now().isoformat()
            }
            
            # Save user profile
            save_user_data(user_id, user_profile)
            
            return {"user_id": user_id}, 201
        
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}", exc_info=True)
            return {"error": f"Failed to create user: {str(e)}"}, 500

@users_ns.route('/<string:user_id>')
class UserManagement(Resource):
    def get(self, user_id):
        """Get user profile"""
        try:
            print("===============Inside GET/Retrieve User Method =================== ")
            user_data = get_user_data(user_id)
            
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Remove sensitive info for response
            response_data = {
                "user_id": user_data["user_id"],
                "profile": user_data["profile"],
                "calibration": user_data.get("calibration", {}),
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
            print("===============Inside Update User Method =================== ")
            user_data = get_user_data(user_id)
            
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Update profile with new data
            data = request.json
            print (f"Received Request Payload: {data}")

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
            
            return {"message": "User profile updated"}, 200
        
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to update user profile"}, 500

@food_ns.route('/analyze')
class FoodAnalysis(Resource):
    @api.doc(params={'food_image': 'Food image file', 'user_id': 'User ID'})
    @api.expect(api.parser().add_argument('food_image', location='files', type='file', required=True, help='Food image file'))
    def post(self):
        """Analyze food image and provide personalized GI info"""
        try:
            # Check if the post request has the file part
            if 'food_image' not in request.files:
                return {"error": "No file part"}, 400
            
            file = request.files['food_image']
            user_id = request.form.get('user_id')
            
            # If user doesn't submit a file
            if file.filename == '':
                return {"error": "No selected file"}, 400
            
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            if file and allowed_file(file.filename):
                # Save the file
                filename = secure_filename(f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"Media file is saved successfully in the filepath: {filepath}")

                # Identify food in the image
                food_items = identify_food_in_image(filepath)
                
                # Get personalized GI values
                results = []
                for food in food_items:
                    # Get nutritional info for the food
                    nutritional_info = get_nutritional_info(food['name'])
                    
                    # Look up base GI value from database or use default
                    # First try to read from the GI database using pandas
                    try:
                        import pandas as pd
                        gi_data = pd.read_csv(os.path.join('data', 'gi_database.csv'))
                        gi_row = gi_data[gi_data['food_name'] == food['name']]
                        if not gi_row.empty:
                            base_gi = float(gi_row.iloc[0]['glycemic_index'])
                        else:
                            # Use default moderate GI if not found
                            base_gi = 50
                    except Exception as e:
                        logger.warning(f"Error reading GI database: {str(e)}")
                        base_gi = 50
                    
                    # Create food item with nutritional info
                    food_item = {
                        "name": food['name'],
                        "base_gi": base_gi,
                        "nutritional_info": nutritional_info
                    }
                    
                    # Personalize GI impact using ML model
                    personalized = ml_models.personalize_gi_impact(base_gi, user_data['profile'])
                    
                    # Add food analysis result
                    food_result = {
                        "food_name": food['name'],
                        "confidence": food['confidence'],
                        "base_gi": base_gi,
                        "personalized_gi": personalized,
                        "nutritional_info": nutritional_info
                    }
                    results.append(food_result)
                
                # Save this analysis to user history
                meal_id = str(uuid.uuid4())
                meal_data = {
                    "meal_id": meal_id,
                    "timestamp": datetime.now().isoformat(),
                    "food_items": results,
                    "image_path": filepath
                }
                user_data['meals'].append(meal_data)
                
                # Save updated user data
                save_user_data(user_id, user_data)
                
                return {
                    "meal_id": meal_id,
                    "results": results
                }, 200
            
            return {"error": "Invalid file format"}, 400
        
        except Exception as e:
            logger.error(f"Error analyzing food: {str(e)}", exc_info=True)
            return {"error": "Failed to analyze food"}, 500

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@calibration_ns.route('/<string:user_id>')
class CalibrationSubmission(Resource):
    @api.expect(api.model('CalibrationSubmission', {
        'glucose_readings': fields.List(fields.Nested(glucose_reading_model), required=True)
    }))
    def post(self, user_id):
        """Submit calibration meal data"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Get glucose readings from request
            data = request.json
            if 'glucose_readings' not in data:
                return {"error": "Missing glucose readings"}, 400
            
            glucose_readings = data['glucose_readings']
            
            # Validate glucose readings
            validation_error = validate_glucose_readings(glucose_readings)
            if validation_error:
                return {"error": validation_error}, 400
            
            # Process calibration meal using ML model
            response_factor = ml_models.process_calibration_meal(glucose_readings)
            
            # Update user profile with calibration data
            user_data['calibration'] = {
                "calibration_factor": response_factor,
                "timestamp": datetime.now().isoformat(),
                "readings": glucose_readings,
                "auto_calibrated": True
            }
            
            # Save updated profile
            save_user_data(user_id, user_data)
            
            return {
                "calibration_factor": response_factor,
                "message": "Calibration completed successfully"
            }, 200
        
        except Exception as e:
            logger.error(f"Error processing calibration for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to process calibration"}, 500

    def get(self, user_id):
        """Get calibration status"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Check if calibration exists
            if 'calibration' not in user_data or not user_data['calibration']:
                # Auto-calibrate based on user profile
                profile = user_data['profile']
                
                # Generate simulated glucose readings based on profile
                fasting_glucose = float(profile.get('fasting_glucose', 5.0))
                hba1c = float(profile.get('hba1c', 5.5))
                age = float(profile.get('age', 35))
                bmi = float(profile.get('bmi', 25.0))
                
                # Simple auto-calibration factor based on profile
                # Higher values indicate higher sensitivity to carbs
                diabetes_risk = 1.0
                if hba1c > 6.0:
                    diabetes_risk = 1.5
                if fasting_glucose > 6.0:
                    diabetes_risk += 0.3
                if age > 50:
                    diabetes_risk += 0.2
                if bmi > 30:
                    diabetes_risk += 0.3
                
                # Normalize to reasonable range (0.5 - 2.0)
                auto_factor = max(0.5, min(2.0, diabetes_risk))
                
                # Save auto-calibration
                user_data['calibration'] = {
                    "calibration_factor": auto_factor,
                    "timestamp": datetime.now().isoformat(),
                    "auto_calibrated": True
                }
                
                save_user_data(user_id, user_data)
                
                return {
                    "calibration_factor": auto_factor,
                    "auto_calibrated": True,
                    "message": "Auto-calibration completed"
                }, 200
            
            # Return existing calibration
            return user_data['calibration'], 200
            
        except Exception as e:
            logger.error(f"Error getting calibration for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get calibration status"}, 500

@meals_ns.route('/<string:user_id>/<string:meal_id>/response')
class MealResponse(Resource):
    @api.expect(meal_response_model)
    def post(self, user_id, meal_id):
        """Log user's response to a meal"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Find the meal
            meal_found = False
            for meal in user_data['meals']:
                if meal['meal_id'] == meal_id:
                    # Add user response
                    meal['user_response'] = request.json
                    meal['response_time'] = datetime.now().isoformat()
                    meal_found = True
                    break
            
            if not meal_found:
                return {"error": "Meal not found"}, 404
            
            # Update user's personal model if enough data points
            model_updated = False
            meals_with_responses = [m for m in user_data['meals'] if 'user_response' in m]
            
            if len(meals_with_responses) >= 3:  # Reduced from 5 to 3 for demonstration
                # Prepare training data
                feature_data, response_data = ml_models.prepare_training_data(meals_with_responses)
                
                # Train model if we have enough data
                if len(feature_data) > 0 and len(response_data) > 0:
                    # Update user's personalized model
                    model_updated = ml_models.update_user_model(user_id, feature_data, response_data)
                    
                    # Record model update
                    if model_updated:
                        user_data['model_updated_at'] = datetime.now().isoformat()
                        user_data['model_version'] = user_data.get('model_version', 0) + 1
            
            # Save updated user data
            save_user_data(user_id, user_data)
            
            response = {"message": "Response recorded"}
            if model_updated:
                response["model_updated"] = True
                response["message"] = "Your personal profile has been updated based on your response"
            
            return response, 200
        
        except Exception as e:
            logger.error(f"Error logging meal response for user {user_id}, meal {meal_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to log meal response"}, 500

@meals_ns.route('/<string:user_id>')
class UserMeals(Resource):
    def get(self, user_id):
        """Get user's meal history"""
        try:
            # Check if user exists
            user_data = get_user_data(user_id)
            if not user_data:
                return {"error": "User not found"}, 404
            
            # Return meals
            return {"meals": user_data.get('meals', [])}, 200
        
        except Exception as e:
            logger.error(f"Error getting meals for user {user_id}: {str(e)}", exc_info=True)
            return {"error": "Failed to get meal history"}, 500

# Application entry point
def create_app(config_class=Config):
    return app

# Run the application when executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

# Add this at the module level
app = create_app()
