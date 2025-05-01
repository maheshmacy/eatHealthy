from flask import Flask, request
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.datastructures import FileStorage
import os
import json
import pandas as pd
import numpy as np
import logging
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from glucose_prediction_model import GlucoseResponsePredictor
# Suppress AVX instructions warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
api = Api(app, version='1.0', 
          title='EAT Smart AI API',
          description='API for predicting personalized glucose responses based on individual characteristics and meal composition',
          doc='/swagger')

# Create namespaces for API organization
health_ns = Namespace('health', description='API health and status')
prediction_ns = Namespace('prediction', description='Glucose response prediction endpoints')
meal_ns = Namespace('meal', description='Meal analysis endpoints')
image_ns = Namespace('image', description='Food image analysis endpoints')
model_ns = Namespace('model', description='Model management endpoints')

# Add namespaces to the API
api.add_namespace(health_ns, path='/api/health')
api.add_namespace(prediction_ns, path='/api/prediction')
api.add_namespace(meal_ns, path='/api/meal')
api.add_namespace(image_ns, path='/api/image')
api.add_namespace(model_ns, path='/api/model')

# Initialize the model
predictor = GlucoseResponsePredictor()

# Check if we have a saved model
MODEL_PATH = "glucose_model.joblib"
if os.path.exists(MODEL_PATH):
    # Load existing model
    predictor.load_model(MODEL_PATH)
else:
    # Train a new model
    if os.path.exists("sample_meals.csv"):
        predictor.train_model("sample_meals.csv")
        predictor.save_model(MODEL_PATH)
    else:
        print("Warning: No training data found. Model not initialized!")

# Food recognition model configuration
FOOD_MODEL_PATH = "model/food_classifier_pro.h5"
DATASET_PATH = "dataset/train"
IMG_SIZE = 224
"""
# Setup GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
"""
# Load food recognition model
try:
    food_model = tf.keras.models.load_model(FOOD_MODEL_PATH)
    if os.path.exists(DATASET_PATH):
        class_names = sorted(os.listdir(DATASET_PATH))
        logging.info(f"Food model loaded successfully. Found {len(class_names)} classes")
    else:
        class_names = []
        logging.warning("Dataset path not found. Class names will not be available.")
except Exception as e:
    logging.error(f"Failed to load food model: {e}")
    food_model = None
    class_names = []

# Load glycemic index dataset
try:
    gi_df = pd.read_csv("glycemic_index.csv")
except Exception as e:
    logging.error(f"Failed to load glycemic index data: {e}")
    gi_df = pd.DataFrame(columns=['food', 'glycemic_index'])

# Define GI lookup function
def lookup_gi_from_csv(dish_name):
    try:
        row = gi_df[gi_df['food'].str.contains(dish_name, case=False, na=False)]
        if not row.empty:
            return int(row.iloc[0]['glycemic_index'])
        else:
            return "Unknown"
    except Exception as e:
        logging.error(f"GI CSV lookup failed: {e}")
        return "Unknown"

# Define GL calculation function
def calculate_gl(carbs, gi):
    if isinstance(gi, int) and carbs:
        return round((float(carbs) * gi) / 100, 2)
    return "Unknown"

# Nutrition client class
class NutritionClient:
    def __init__(self):
        self.session = requests.Session()
        self.api_key = os.getenv('SPOONACULAR_API_KEY')
        if not self.api_key:
            logging.warning("No Spoonacular API key found")

    def fetch_recipe_id(self, dish_name):
        try:
            search_url = "https://api.spoonacular.com/recipes/complexSearch"
            params = {
                "apiKey": self.api_key,
                "query": dish_name,
                "number": 1
            }
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                return data["results"][0]["id"]
            else:
                return None
        except Exception as e:
            logging.error(f"Failed to fetch recipe ID: {e}")
            return None

    def fetch_nutrition(self, dish_name, normalize_to_100g=True):
        try:
            recipe_id = self.fetch_recipe_id(dish_name)
            if recipe_id is None:
                return {}

            nutrition_url = f"https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json"
            params = {"apiKey": self.api_key}
            response = self.session.get(nutrition_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract fields
            calories = float(data["calories"].replace("kcal", "").strip())
            carbs = float(data["carbs"].replace("g", "").strip())
            fat = float(data["fat"].replace("g", "").strip())
            protein = float(data["protein"].replace("g", "").strip())
            fiber = float(data.get("fiber", "0g").replace("g", "").strip())
            sugar = float(data.get("sugar", "0g").replace("g", "").strip())
            serving_weight = float(data.get("serving_weight_grams", 100))

            # Normalize to 100g if needed
            if normalize_to_100g and serving_weight != 100:
                factor = 100 / serving_weight
                calories = round(calories * factor, 2)
                carbs = round(carbs * factor, 2)
                fat = round(fat * factor, 2)
                protein = round(protein * factor, 2)
                fiber = round(fiber * factor, 2)
                sugar = round(sugar * factor, 2)

            nutrition = {
                "calories": calories,
                "carbohydrates": carbs,
                "fat": fat,
                "protein": protein,
                "fiber": fiber,
                "sugar": sugar,
                "serving_weight_grams": serving_weight
            }
            return nutrition

        except Exception as e:
            logging.error(f"Failed to fetch detailed nutrition: {e}")
            return {}

    def close(self):
        self.session.close()

# Initialize nutrition client
nutrition_client = NutritionClient()

# Define request and response models for Swagger documentation

# Person model
person_model = api.model('Person', {
    'bmi': fields.Float(required=True, description='Body Mass Index', example=25.0),
    'age': fields.Integer(required=True, description='Age in years', example=35),
    'sex': fields.String(required=True, description='Sex (M or F)', example='M', enum=['M', 'F'])
})

# Meal model
meal_model = api.model('Meal', {
    'calories': fields.Float(required=True, description='Total calories', example=500),
    'carbs': fields.Float(required=True, description='Carbohydrates in grams', example=60),
    'fat': fields.Float(required=True, description='Fat in grams', example=15),
    'protein': fields.Float(required=True, description='Protein in grams', example=25),
    'fiber': fields.Float(required=True, description='Fiber in grams', example=5),
    'sugar': fields.Float(required=True, description='Sugar in grams', example=20)
})

# Prediction request model
prediction_request_model = api.model('PredictionRequest', {
    'person': fields.Nested(person_model, required=True),
    'meal': fields.Nested(meal_model, required=True)
})

# Meal analysis request model
meal_analysis_request_model = api.model('MealAnalysisRequest', {
    'meal': fields.Nested(meal_model, required=True)
})

# Nutrient composition model
nutrient_composition_model = api.model('NutrientComposition', {
    'carbs_g': fields.Float(description='Carbohydrates in grams'),
    'fat_g': fields.Float(description='Fat in grams'),
    'protein_g': fields.Float(description='Protein in grams'),
    'fiber_g': fields.Float(description='Fiber in grams'),
    'sugar_g': fields.Float(description='Sugar in grams'),
    'carbs_pct': fields.Float(description='Carbohydrates percentage of total macros'),
    'fat_pct': fields.Float(description='Fat percentage of total macros'),
    'protein_pct': fields.Float(description='Protein percentage of total macros')
})

# Calorie distribution model
calorie_distribution_model = api.model('CalorieDistribution', {
    'total_calories': fields.Float(description='Total calories'),
    'carbs_calories': fields.Float(description='Calories from carbohydrates'),
    'fat_calories': fields.Float(description='Calories from fat'),
    'protein_calories': fields.Float(description='Calories from protein'),
    'carbs_cal_pct': fields.Float(description='Carbohydrates percentage of total calories'),
    'fat_cal_pct': fields.Float(description='Fat percentage of total calories'),
    'protein_cal_pct': fields.Float(description='Protein percentage of total calories')
})

# Nutrient ratios model
nutrient_ratios_model = api.model('NutrientRatios', {
    'fiber_to_carb_ratio': fields.Float(description='Ratio of fiber to carbohydrates'),
    'protein_to_carb_ratio': fields.Float(description='Ratio of protein to carbohydrates')
})

# Glycemic impact factors model
glycemic_impact_model = api.model('GlycemicImpactFactors', {
    'meal_characteristics': fields.List(fields.String, description='Meal characteristics based on macronutrient content'),
    'sugar_to_carb_ratio': fields.Float(description='Ratio of sugar to carbohydrates'),
    'estimated_glycemic_impact': fields.String(description='Estimated glycemic impact (Low, Medium, High)')
})

# Meal analysis response model
meal_analysis_model = api.model('MealAnalysis', {
    'macronutrient_composition': fields.Nested(nutrient_composition_model),
    'calorie_distribution': fields.Nested(calorie_distribution_model),
    'nutrient_ratios': fields.Nested(nutrient_ratios_model),
    'glycemic_impact_factors': fields.Nested(glycemic_impact_model)
})

# Meal analysis response with recommendations
meal_analysis_response_model = api.model('MealAnalysisResponse', {
    'analysis': fields.Nested(meal_analysis_model),
    'recommendations': fields.List(fields.String, description='Dietary recommendations')
})

# Prediction response model
prediction_response_model = api.model('PredictionResponse', {
    'iauc_prediction': fields.Float(description='Predicted insulin area under curve'),
    'risk_level': fields.String(description='Glucose response risk level (Low, Medium, High)'),
    'percentile': fields.Integer(description='Percentile in population'),
    'explanation': fields.String(description='Explanation of prediction factors'),
    'guidelines': fields.List(fields.String, description='General guidelines based on risk level'),
    'recommendations': fields.List(fields.String, description='Personalized dietary recommendations')
})

# Health check response model
health_response_model = api.model('HealthResponse', {
    'status': fields.String(description='API status'),
    'model_loaded': fields.Boolean(description='Whether the prediction model is loaded')
})

# Training response model
training_response_model = api.model('TrainingResponse', {
    'status': fields.String(description='Training status (success or error)'),
    'message': fields.String(description='Training result message'),
    'details': fields.Raw(description='Detailed training results')
})

# Food nutrient model
food_nutrient_model = api.model('FoodNutrient', {
    'calories': fields.Float(description='Calories'),
    'carbs': fields.Float(description='Carbohydrates in grams'),
    'fat': fields.Float(description='Fat in grams'),
    'protein': fields.Float(description='Protein in grams'),
    'fiber': fields.Float(description='Fiber in grams'),
    'sugar': fields.Float(description='Sugar in grams')
})

# Detected food item model
food_item_model = api.model('FoodItem', {
    'name': fields.String(description='Food name'),
    'confidence': fields.Float(description='Detection confidence score'),
    'portion_size': fields.String(description='Estimated portion size'),
    'nutrients': fields.Nested(food_nutrient_model, description='Nutrient information')
})

# Food analysis response model
food_analysis_model = api.model('FoodAnalysis', {
    'identified_foods': fields.List(fields.Nested(food_item_model), description='Detected food items'),
    'total_nutrients': fields.Nested(food_nutrient_model, description='Total nutrient content')
})

# Food image analysis response model
food_image_response_model = api.model('FoodImageResponse', {
    'food_analysis': fields.Nested(food_analysis_model),
    'glucose_prediction': fields.Nested(prediction_response_model)
})

# Upload parser for file uploads
upload_parser = api.parser()
upload_parser.add_argument('data_file', location='files', type=FileStorage, required=True, help='CSV training data file')

# Food image parser
food_image_parser = api.parser()
food_image_parser.add_argument('food_image', location='files', type=FileStorage, required=True, help='Food image file')
food_image_parser.add_argument('person_info', location='form', type=str, required=False, help='Person information in JSON format')

# Define API endpoints

@health_ns.route('')
class HealthCheck(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_response_model)
    def get(self):
        """Check if the API is running and model is loaded"""
        return {
            'status': 'ok',
            'model_loaded': predictor.model is not None
        }

@prediction_ns.route('/glucose')
class GlucoseResponsePrediction(Resource):
    @prediction_ns.doc('predict_glucose_response')
    @prediction_ns.expect(prediction_request_model)
    @prediction_ns.marshal_with(prediction_response_model, code=200)
    @prediction_ns.response(400, 'Validation Error')
    @prediction_ns.response(503, 'Model Not Loaded')
    def post(self):
        """Predict glucose response based on person and meal details"""
        # Check if model is loaded
        if predictor.model is None:
            api.abort(503, "Model not loaded. Please train the model first.")
        
        # Parse request data
        data = request.json
        
        # Validate input
        try:
            person = data.get('person', {})
            meal = data.get('meal', {})
            
            # Make prediction
            prediction = predictor.predict_glucose_response(person, meal)
            
            # Enhance response with general guidelines
            guidelines = []
            
            if prediction['risk_level'] == 'High':
                guidelines.append("Consider reducing carbohydrates or adding more fiber to this meal.")
                guidelines.append("Pairing carbohydrates with protein and healthy fats may help reduce glucose impact.")
            elif prediction['risk_level'] == 'Medium':
                guidelines.append("This meal has a moderate impact on blood glucose.")
                guidelines.append("Consider physical activity after eating to help manage glucose levels.")
            else:
                guidelines.append("This meal should have minimal impact on blood glucose levels.")
            
            # Add personalized recommendations
            recommendations = []
            
            # Check if high carbs with low fiber
            if meal['carbs'] > 50 and meal['fiber'] < 5:
                recommendations.append("This meal is high in carbs and low in fiber. Adding fiber can help reduce glucose impact.")
            
            # Check if high sugar
            if meal['sugar'] > 25:
                recommendations.append("This meal is high in sugar. Consider reducing added sugars.")
            
            # Check if balanced macros
            total_cals = meal['carbs'] * 4 + meal['fat'] * 9 + meal['protein'] * 4
            carb_pct = (meal['carbs'] * 4 / total_cals) * 100 if total_cals > 0 else 0
            
            if carb_pct > 60:
                recommendations.append("This meal is very high in carbohydrates. Consider adding more protein or healthy fats.")
            
            # Enhance the response
            enhanced_response = {
                **prediction,
                "guidelines": guidelines,
                "recommendations": recommendations
            }
            
            return enhanced_response
            
        except Exception as e:
            api.abort(400, f"Error processing request: {str(e)}")

@meal_ns.route('/analyze')
class MealAnalysis(Resource):
    @meal_ns.doc('analyze_meal')
    @meal_ns.expect(meal_analysis_request_model)
    @meal_ns.marshal_with(meal_analysis_response_model, code=200)
    @meal_ns.response(400, 'Validation Error')
    def post(self):
        """Analyze a meal without making a full glucose prediction"""
        # Parse request data
        data = request.json
        if not data or 'meal' not in data:
            api.abort(400, "No meal data provided")
        
        meal = data['meal']
        
        # Analyze meal composition
        try:
            total_macros = meal['carbs'] + meal['fat'] + meal['protein']
            carb_pct = (meal['carbs'] / total_macros) * 100 if total_macros > 0 else 0
            fat_pct = (meal['fat'] / total_macros) * 100 if total_macros > 0 else 0
            protein_pct = (meal['protein'] / total_macros) * 100 if total_macros > 0 else 0
            
            # Calculate calorie distribution
            carb_cals = meal['carbs'] * 4
            fat_cals = meal['fat'] * 9
            protein_cals = meal['protein'] * 4
            total_cals = carb_cals + fat_cals + protein_cals
            
            carb_cal_pct = (carb_cals / total_cals) * 100 if total_cals > 0 else 0
            fat_cal_pct = (fat_cals / total_cals) * 100 if total_cals > 0 else 0
            protein_cal_pct = (protein_cals / total_cals) * 100 if total_cals > 0 else 0
            
            # Calculate fiber-to-carb ratio
            fiber_to_carb = meal['fiber'] / meal['carbs'] if meal['carbs'] > 0 else 0
            
            # Analyze meal characteristics
            characteristics = []
            
            if carb_pct > 60:
                characteristics.append("High-carbohydrate meal")
            elif carb_pct < 20:
                characteristics.append("Low-carbohydrate meal")
                
            if fat_pct > 40:
                characteristics.append("High-fat meal")
            elif fat_pct < 15:
                characteristics.append("Low-fat meal")
                
            if protein_pct > 30:
                characteristics.append("High-protein meal")
            elif protein_pct < 10:
                characteristics.append("Low-protein meal")
                
            if meal['fiber'] > 8:
                characteristics.append("High-fiber meal")
            elif meal['fiber'] < 2:
                characteristics.append("Low-fiber meal")
                
            if meal['sugar'] > 25:
                characteristics.append("High-sugar meal")
            
            # Generate recommendations
            recommendations = []
            
            if meal['fiber'] < 5 and meal['carbs'] > 30:
                recommendations.append("Consider adding more fiber to help slow glucose absorption.")
                
            if meal['sugar'] > 20 and meal['sugar'] / meal['carbs'] > 0.5:
                recommendations.append("This meal is high in sugar relative to total carbs. Consider reducing added sugars.")
                
            if fiber_to_carb < 0.1 and meal['carbs'] > 30:
                recommendations.append("The fiber-to-carb ratio is low. Adding fiber may help moderate glucose response.")
                
            if fat_pct < 10 and carb_pct > 60:
                recommendations.append("Consider adding healthy fats to help balance this high-carb meal.")
            
            # Estimate glycemic impact
            glycemic_impact = estimate_glycemic_impact(meal)
            
            return {
                "analysis": {
                    "macronutrient_composition": {
                        "carbs_g": meal['carbs'],
                        "fat_g": meal['fat'],
                        "protein_g": meal['protein'],
                        "fiber_g": meal['fiber'],
                        "sugar_g": meal['sugar'],
                        "carbs_pct": round(carb_pct, 1),
                        "fat_pct": round(fat_pct, 1),
                        "protein_pct": round(protein_pct, 1)
                    },
                    "calorie_distribution": {
                        "total_calories": round(total_cals, 0),
                        "carbs_calories": round(carb_cals, 0),
                        "fat_calories": round(fat_cals, 0),
                        "protein_calories": round(protein_cals, 0),
                        "carbs_cal_pct": round(carb_cal_pct, 1),
                        "fat_cal_pct": round(fat_cal_pct, 1),
                        "protein_cal_pct": round(protein_cal_pct, 1)
                    },
                    "nutrient_ratios": {
                        "fiber_to_carb_ratio": round(fiber_to_carb, 3),
                        "protein_to_carb_ratio": round(meal['protein'] / meal['carbs'], 3) if meal['carbs'] > 0 else 0
                    },
                    "glycemic_impact_factors": {
                        "meal_characteristics": characteristics,
                        "sugar_to_carb_ratio": round(meal['sugar'] / meal['carbs'], 2) if meal['carbs'] > 0 else 0,
                        "estimated_glycemic_impact": glycemic_impact
                    }
                },
                "recommendations": recommendations
            }
        
        except Exception as e:
            api.abort(400, f"Analysis error: {str(e)}")

def estimate_glycemic_impact(meal):
    """Estimate glycemic impact of a meal without using the ML model"""
    # Simple heuristic estimation based on meal composition
    # This is a simplified approach - the ML model provides more accurate predictions
    
    # Base impact from carbs
    impact = meal['carbs'] * 1.0
    
    # Reduce impact based on fiber
    if meal['fiber'] > 0:
        fiber_factor = min(0.5, meal['fiber'] / meal['carbs'] if meal['carbs'] > 0 else 0)
        impact *= (1 - fiber_factor)
    
    # Increase impact based on sugar
    sugar_factor = min(0.5, meal['sugar'] / meal['carbs'] if meal['carbs'] > 0 else 0)
    impact *= (1 + sugar_factor)
    
    # Reduce impact based on fat and protein
    fat_protein = meal['fat'] + meal['protein']
    if fat_protein > 0:
        fp_factor = min(0.4, (fat_protein) / (meal['carbs'] * 2) if meal['carbs'] > 0 else 0)
        impact *= (1 - fp_factor)
    
    # Categorize the impact
    if impact < 20:
        return "Low"
    elif impact < 40:
        return "Medium"
    else:
        return "High"

@image_ns.route('/analyze-food')
class FoodImageAnalysis(Resource):
    @image_ns.doc('analyze_food_image')
    @image_ns.expect(food_image_parser)
    @image_ns.marshal_with(food_image_response_model, code=200)
    @image_ns.response(400, 'Validation Error')
    def post(self):
        """Analyze a food image and predict nutritional content and glucose response"""
        args = food_image_parser.parse_args()
        
        if 'food_image' not in request.files:
            api.abort(400, "No image file provided")
        
        file = request.files['food_image']
        if file.filename == '':
            api.abort(400, "No file selected")
        
        # Get person info if provided
        person = {}
        if args['person_info']:
            try:
                person = json.loads(args['person_info'])
            except:
                api.abort(400, "Invalid person_info format")
        
        # Save uploaded image temporarily
        temp_image_path = "temp_food_image.jpg"
        file.save(temp_image_path)
        
        try:
            # Check if food model is loaded
            if food_model is None:
                api.abort(503, "Food recognition model not loaded.")
            
            # Process image for prediction
            img = load_img(temp_image_path, target_size=(IMG_SIZE, IMG_SIZE))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            
            # Make prediction
            with tf.device('/CPU:0'):  # Using CPU for simplicity
                prediction = food_model(x, training=False)[0]
            
            # Get prediction results
            top_idx = np.argmax(prediction)
            if len(class_names) > top_idx:
                predicted_dish = class_names[top_idx]
                confidence = float(prediction[top_idx])
            else:
                predicted_dish = "Unknown food"
                confidence = 0.0
            
            # Fetch nutrition information
            nutrition = nutrition_client.fetch_nutrition(predicted_dish)
            if not nutrition:
                # Fallback default nutrition values
                nutrition = {
                    "calories": 200,
                    "carbohydrates": 30,
                    "fat": 10,
                    "protein": 15,
                    "fiber": 5,
                    "sugar": 10
                }
            
            # Get glycemic index and calculate glycemic load
            gi = lookup_gi_from_csv(predicted_dish)
            carbs = nutrition.get("carbohydrates", None)
            gl = calculate_gl(carbs, gi)
            
            # Log prediction
            logging.info(
                f"Food prediction: {predicted_dish}, Confidence: {confidence:.2f}, "
                f"GI: {gi}, GL: {gl}"
            )
            
            # Format food item response
            food_item = {
                "name": predicted_dish,
                "confidence": confidence,
                "portion_size": "1 serving",
                "nutrients": {
                    "calories": nutrition.get("calories", 0),
                    "carbs": nutrition.get("carbohydrates", 0),
                    "fat": nutrition.get("fat", 0),
                    "protein": nutrition.get("protein", 0),
                    "fiber": nutrition.get("fiber", 0),
                    "sugar": nutrition.get("sugar", 0)
                }
            }
            
            # Prepare food analysis response
            food_analysis = {
                "identified_foods": [food_item],
                "total_nutrients": {
                    "calories": nutrition.get("calories", 0),
                    "carbs": nutrition.get("carbohydrates", 0),
                    "fat": nutrition.get("fat", 0),
                    "protein": nutrition.get("protein", 0),
                    "fiber": nutrition.get("fiber", 0),
                    "sugar": nutrition.get("sugar", 0)
                }
            }
            
            # If person info is provided, make glucose prediction
            glucose_prediction = None
            if person and all(k in person for k in ['bmi', 'age', 'sex']) and predictor.model is not None:
                glucose_prediction = predictor.predict_glucose_response(
                    person,
                    food_analysis["total_nutrients"]
                )
                
                # Add guidelines and recommendations
                if glucose_prediction:
                    guidelines = []
                    
                    if glucose_prediction['risk_level'] == 'High':
                        guidelines.append("Consider reducing carbohydrates or adding more fiber to this meal.")
                        guidelines.append("Pairing carbohydrates with protein and healthy fats may help reduce glucose impact.")
                    elif glucose_prediction['risk_level'] == 'Medium':
                        guidelines.append("This meal has a moderate impact on blood glucose.")
                        guidelines.append("Consider physical activity after eating to help manage glucose levels.")
                    else:
                        guidelines.append("This meal should have minimal impact on blood glucose levels.")
                    
                    recommendations = []
                    nutrients = food_analysis["total_nutrients"]
                    
                    if nutrients['carbs'] > 50 and nutrients['fiber'] < 5:
                        recommendations.append("This meal is high in carbs and low in fiber. Adding fiber can help reduce glucose impact.")
                    
                    if nutrients['sugar'] > 25:
                        recommendations.append("This meal is high in sugar. Consider reducing added sugars.")
                    
                    glucose_prediction["guidelines"] = guidelines
                    glucose_prediction["recommendations"] = recommendations
            
            return {
                "food_analysis": food_analysis,
                "glucose_prediction": glucose_prediction
            }
        
        except Exception as e:
            api.abort(500, f"Image analysis error: {str(e)}")
        
        finally:
            # Clean up temporary image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

@model_ns.route('/train')
class ModelTraining(Resource):
    @model_ns.doc('train_model')
    @model_ns.expect(upload_parser)
    @model_ns.marshal_with(training_response_model, code=200)
    @model_ns.response(400, 'Validation Error')
    @model_ns.response(500, 'Training Error')
    def post(self):
        """Train or retrain the model with provided data"""
        args = upload_parser.parse_args()
        
        if 'data_file' not in request.files:
            api.abort(400, "No file provided")
        
        file = request.files['data_file']
        if file.filename == '':
            api.abort(400, "No file selected")
        
        # Save uploaded file temporarily
        temp_path = "temp_upload.csv"
        file.save(temp_path)

        try:
            # Train the model
            training_result = predictor.train_model(temp_path)

            if training_result:
                # Save the trained model
                predictor.save_model(MODEL_PATH)
                return {
                    "status": "success",
                    "message": "Model trained successfully",
                    "details": training_result
                }
            else:
                api.abort(500, "Model training failed")

        except Exception as e:
            api.abort(500, f"Training error: {str(e)}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Setup shutdown handler to clean up resources
@app.teardown_appcontext
def shutdown_nutrition_client(exception=None):
    """Close nutrition client on app shutdown"""
    if hasattr(app, 'nutrition_client'):
        app.nutrition_client.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
