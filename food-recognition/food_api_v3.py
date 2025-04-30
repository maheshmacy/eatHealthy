import os
import numpy as np
import logging
import asyncio
import aiohttp
import aiofiles
import requests
import pandas as pd
import tensorflow as tf
import joblib
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.mixed_precision import set_global_policy
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from cachetools import TTLCache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_analyzer_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Load Glycemic Index Dataset ===
gi_df = pd.read_csv("glycemic_index.csv")

def lookup_gi_from_csv(dish_name):
    try:
        row = gi_df[gi_df['food'].str.contains(dish_name, case=False, na=False)]
        if not row.empty:
            return int(row.iloc[0]['glycemic_index'])
        else:
            return "Unknown"
    except Exception as e:
        print(f"⚠️ GI CSV lookup failed: {e}")
        return "Unknown"

def calculate_gl(carbs, gi):
    if isinstance(gi, int) and carbs:
        return round((float(carbs) * gi) / 100, 2)
    return "Unknown"

# Configuration
class Config:
    MODEL_PATH = Path("model/food_classifier_pro.h5")
    GLUCOSE_MODEL_PATH = Path("model/time_series_glucose_model.pkl")
    DATASET_PATH = Path("dataset/train")
    TEMP_DIR = Path("temp")
    IMG_SIZE = 224
    CACHE_TTL = 3600  # 1 hour cache for nutrition data
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    
    # API Configuration
    SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')
    SPOONACULAR_BASE_URL = "https://api.spoonacular.com"
    
    if not SPOONACULAR_API_KEY:
        logger.error("Missing Spoonacular API key")
        raise ValueError("SPOONACULAR_API_KEY must be set in .env file")

# Security configuration
class SecurityConfig:
    # Define allowed origins - update these with your actual frontend URLs
    ALLOWED_ORIGINS = [
        "http://localhost:3000",        # Local development
        "http://localhost:8000",        # FastAPI docs
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # Define allowed methods
    ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
    
    # Define allowed headers
    ALLOWED_HEADERS = [
        "Content-Type",
        "Authorization",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods",
        "Access-Control-Allow-Headers"
    ]
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }

# Initialize FastAPI app with CORS and custom configuration
app = FastAPI(
    title="Food Analysis & Glucose Prediction API",
    description="API for food image classification, nutrition analysis, and personalized glucose response prediction",
    version="3.0.0"
)

# Add CORS middleware with specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=SecurityConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=SecurityConfig.ALLOWED_METHODS,
    allow_headers=SecurityConfig.ALLOWED_HEADERS,
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Initialize cache
nutrition_cache = TTLCache(maxsize=100, ttl=Config.CACHE_TTL)

# Response Models
class NutrientInfo(BaseModel):
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: float
    sugar: float
    serving_weight_grams: float

class GlucoseResponse(BaseModel):
    risk_level: str
    max_glucose: float
    time_to_peak: int
    iauc_prediction: Optional[float] = None
    explanation: str
    guidelines: List[str]
    recommendations: List[str]

class FoodAnalysis(BaseModel):
    predicted_food: str
    confidence: float
    nutrients: NutrientInfo
    glycemic_index: str
    glycemic_load: str

class FoodRecommendations(BaseModel):
    low_risk_alternatives: List[Dict[str, Any]]
    moderate_risk_alternatives: List[Dict[str, Any]]
    high_risk_alternatives: List[Dict[str, Any]]

class AnalyzeResponse(BaseModel):
    food_analysis: FoodAnalysis
    glucose_prediction: GlucoseResponse
    recommendations: FoodRecommendations
    timestamp: datetime

class PersonInfo(BaseModel):
    age: int
    sex: str
    bmi: float
    has_diabetes: bool = False
    takes_medication: bool = False
    activity_level: int = 5  # Scale 1-10

# GPU Configuration
class GPUConfig:
    @staticmethod
    def setup_gpu() -> Tuple[bool, str]:
        """Configure GPU settings for optimal performance."""
        try:
            # List physical devices
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if not physical_devices:
                logger.warning("No GPU devices found. Using CPU for inference.")
                return False, "CPU"
            
            # Configure primary GPU
            device = physical_devices[0]  # Use first GPU
            try:
                # Enable memory growth to prevent tensorflow from allocating all memory
                tf.config.experimental.set_memory_growth(device, True)
                if len(physical_devices) > 1:
                    tf.config.experimental.set_memory_growth(physical_devices[1], True)
                
                # Set mixed precision policy for better performance
                set_global_policy('mixed_float16')
                
                # Get device name
                device_name = device.name.split('/')[-1]
                
                logger.info(f"GPU configuration successful. Using device: {device_name}")
                return True, device_name
                
            except RuntimeError as e:
                logger.error(f"Failed to configure GPU {device}: {e}")
                return False, "CPU"
            
        except Exception as e:
            logger.error(f"Error configuring GPU: {e}")
            return False, "CPU"

# Configure GPU
has_gpu, device = GPUConfig.setup_gpu()

# Load ML models and class names
try:
    # Load food classification model with GPU acceleration if available
    with tf.device('/GPU:0' if has_gpu else '/CPU:0'):
        food_model = load_model(Config.MODEL_PATH)
        # Optimize graph and enable XLA
        food_model = tf.function(
            food_model,
            jit_compile=True,  # Enable XLA compilation
            reduce_retracing=True  # Reduce graph retracing
        )
    
    # Load glucose prediction model    
    glucose_model_results = joblib.load(Config.GLUCOSE_MODEL_PATH)
    glucose_model = glucose_model_results['model']
    glucose_features = glucose_model_results['features']
    
    # Load food class names
    class_names = sorted(os.listdir(Config.DATASET_PATH))
    logger.info(f"Models loaded successfully on {device}. Found {len(class_names)} food classes")
    
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Utility functions
class ImageProcessor:
    @staticmethod
    async def save_upload_file(upload_file: UploadFile) -> Path:
        """Save uploaded file with proper validation."""
        # Validate file size and extension
        content = await upload_file.read(Config.MAX_FILE_SIZE + 1)
        if len(content) > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
            
        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported file type")

        # Create temp directory if it doesn't exist
        Config.TEMP_DIR.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = Config.TEMP_DIR / f"upload_{timestamp}{file_ext}"
        
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
            
        return temp_path

    @staticmethod
    def process_image(image_path: Path) -> np.ndarray:
        """Process image for model prediction."""
        try:
            img = load_img(image_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
            x = img_to_array(img) / 255.0
            return np.expand_dims(x, axis=0)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")

class NutritionClient:
    def __init__(self):
        self.session = requests.Session()

    def fetch_recipe_id(self, dish_name):
        """Search for the dish and get a recipe ID."""
        try:
            search_url = f"{Config.SPOONACULAR_BASE_URL}/recipes/complexSearch"
            params = {
                "apiKey": Config.SPOONACULAR_API_KEY,
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
            logger.error(f"Failed to fetch recipe ID: {e}")
            return None

    def fetch_nutrition(self, dish_name, normalize_to_100g=True):
        """Fetch detailed nutrition and optionally normalize per 100g."""
        try:
            # Check cache first
            cache_key = f"{dish_name}_{normalize_to_100g}"
            if cache_key in nutrition_cache:
                return nutrition_cache[cache_key]
                
            recipe_id = self.fetch_recipe_id(dish_name)
            if recipe_id is None:
                return {}

            nutrition_url = f"{Config.SPOONACULAR_BASE_URL}/recipes/{recipe_id}/nutritionWidget.json"
            params = {"apiKey": Config.SPOONACULAR_API_KEY}
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
                "serving_weight_grams": serving_weight if not normalize_to_100g else 100
            }
            
            # Store in cache
            nutrition_cache[cache_key] = nutrition
            return nutrition

        except Exception as e:
            logger.error(f"Failed to fetch detailed nutrition: {e}")
            return {}

    def find_alternatives(self, food_category, max_gi=None, min_gi=None, max_items=3):
        """Find alternative foods with lower or higher glycemic index."""
        try:
            search_url = f"{Config.SPOONACULAR_BASE_URL}/recipes/complexSearch"
            params = {
                "apiKey": Config.SPOONACULAR_API_KEY,
                "query": food_category,
                "instructionsRequired": False,
                "sort": "popularity",
                "sortDirection": "desc",
                "number": 10
            }
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            alternatives = []
            for item in data.get("results", [])[:6]:  # Process top 6 results
                food_name = item["title"]
                
                # Get glycemic index
                gi = lookup_gi_from_csv(food_name)
                if isinstance(gi, str):
                    continue  # Skip items with unknown GI
                    
                # Filter by GI criteria if provided
                if max_gi is not None and gi > max_gi:
                    continue
                if min_gi is not None and gi < min_gi:
                    continue
                    
                # Get nutrition info
                nutrition = self.fetch_nutrition(food_name)
                if not nutrition:
                    continue
                    
                # Calculate glycemic load
                gl = calculate_gl(nutrition.get("carbohydrates"), gi)
                
                alternatives.append({
                    "name": food_name,
                    "glycemic_index": gi,
                    "glycemic_load": gl,
                    "nutrients": nutrition
                })
                
                if len(alternatives) >= max_items:
                    break
                    
            return alternatives
            
        except Exception as e:
            logger.error(f"Failed to find alternatives: {e}")
            return []

    def close(self):
        """Close the requests session."""
        self.session.close()

# Initialize nutrition client
nutrition_client = NutritionClient()

class GlucosePredictor:
    def __init__(self, model_results):
        self.model = model_results['model']
        self.features = model_results['features']
    
    def predict_glucose_response(self, meal_data, person_info=None, time_points=24):
        """
        Predict glucose response over time for a meal with personal characteristics
        """
        # Create a standardized input for the model
        processed_input = self._prepare_model_input(meal_data, person_info)
        
        # Make predictions for each time point
        time_series = []
        glucose_predictions = []
        
        # If we're using a time-based model, predict for each time point
        if 'Minutes_After_Meal' in self.features:
            for i in range(time_points):
                minutes = i * 5  # 5-minute intervals
                time_series.append(minutes)
                
                # Update time point in the input data
                time_data = processed_input.copy()
                time_data['Minutes_After_Meal'] = minutes
                
                # Predict for this time point
                time_prediction = self._predict_single_point(time_data)
                glucose_predictions.append(time_prediction)
        else:
            # For non-time-based model, make a single prediction
            glucose = self._predict_single_point(processed_input)
            glucose_predictions.append(glucose)
            time_series.append(0)
            
        # Calculate key metrics
        max_glucose = max(glucose_predictions)
        max_glucose_idx = glucose_predictions.index(max_glucose)
        max_glucose_time = time_series[max_glucose_idx]
        
        # Calculate incremental area under curve (iAUC)
        baseline = glucose_predictions[0]  # Use first glucose value as baseline
        above_baseline = [max(g - baseline, 0) for g in glucose_predictions]
        iauc = sum(above_baseline) * 5 / 60  # Convert 5-min intervals to hours
        
        # Determine risk level
        risk_level = self._determine_risk_level(max_glucose, iauc)
        
        # Generate explanations and recommendations
        explanation = self._generate_explanation(processed_input, max_glucose, max_glucose_time)
        guidelines = self._generate_guidelines(risk_level)
        recommendations = self._generate_recommendations(processed_input, risk_level)
        
        return {
            'time_series': time_series,
            'glucose_predictions': glucose_predictions,
            'max_glucose': max_glucose,
            'time_to_peak': max_glucose_time,
            'iauc_prediction': round(iauc, 2),
            'risk_level': risk_level,
            'explanation': explanation,
            'guidelines': guidelines,
            'recommendations': recommendations
        }
    
    def _prepare_model_input(self, meal_data, person_info):
        """Prepare the input data for the model."""
        # Create a standardized input dictionary
        model_input = {}
        
        # Add meal data
        if 'carbohydrates' in meal_data:
            model_input['Carbs'] = meal_data.get('carbohydrates', 0)
        else:
            model_input['Carbs'] = meal_data.get('Carbs', 0)
            
        if 'fat' in meal_data:
            model_input['Fat'] = meal_data.get('fat', 0)
        else:
            model_input['Fat'] = meal_data.get('Fat', 0)
            
        if 'protein' in meal_data:
            model_input['Protein'] = meal_data.get('protein', 0)
        else:
            model_input['Protein'] = meal_data.get('Protein', 0)
            
        if 'fiber' in meal_data:
            model_input['Fiber'] = meal_data.get('fiber', 0)
        else:
            model_input['Fiber'] = meal_data.get('Fiber', 0)
            
        if 'sugar' in meal_data:
            model_input['Sugar'] = meal_data.get('sugar', 0)
        else:
            model_input['Sugar'] = meal_data.get('Sugar', 0)
        
        # Add glycemic index and calculate glycemic load
        gi = meal_data.get('GI', meal_data.get('glycemic_index', 50))  # Default to medium GI if not provided
        if isinstance(gi, str):
            gi = 50  # Default if GI is "Unknown"
        model_input['GI'] = gi
        model_input['Glycemic_Load'] = calculate_gl(model_input['Carbs'], gi)
        
        # Add person information if provided
        if person_info:
            model_input['Age'] = person_info.get('age', 35)
            
            # Convert sex to numeric format
            sex_str = person_info.get('sex', 'M')
            model_input['Sex'] = 1 if sex_str.upper() in ['M', 'MALE'] else 0
            
            model_input['BMI'] = person_info.get('bmi', 24)
            
            # Determine diabetes category
            if person_info.get('has_diabetes', False):
                model_input['Category'] = 2  # Diabetic
            else:
                model_input['Category'] = 0  # Normal
            
            model_input['Takes_Medication'] = 1 if person_info.get('takes_medication', False) else 0
            model_input['Family_History'] = person_info.get('family_history', 0)
        else:
            # Default values if no person info provided
            model_input['Age'] = 35
            model_input['Sex'] = 1  # Male
            model_input['BMI'] = 24
            model_input['Category'] = 0  # Normal
            model_input['Takes_Medication'] = 0
            model_input['Family_History'] = 0
        
        # Add activity information
        model_input['Activity_Level'] = person_info.get('activity_level', 5) if person_info else 5
        model_input['Activity_Timing'] = 1  # Default to 'after' (assumes encoding where before=0, after=1, none=2)
        model_input['Stress_Level'] = 5
        model_input['Sleep_Quality'] = 7
        model_input['Time_Since_Last_Meal'] = 4
        
        # Minutes after meal will be added in the prediction loop
        model_input['Minutes_After_Meal'] = 0
        
        return model_input
    
    def _predict_single_point(self, data_point):
        """Make a single prediction using the model."""
        # Create a DataFrame with all required features
        df = pd.DataFrame([data_point])
        
        # Ensure all features expected by the model are present
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select only the features used by the model
        X = df[self.features]
        
        # Make the prediction
        glucose = self.model.predict(X)[0]
        return glucose
    
    def _determine_risk_level(self, max_glucose, iauc):
        """Determine the risk level based on maximum glucose and iAUC."""
        if max_glucose > 180:
            return "High"
        elif max_glucose > 140:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_explanation(self, data, max_glucose, peak_time):
        """Generate an explanation of the prediction."""
        carbs = data.get('Carbs', 0)
        gi = data.get('GI', 0)
        fiber = data.get('Fiber', 0)
        fat = data.get('Fat', 0)
        protein = data.get('Protein', 0)
        
        explanation = f"Based on the nutrient composition (carbs: {carbs}g, fat: {fat}g, protein: {protein}g, fiber: {fiber}g), "
        explanation += f"we predict a peak glucose level of {max_glucose:.1f} mg/dL occurring {peak_time} minutes after consumption. "
        
        if carbs > 60 and fiber < 5:
            explanation += "The high carbohydrate content combined with low fiber contributes to a rapid glucose spike. "
        elif carbs > 30 and gi > 65:
            explanation += "The moderate carbohydrate content with high glycemic index leads to faster absorption. "
        elif fat > 20 and protein > 20:
            explanation += "The significant fat and protein content may help slow down glucose absorption. "
            
        return explanation
    
    def _generate_guidelines(self, risk_level):
        """Generate general guidelines based on risk level."""
        guidelines = []
        
        if risk_level == "High":
            guidelines = [
                "Consider reducing portion size or pairing with protein and healthy fats.",
                "A short walk after eating may help lower glucose response.",
                "Monitor your blood glucose if you have diabetes or pre-diabetes.",
                "Stay hydrated to help your body process glucose more efficiently."
            ]
        elif risk_level == "Moderate":
            guidelines = [
                "This meal has a moderate impact on blood glucose.",
                "Adding vegetables or protein can help reduce the glucose impact.",
                "Consider eating this food earlier in the day when insulin sensitivity is higher.",
                "Pair with a side of non-starchy vegetables for better balance."
            ]
        else:
            guidelines = [
                "This meal should have minimal impact on blood glucose levels.",
                "The balanced nutrient profile promotes stable glucose levels.",
                "This is a good option for those monitoring their glucose levels.",
                "The fiber content helps moderate the glucose response."
            ]
            
        return guidelines
    
    def _generate_recommendations(self, data, risk_level):
        """Generate personalized recommendations based on meal data and risk level."""
        recommendations = []
        
        carbs = data.get('Carbs', 0)
        fiber = data.get('Fiber', 0)
        protein = data.get('Protein', 0)
        fat = data.get('Fat', 0)
        
        # Check if high carbs with low fiber
        if carbs > 50 and fiber < 5:
            recommendations.append("This meal is high in carbs and low in fiber. Adding fiber can help reduce glucose impact.")
        
        # Check if high sugar
        if data.get('Sugar', 0) > 25:
            recommendations.append("This meal is high in sugar. Consider reducing added sugars.")
        
        # Check if balanced macros
        total_cals = carbs * 4 + fat * 9 + protein * 4
        carb_pct = (carbs * 4 / total_cals) * 100 if total_cals > 0 else 0
        
        if carb_pct > 60:
            recommendations.append("This meal is very high in carbohydrates. Consider adding more protein or healthy fats.")
            
        # Fiber recommendations
        if carbs > 30 and fiber / carbs < 0.1:
            recommendations.append("Try to increase the fiber-to-carb ratio by adding vegetables, legumes, or whole grains.")
        
        # Activity recommendations
        if risk_level in ["High", "Moderate"]:
            recommendations.append("Physical activity after this meal can help reduce the glucose response significantly.")
        
        # Timing recommendations
        if risk_level == "High":
            recommendations.append("Consider consuming this food earlier in the day when insulin sensitivity is higher.")
        
        return recommendations

# Initialize glucose predictor
glucose_predictor = GlucosePredictor(glucose_model_results)

# Cleanup task
async def cleanup_temp_file(file_path: Path):
    """Remove temporary file after processing."""
    try:
        await asyncio.sleep(60)  # Wait for 1 minute
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

# API Endpoints
# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    for header, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

@app.post("/api/analyze-food/", response_model=AnalyzeResponse)
async def analyze_food(
    file: UploadFile = File(...), 
    person_data: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze a food image, predict nutrients, glucose response, and provide alternatives.
    
    Parameters:
    - file: Food image file
    - person_data: Optional JSON string with person information (age, sex, bmi, has_diabetes, etc.)
    
    Returns:
    - Food analysis with nutrition info
    - Glucose prediction with risk level
    - Alternative food recommendations
    """
    try:
        # Parse person information if provided
        person_info = None
        if person_data:
            try:
                person_info = json.loads(person_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid person data format")
        
        # Save and process uploaded file
        temp_path = await ImageProcessor.save_upload_file(file)
        background_tasks.add_task(cleanup_temp_file, temp_path)

        # Process image
        x = ImageProcessor.process_image(temp_path)
        
        # Make food classification prediction with GPU acceleration
        with tf.device('/GPU:0' if has_gpu else '/CPU:0'):
            prediction = food_model(x, training=False)[0]

        # Get food classification results
        top_idx = np.argmax(prediction)
        predicted_food = class_names[top_idx]
        confidence = float(prediction[top_idx])

        # Fetch nutrition info
        nutrition = nutrition_client.fetch_nutrition(predicted_food)
        if not nutrition:
            # Fallback to default nutrition values if API fails
            nutrition = {
                "calories": 250,
                "carbohydrates": 30,
                "fat": 10,
                "protein": 15,
                "fiber": 3,
                "sugar": 5,
                "serving_weight_grams": 100
            }
        
        # Get glycemic index and calculate glycemic load
        gi = lookup_gi_from_csv(predicted_food)
        carbs = nutrition.get("carbohydrates", None)
        gl = calculate_gl(carbs, gi)
        
        # Predict glucose response
        glucose_response = glucose_predictor.predict_glucose_response(nutrition, person_info)
        risk_level = glucose_response['risk_level']
        
        # Find food alternatives based on risk level
        food_category = predicted_food.split('_')[0] if '_' in predicted_food else predicted_food
        
        # Get lower GI alternatives for high/moderate risk foods
        low_risk_alternatives = []
        if risk_level in ["High", "Moderate"]:
            low_risk_alternatives = nutrition_client.find_alternatives(food_category, max_gi=55)
        
        # Get moderate GI alternatives
        moderate_risk_alternatives = nutrition_client.find_alternatives(food_category, min_gi=56, max_gi=69)
        
        # Get higher GI alternatives if the original food was low GI
        high_risk_alternatives = []
        if risk_level == "Low":
            high_risk_alternatives = nutrition_client.find_alternatives(food_category, min_gi=70)
        else:
            # Still provide some higher GI options for comparison
            high_risk_alternatives = nutrition_client.find_alternatives(food_category, min_gi=70)
        
        # Log prediction stats
        logger.info(
            f"Analysis completed - Food: {predicted_food}, "
            f"Confidence: {confidence:.2f}, "
            f"GI: {gi}, "
            f"Glucose Risk: {risk_level}"
        )

        # Construct response
        response = AnalyzeResponse(
            food_analysis=FoodAnalysis(
                predicted_food=predicted_food,
                confidence=confidence,
                nutrients=NutrientInfo(
                    calories=nutrition.get("calories", 0),
                    protein=nutrition.get("protein", 0),
                    fat=nutrition.get("fat", 0),
                    carbohydrates=nutrition.get("carbohydrates", 0),
                    fiber=nutrition.get("fiber", 0),
                    sugar=nutrition.get("sugar", 0),
                    serving_weight_grams=nutrition.get("serving_weight_grams", 100)
                ),
                glycemic_index=str(gi),
                glycemic_load=str(gl)
            ),
            glucose_prediction=GlucoseResponse(
                risk_level=glucose_response['risk_level'],
                max_glucose=glucose_response['max_glucose'],
                time_to_peak=glucose_response['time_to_peak'],
                iauc_prediction=glucose_response['iauc_prediction'],
                explanation=glucose_response['explanation'],
                guidelines=glucose_response['guidelines'],
                recommendations=glucose_response['recommendations']
            ),
            recommendations=FoodRecommendations(
                low_risk_alternatives=low_risk_alternatives,
                moderate_risk_alternatives=moderate_risk_alternatives,
                high_risk_alternatives=high_risk_alternatives
            ),
            timestamp=datetime.now()
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "models_loaded": {
            "food_classification": food_model is not None,
            "glucose_prediction": glucose_model is not None
        }
    }

@app.get("/api/food-categories")
async def get_food_categories():
    """Get the list of supported food categories."""
    return {"categories": class_names}

@app.get("/api/glycemic-index/{food_name}")
async def get_glycemic_index(food_name: str):
    """Get the glycemic index for a specific food."""
    gi = lookup_gi_from_csv(food_name)
    return {"food": food_name, "glycemic_index": gi}

# Shutdown event
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    nutrition_client.close()
    # Cleanup temp directory
    for file in Config.TEMP_DIR.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
