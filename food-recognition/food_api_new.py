import os
import numpy as np
import logging
import asyncio
import aiohttp
import aiofiles
import requests
import pandas as pd
import tensorflow as tf
import json
import joblib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body, Form
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
        logging.FileHandler('food_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Load Glycemic Index Dataset ===
try:
    gi_df = pd.read_csv("glycemic_index.csv")
except Exception as e:
    logger.error(f"Failed to load glycemic index data: {e}")
    gi_df = pd.DataFrame(columns=['food', 'glycemic_index'])

def lookup_gi_from_csv(dish_name):
    try:
        row = gi_df[gi_df['food'].str.contains(dish_name, case=False, na=False)]
        if not row.empty:
            return int(row.iloc[0]['glycemic_index'])
        else:
            return "Unknown"
    except Exception as e:
        logger.error(f"GI CSV lookup failed: {e}")
        return "Unknown"

def calculate_gl(carbs, gi):
    if isinstance(gi, int) and carbs:
        return round((float(carbs) * gi) / 100, 2)
    return "Unknown"

# Configuration
class Config:
    MODEL_PATH = Path("model/food_classifier_pro.h5")
    DATASET_PATH = Path("dataset/train")
    TEMP_DIR = Path("temp")
    IMG_SIZE = 224
    CACHE_TTL = 3600  # 1 hour cache for nutrition data
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    
    # API Configuration
    SPOONACULAR_API_KEY = os.getenv('SPOONACULAR_API_KEY')
    SPOONACULAR_BASE_URL = "https://api.spoonacular.com"
    
    # Glucose model path
    GLUCOSE_MODEL_PATH = "glucose_model.joblib"
    
    if not SPOONACULAR_API_KEY:
        logger.warning("Missing Spoonacular API key, will use mock data")

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
    title="Food Classification API",
    description="API for food image classification, nutrition information, and glucose response prediction",
    version="2.1.0"
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
class NutritionInfo(BaseModel):
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: Optional[float] = 0
    sugar: Optional[float] = 0
    serving_weight_grams: float

class Person(BaseModel):
    bmi: float
    age: int
    sex: str  # 'M' or 'F'

class Meal(BaseModel):
    calories: float
    carbs: float
    fat: float
    protein: float
    fiber: float
    sugar: float

class GlucosePredictionRequest(BaseModel):
    person: Person
    meal: Meal

class GlucosePredictionResponse(BaseModel):
    iauc_prediction: float
    risk_level: str
    percentile: int
    explanation: str
    guidelines: List[str]
    recommendations: List[str]

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    nutrition: Optional[Dict[str, float]]
    glycemic_index: str
    glycemic_load: str
    timestamp: datetime
    glucose_prediction: Optional[GlucosePredictionResponse] = None

class FoodItemResponse(BaseModel):
    name: str
    confidence: float
    portion_size: str
    nutrients: Dict[str, float]

class FoodAnalysisResponse(BaseModel):
    identified_foods: List[FoodItemResponse]
    total_nutrients: Dict[str, float]

class FoodImageWithPersonResponse(BaseModel):
    food_analysis: FoodAnalysisResponse
    glucose_prediction: Optional[GlucosePredictionResponse] = None

class ModelTrainingResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

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

# Glucose Response Predictor class
class GlucoseResponsePredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = ['bmi', 'age', 'is_male', 'calories', 'carbs', 'fat', 'protein', 'fiber', 'sugar']
    
    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
            logger.info("Glucose prediction model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading glucose model: {e}")
            return False
    
    def save_model(self, model_path):
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Glucose model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving glucose model: {e}")
            return False
    
    def train_model(self, data_path):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Load and prepare data
            df = pd.read_csv(data_path)
            
            # Process sex to binary feature
            df['is_male'] = df['sex'].apply(lambda x: 1 if x == 'M' else 0)
            
            # Split data
            X = df[self.feature_cols]
            y = df['iauc']  # insulin area under curve
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            logger.info(f"Glucose model trained - Train score: {train_score:.4f}, Test score: {test_score:.4f}")
            
            self.model = model
            
            return {
                "train_score": train_score,
                "test_score": test_score,
                "feature_importance": dict(zip(self.feature_cols, model.feature_importances_))
            }
        
        except Exception as e:
            logger.error(f"Error training glucose model: {e}")
            return None
    
    def predict_glucose_response(self, person, meal):
        try:
            if self.model is None:
                raise ValueError("Glucose model not loaded")
            
            # Create feature array
            features = np.zeros(len(self.feature_cols))
            feature_dict = {
                'bmi': person.get('bmi', 0),
                'age': person.get('age', 0),
                'is_male': 1 if person.get('sex', '').upper() == 'M' else 0,
                'calories': meal.get('calories', 0),
                'carbs': meal.get('carbs', 0),
                'fat': meal.get('fat', 0),
                'protein': meal.get('protein', 0),
                'fiber': meal.get('fiber', 0),
                'sugar': meal.get('sugar', 0)
            }
            
            for i, col in enumerate(self.feature_cols):
                features[i] = feature_dict.get(col, 0)
            
            # Make prediction
            features = features.reshape(1, -1)
            prediction = float(self.model.predict(features)[0])
            
            # Calculate risk level
            if prediction < 30:
                risk_level = "Low"
                percentile = 25
            elif prediction < 60:
                risk_level = "Medium"
                percentile = 50
            else:
                risk_level = "High"
                percentile = 75
            
            # Generate explanation
            explanation = f"Based on your profile (BMI: {person.get('bmi')}, Age: {person.get('age')}) "
            explanation += f"and this meal's composition ({meal.get('carbs')}g carbs, {meal.get('fat')}g fat, "
            explanation += f"{meal.get('protein')}g protein), your predicted insulin response is {risk_level.lower()}."
            
            return {
                'iauc_prediction': round(prediction, 2),
                'risk_level': risk_level,
                'percentile': percentile,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Glucose prediction error: {e}")
            return {
                'iauc_prediction': 0,
                'risk_level': "Unknown",
                'percentile': 0,
                'explanation': f"Error making prediction: {str(e)}"
            }

# Create and initialize glucose predictor
glucose_predictor = GlucoseResponsePredictor()
if os.path.exists(Config.GLUCOSE_MODEL_PATH):
    glucose_predictor.load_model(Config.GLUCOSE_MODEL_PATH)
else:
    logger.warning(f"Glucose model not found at {Config.GLUCOSE_MODEL_PATH}. Some functionality will be limited.")

# Configure GPU
has_gpu, device = GPUConfig.setup_gpu()

# Load ML model and class names
try:
    # Load model with GPU acceleration if available
    with tf.device('/CPU:0' if has_gpu else '/CPU:0'):
        model = load_model(Config.MODEL_PATH)
        # Optimize graph and enable XLA
        model = tf.function(
            model,
            jit_compile=True,  # Enable XLA compilation
            reduce_retracing=True  # Reduce graph retracing
        )
        
    class_names = sorted(os.listdir(Config.DATASET_PATH))
    logger.info(f"Food model loaded successfully on {device}. Found {len(class_names)} classes")
    
except Exception as e:
    logger.error(f"Failed to load food model: {e}")
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
        self.api_key = Config.SPOONACULAR_API_KEY

    def fetch_recipe_id(self, dish_name):
        """Search for the dish and get a recipe ID."""
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
            logger.error(f"Failed to fetch recipe ID: {e}")
            return None

    def fetch_nutrition(self, dish_name, normalize_to_100g=True):
        """Fetch detailed nutrition and optionally normalize per 100g."""
        # If no API key, use mock data
        if not self.api_key:
            mock_data = {
                "pasta": {"calories": 350, "carbohydrates": 65, "fat": 2, "protein": 12, "fiber": 3, "sugar": 3},
                "pizza": {"calories": 300, "carbohydrates": 35, "fat": 12, "protein": 15, "fiber": 2, "sugar": 4},
                "salad": {"calories": 150, "carbohydrates": 10, "fat": 8, "protein": 5, "fiber": 4, "sugar": 5},
                "sandwich": {"calories": 350, "carbohydrates": 40, "fat": 12, "protein": 20, "fiber": 3, "sugar": 4},
                "steak": {"calories": 300, "carbohydrates": 0, "fat": 20, "protein": 30, "fiber": 0, "sugar": 0},
                "sushi": {"calories": 250, "carbohydrates": 40, "fat": 2, "protein": 15, "fiber": 1, "sugar": 5},
                "soup": {"calories": 180, "carbohydrates": 25, "fat": 5, "protein": 10, "fiber": 2, "sugar": 3},
                "chicken": {"calories": 200, "carbohydrates": 0, "fat": 8, "protein": 30, "fiber": 0, "sugar": 0},
                "rice": {"calories": 200, "carbohydrates": 45, "fat": 0.5, "protein": 4, "fiber": 0.5, "sugar": 0.1},
                "burger": {"calories": 400, "carbohydrates": 35, "fat": 20, "protein": 25, "fiber": 2, "sugar": 6},
                "fish": {"calories": 180, "carbohydrates": 0, "fat": 6, "protein": 30, "fiber": 0, "sugar": 0},
                "taco": {"calories": 250, "carbohydrates": 25, "fat": 12, "protein": 15, "fiber": 3, "sugar": 2},
                "apple": {"calories": 95, "carbohydrates": 25, "fat": 0.3, "protein": 0.5, "fiber": 4.4, "sugar": 19}
            }
            
            # Find closest match in mock data
            for key in mock_data.keys():
                if key.lower() in dish_name.lower():
                    mock_data[key]["serving_weight_grams"] = 100
                    return mock_data[key]
            
            # Default values if no match
            return {
                "calories": 200, 
                "carbohydrates": 30, 
                "fat": 10, 
                "protein": 15, 
                "fiber": 5, 
                "sugar": 10,
                "serving_weight_grams": 100
            }
        
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
            serving_weight = float(data.get("serving_weight_grams", 100))  # Default 100g if missing

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
            logger.error(f"Failed to fetch detailed nutrition: {e}")
            return {}

    def close(self):
        """Close the requests session."""
        self.session.close()

# Function to generate guidelines and recommendations based on glucose prediction
def enhance_glucose_prediction(prediction, meal):
    """Add guidelines and recommendations to glucose prediction"""
    if not prediction:
        return prediction
        
    # Add guidelines based on risk level
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
    if meal.get('carbs', 0) > 50 and meal.get('fiber', 0) < 5:
        recommendations.append("This meal is high in carbs and low in fiber. Adding fiber can help reduce glucose impact.")
    
    # Check if high sugar
    if meal.get('sugar', 0) > 25:
        recommendations.append("This meal is high in sugar. Consider reducing added sugars.")
    
    # Check if balanced macros
    total_cals = meal.get('carbs', 0) * 4 + meal.get('fat', 0) * 9 + meal.get('protein', 0) * 4
    carb_pct = (meal.get('carbs', 0) * 4 / total_cals) * 100 if total_cals > 0 else 0
    
    if carb_pct > 60:
        recommendations.append("This meal is very high in carbohydrates. Consider adding more protein or healthy fats.")
    
    # Enhance the response
    enhanced_response = {
        **prediction,
        "guidelines": guidelines,
        "recommendations": recommendations
    }
    
    return enhanced_response

# Initialize nutrition client
nutrition_client = NutritionClient()

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

@app.post("/predict/", response_model=PredictionResponse)
async def predict_food(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, person_info: str = Form(None)):
    """Predict food class from image and return nutrition information with optional glucose prediction."""
    try:
        # Save and process uploaded file
        temp_path = await ImageProcessor.save_upload_file(file)
        background_tasks.add_task(cleanup_temp_file, temp_path)

        # Process image
        x = ImageProcessor.process_image(temp_path)
        
        # Make prediction with GPU acceleration
        with tf.device('/CPU:0' if has_gpu else '/CPU:0'):
            # Use model in inference mode with XLA optimization
            prediction = model(x, training=False)[0]

        # Get prediction results
        top_idx = np.argmax(prediction)
        predicted_dish = class_names[top_idx]
        confidence = float(prediction[top_idx])

        # Fetch nutrition info
        nutrition = nutrition_client.fetch_nutrition(predicted_dish)
        
        gi = lookup_gi_from_csv(predicted_dish)
        carbs = nutrition.get("carbohydrates", None)
        gl = calculate_gl(carbs, gi)
        
        # Process glucose prediction if person info is provided
        glucose_prediction = None
        if person_info and glucose_predictor.model is not None:
            try:
                person = json.loads(person_info)
                if all(k in person for k in ['bmi', 'age', 'sex']):
                    # Make glucose prediction
                    glucose_result = glucose_predictor.predict_glucose_response(person, nutrition)
                    # Enhance with guidelines and recommendations
                    glucose_prediction = enhance_glucose_prediction(glucose_result, nutrition)
            except json.JSONDecodeError:
                logger.error("Invalid person_info format")
        
        # Log prediction stats
        logger.info(
            f"Prediction completed - Class: {predicted_dish}, "
            f"Confidence: {confidence:.2f}, "
            f"Device: {device}",
        )

        return PredictionResponse(
            predicted_class=predicted_dish,
            confidence=confidence,
            nutrition=nutrition,
            glycemic_index=str(gi),
            glycemic_load=str(gl),
            timestamp=datetime.now(),
            glucose_prediction=glucose_prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze-food/", response_model=FoodImageWithPersonResponse)
async def analyze_food_image(file: UploadFile = File(...), person_info: str = Form(None), background_tasks: BackgroundTasks = None):
    """Analyze a food image and predict nutritional content with optional glucose response."""
    try:
        # Save and process uploaded file
        temp_path = await ImageProcessor.save_upload_file(file)
        background_tasks.add_task(cleanup_temp_file, temp_path)

        # Process image
        x = ImageProcessor.process_image(temp_path)
        
        # Make prediction with GPU acceleration
        with tf.device('/CPU:0' if has_gpu else '/CPU:0'):
            # Use model in inference mode with XLA optimization
            prediction = model(x, training=False)[0]

        # Get prediction results
        top_idx = np.argmax(prediction)
        predicted_dish = class_names[top_idx]
        confidence = float(prediction[top_idx])

        # Fetch nutrition info
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
        
        # Process glucose prediction if person info is provided
        glucose_prediction = None
        if person_info and glucose_predictor.model is not None:
            try:
                person = json.loads(person_info)
                if all(k in person for k in ['bmi', 'age', 'sex']):
                    # Make glucose prediction
                    glucose_result = glucose_predictor.predict_glucose_response(person, food_analysis["total_nutrients"])
                    # Enhance with guidelines and recommendations
                    glucose_prediction = enhance_glucose_prediction(glucose_result, food_analysis["total_nutrients"])
            except json.JSONDecodeError:
                logger.error("Invalid person_info format")
        
        # Log prediction
        logger.info(
            f"Food analysis completed - Food: {predicted_dish}, "
            f"Confidence: {confidence:.2f}, "
            f"GI: {gi}, GL: {gl}"
        )

        return {
            "food_analysis": food_analysis,
            "glucose_prediction": glucose_prediction
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Food analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict-glucose/", response_model=GlucosePredictionResponse)
async def predict_glucose_response(data: GlucosePredictionRequest):
    """Predict glucose response based on person and meal details."""
    if glucose_predictor.model is None:
        raise HTTPException(status_code=503, detail="Glucose prediction model not loaded")
    
    try:
        # Convert pydantic models to dictionaries
        person = data.person.dict()
        meal = data.meal.dict()
        
        # Make prediction
        prediction = glucose_predictor.predict_glucose_response(person, meal)
        
        # Enhance with guidelines and recommendations
        enhanced_prediction = enhance_glucose_prediction(prediction, meal)
        
        return enhanced_prediction
    except Exception as e:
        logger.error(f"Glucose prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting glucose response: {str(e)}")

@app.post("/train-glucose-model/", response_model=ModelTrainingResponse)
async def train_glucose_model(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Train or retrain the glucose prediction model with provided data."""
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_training_data.csv")
        
        # Read and save file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Train the model
        training_result = glucose_predictor.train_model(str(temp_path))
        
        if not training_result:
            raise HTTPException(status_code=500, detail="Model training failed")
        
        # Save the model
        glucose_predictor.save_model(Config.GLUCOSE_MODEL_PATH)
        
        # Schedule cleanup
        background_tasks.add_task(lambda: temp_path.unlink(missing_ok=True))
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "details": training_result
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

def estimate_glycemic_impact(meal):
    """Estimate glycemic impact of a meal without using the ML model"""
    # Simple heuristic estimation based on meal composition
    # This is a simplified approach - the ML model provides more accurate predictions
    
    # Base impact from carbs
    impact = meal.get('carbs', 0) * 1.0
    
    # Reduce impact based on fiber
    if meal.get('fiber', 0) > 0:
        fiber_factor = min(0.5, meal.get('fiber', 0) / meal.get('carbs', 1) if meal.get('carbs', 0) > 0 else 0)
        impact *= (1 - fiber_factor)
    
    # Increase impact based on sugar
    sugar_factor = min(0.5, meal.get('sugar', 0) / meal.get('carbs', 1) if meal.get('carbs', 0) > 0 else 0)
    impact *= (1 + sugar_factor)
    
    # Reduce impact based on fat and protein
    fat_protein = meal.get('fat', 0) + meal.get('protein', 0)
    if fat_protein > 0:
        fp_factor = min(0.4, (fat_protein) / (meal.get('carbs', 1) * 2) if meal.get('carbs', 0) > 0 else 0)
        impact *= (1 - fp_factor)
    
    # Categorize the impact
    if impact < 20:
        return "Low"
    elif impact < 40:
        return "Medium"
    else:
        return "High"

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "food_model_loaded": model is not None,
        "glucose_model_loaded": glucose_predictor.model is not None
    }

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

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
