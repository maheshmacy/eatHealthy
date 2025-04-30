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
from typing import Dict, Any, List, Optional, Tuple
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
        logging.FileHandler('food_health_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Load Glycemic Index Dataset ===
gi_df = pd.read_csv("food-recognition/glycemic_index.csv")

# Configuration
class Config:
    MODEL_PATH = Path("food-recognition/model/food_classifier_pro.h5")
    GLUCOSE_MODEL_PATH = Path("model/random_forest_model.pkl")
    DATASET_PATH = Path("food-recognition/dataset/train")
    TEMP_DIR = Path("temp")
    IMG_SIZE = 224
    CACHE_TTL = 3600  # 1 hour cache for nutrition data
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    
    # API Configuration
    SPOONACULAR_API_KEY = "d92812bb778f4128acf14b413678d64e"
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
    title="Food Health Analysis API",
    description="API for food image classification, nutrition information, and personalized glycemic impact analysis",
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
class UserProfile(BaseModel):
    age: int
    sex: str  # "Male" or "Female"
    bmi: float
    health_category: str  # "Normal", "Pre-diabetic", or "Diabetic"

class NutritionInfo(BaseModel):
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    serving_weight_grams: float

class GlucoseImpact(BaseModel):
    predicted_glucose: float
    glucose_category: str  # "Low", "Normal", "Elevated", "High"
    personalized_risk: str  # "Low", "Moderate", "High" 
    recommendation: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    nutrition: Optional[Dict[str, float]]
    glycemic_index: str
    glycemic_load: str
    glucose_impact: Optional[GlucoseImpact]
    timestamp: datetime

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
    with tf.device('/CPU:0' if has_gpu else '/CPU:0'):
        food_model = load_model(Config.MODEL_PATH)
        # Optimize graph and enable XLA
        food_model = tf.function(
            food_model,
            jit_compile=True,  # Enable XLA compilation
            reduce_retracing=True  # Reduce graph retracing
        )
    
    # Load glucose prediction model
    try:
        glucose_model = joblib.load(Config.GLUCOSE_MODEL_PATH)
        logger.info("Glucose prediction model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load glucose prediction model: {e}")
        glucose_model = None
        
    class_names = sorted(os.listdir(Config.DATASET_PATH))
    logger.info(f"Food model loaded successfully on {device}. Found {len(class_names)} classes")
    
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

# Utility functions
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

def categorize_glucose(glucose_value):
    if glucose_value < 70:
        return "Low"
    elif 70 <= glucose_value <= 139:
        return "Normal"
    elif 140 <= glucose_value <= 179:
        return "Elevated"
    else:
        return "High"

def calculate_personalized_risk(glucose_category, health_category):
    risk_matrix = {
        "Normal": {
            "Low": "Low",
            "Normal": "Low",
            "Elevated": "Moderate",
            "High": "Moderate"
        },
        "Pre-diabetic": {
            "Low": "Low",
            "Normal": "Moderate",
            "Elevated": "High",
            "High": "High"
        },
        "Diabetic": {
            "Low": "Moderate", 
            "Normal": "Moderate",
            "Elevated": "High",
            "High": "High"
        }
    }
    return risk_matrix.get(health_category, {}).get(glucose_category, "Unknown")

def get_recommendation(personalized_risk, food_name, gi_value):
    if personalized_risk == "Low":
        return f"This food is suitable for your health profile. Enjoy your {food_name}!"
    
    elif personalized_risk == "Moderate":
        if isinstance(gi_value, int) and gi_value > 55:
            return f"Consider having smaller portions of {food_name} or pairing it with protein or healthy fats to slow glucose absorption."
        else:
            return f"This food should be fine in moderate amounts, but monitor your response to {food_name}."
    
    elif personalized_risk == "High":
        if isinstance(gi_value, int) and gi_value > 70:
            return f"This high-glycemic food ({food_name}) may cause significant glucose spikes given your health profile. Consider alternatives with lower glycemic index."
        else:
            return f"For your health profile, limit consumption of {food_name} or pair with fiber, protein, and healthy fats to minimize glucose impact."
    
    return "Unable to provide a personalized recommendation with the available data."

def predict_glucose_impact(user_profile, food_data):
    if glucose_model is None:
        return None
    
    try:
        # Encode sex
        sex_encoded = 1 if user_profile.sex == "Male" else 0
        
        # Encode health category
        if user_profile.health_category == "Normal":
            category_encoded = 0
        elif user_profile.health_category == "Pre-diabetic":
            category_encoded = 1
        else:  # Diabetic
            category_encoded = 2
        
        # Prepare data for prediction
        gi = food_data.get("glycemic_index")
        if gi == "Unknown":
            gi = 50  # Use average GI if unknown
        else:
            gi = int(gi)
            
        carbs = food_data.get("carbohydrates", 0)
        fat = food_data.get("fat", 0)
        fiber = food_data.get("fiber", 3)  # Default fiber value if not available
        glycemic_load = gi * carbs / 100
        
        # Create input for model
        input_data = pd.DataFrame([{
            'Age': user_profile.age,
            'Sex': sex_encoded,
            'BMI': user_profile.bmi,
            'Fat': fat,
            'Fiber': fiber,
            'Glycemic_Load': glycemic_load,
            'Category': category_encoded
        }])
        
        # Predict glucose level
        predicted_glucose = glucose_model.predict(input_data)[0]
        glucose_category = categorize_glucose(predicted_glucose)
        personalized_risk = calculate_personalized_risk(glucose_category, user_profile.health_category)
        recommendation = get_recommendation(personalized_risk, food_data.get("food_name", "this food"), gi)
        
        return GlucoseImpact(
            predicted_glucose=round(float(predicted_glucose), 2),
            glucose_category=glucose_category,
            personalized_risk=personalized_risk,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Failed to predict glucose impact: {e}")
        return None

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
        # Check cache first
        cache_key = f"{dish_name}_{normalize_to_100g}"
        if cache_key in nutrition_cache:
            return nutrition_cache[cache_key]
            
        try:
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
            serving_weight = float(data.get("serving_weight_grams", 100))  # Default 100g if missing
            
            # Try to extract fiber information
            fiber = 0
            for nutrient in data.get("nutrients", []):
                if nutrient.get("name", "").lower() == "fiber":
                    fiber = float(nutrient.get("amount", 0))
                    break

            # Normalize to 100g if needed
            if normalize_to_100g and serving_weight != 100:
                factor = 100 / serving_weight
                calories = round(calories * factor, 2)
                carbs = round(carbs * factor, 2)
                fat = round(fat * factor, 2)
                protein = round(protein * factor, 2)
                fiber = round(fiber * factor, 2)

            nutrition = {
                "calories": calories,
                "carbohydrates": carbs,
                "fat": fat,
                "protein": protein,
                "fiber": fiber,
                "serving_weight_grams": serving_weight,
                "food_name": dish_name
            }
            
            # Cache the result
            nutrition_cache[cache_key] = nutrition
            return nutrition

        except Exception as e:
            logger.error(f"Failed to fetch detailed nutrition: {e}")
            return {}

    def close(self):
        """Close the requests session."""
        self.session.close()

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
async def predict_food(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    health_category: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Predict food class from image and return nutrition and personalized health information."""
    try:
        # Validate user profile data
        if age < 18 or age > 100:
            raise HTTPException(status_code=400, detail="Age must be between 18 and 100")
            
        if sex not in ["Male", "Female"]:
            raise HTTPException(status_code=400, detail="Sex must be 'Male' or 'Female'")
            
        if bmi < 15 or bmi > 50:
            raise HTTPException(status_code=400, detail="BMI must be between 15 and 50")
            
        if health_category not in ["Normal", "Pre-diabetic", "Diabetic"]:
            raise HTTPException(status_code=400, detail="Health category must be 'Normal', 'Pre-diabetic', or 'Diabetic'")
        
        # Create user profile
        user_profile = UserProfile(
            age=age,
            sex=sex,
            bmi=bmi,
            health_category=health_category
        )
        
        # Save and process uploaded file
        temp_path = await ImageProcessor.save_upload_file(file)
        background_tasks.add_task(cleanup_temp_file, temp_path)

        # Process image
        x = ImageProcessor.process_image(temp_path)
        
        # Make prediction with GPU acceleration
        with tf.device('/CPU:0' if has_gpu else '/CPU:0'):
            # Use model in inference mode with XLA optimization
            prediction = food_model(x, training=False)[0]

        # Get prediction results
        top_idx = np.argmax(prediction)
        predicted_dish = class_names[top_idx]
        confidence = float(prediction[top_idx])

        # Fetch nutrition info
        nutrition = nutrition_client.fetch_nutrition(predicted_dish)
        
        # Get glycemic index and load
        gi = lookup_gi_from_csv(predicted_dish)
        carbs = nutrition.get("carbohydrates", None)
        gl = calculate_gl(carbs, gi)
        
        # Add glycemic info to nutrition data for glucose prediction
        nutrition["glycemic_index"] = gi
        nutrition["glycemic_load"] = gl
        
        # Predict glucose impact
        glucose_impact = predict_glucose_impact(user_profile, nutrition)

        # Log prediction stats
        logger.info(
            f"Prediction completed - Class: {predicted_dish}, "
            f"Confidence: {confidence:.2f}, "
            f"Device: {device}, "
            f"User profile: {health_category}"
        )

        return PredictionResponse(
            predicted_class=predicted_dish,
            confidence=confidence,
            nutrition=nutrition,
            glycemic_index=str(gi),
            glycemic_load=str(gl),
            glucose_impact=glucose_impact,
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "models": {
            "food_classification": food_model is not None,
            "glucose_prediction": glucose_model is not None
        }
    }

@app.get("/analyze_meal/")
async def analyze_meal(
    foods: List[str] = Query(...),
    age: int = Query(...),
    sex: str = Query(...),
    bmi: float = Query(...),
    health_category: str = Query(...)
):
    """Analyze a complete meal with multiple food items."""
    try:
        # Validate user profile data
        if age < 18 or age > 100:
            raise HTTPException(status_code=400, detail="Age must be between 18 and 100")
            
        if sex not in ["Male", "Female"]:
            raise HTTPException(status_code=400, detail="Sex must be 'Male' or 'Female'")
            
        if bmi < 15 or bmi > 50:
            raise HTTPException(status_code=400, detail="BMI must be between 15 and 50")
            
        if health_category not in ["Normal", "Pre-diabetic", "Diabetic"]:
            raise HTTPException(status_code=400, detail="Health category must be 'Normal', 'Pre-diabetic', or 'Diabetic'")
        
        # Create user profile
        user_profile = UserProfile(
            age=age,
            sex=sex,
            bmi=bmi,
            health_category=health_category
        )
        
        meal_results = []
        total_calories = 0
        total_carbs = 0
        total_fat = 0
        total_protein = 0
        total_fiber = 0
        weighted_gi = 0
        total_gi_carbs = 0
        
        # Process each food item
        for food in foods:
            # Fetch nutrition info
            nutrition = nutrition_client.fetch_nutrition(food)
            
            # Skip items with no nutrition data
            if not nutrition:
                continue
                
            # Get glycemic index and load
            gi = lookup_gi_from_csv(food)
            carbs = nutrition.get("carbohydrates", 0)
            gl = calculate_gl(carbs, gi)
            
            # Add glycemic info to nutrition data for glucose prediction
            nutrition["glycemic_index"] = gi
            nutrition["glycemic_load"] = gl
            
            # Accumulate totals for meal analysis
            total_calories += nutrition.get("calories", 0)
            total_carbs += nutrition.get("carbohydrates", 0)
            total_fat += nutrition.get("fat", 0)
            total_protein += nutrition.get("protein", 0)
            total_fiber += nutrition.get("fiber", 0)
            
            # For weighted GI calculation
            if isinstance(gi, int) and carbs > 0:
                weighted_gi += gi * carbs
                total_gi_carbs += carbs
            
            # Add individual food results
            meal_results.append({
                "food": food,
                "nutrition": nutrition,
                "glycemic_index": str(gi),
                "glycemic_load": str(gl)
            })
        
        # Calculate meal-level glycemic index (weighted average)
        meal_gi = round(weighted_gi / total_gi_carbs) if total_gi_carbs > 0 else "Unknown"
        meal_gl = calculate_gl(total_carbs, meal_gi if isinstance(meal_gi, int) else None)
        
        # Create combined nutrition data for the whole meal
        meal_nutrition = {
            "calories": round(total_calories, 2),
            "carbohydrates": round(total_carbs, 2),
            "fat": round(total_fat, 2),
            "protein": round(total_protein, 2),
            "fiber": round(total_fiber, 2),
            "glycemic_index": meal_gi,
            "glycemic_load": meal_gl,
            "food_name": "Complete meal"
        }
        
        # Predict glucose impact of the full meal
        glucose_impact = predict_glucose_impact(user_profile, meal_nutrition)
        
        return {
            "individual_foods": meal_results,
            "meal_summary": {
                "total_nutrition": meal_nutrition,
                "meal_glycemic_index": str(meal_gi),
                "meal_glycemic_load": str(meal_gl),
                "glucose_impact": glucose_impact
            },
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
