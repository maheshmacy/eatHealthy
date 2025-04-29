import os
import numpy as np
import logging
import asyncio
import aiohttp
import aiofiles
import requests
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    title="Food Classification API",
    description="API for food image classification and nutrition information",
    version="2.0.0"
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
    serving_weight_grams: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    nutrition: Optional[Dict[str, float]]
    glycemic_index: str
    glycemic_load: str
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
    logger.info(f"Model loaded successfully on {device}. Found {len(class_names)} classes")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
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
            search_url = "https://api.spoonacular.com/recipes/complexSearch"
            params = {
                "apiKey": Config.SPOONACULAR_API_KEY,
                "query": dish_name,
                "number": 1
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                return data["results"][0]["id"]
            else:
                return None
        except Exception as e:
            print(f"⚠️ Failed to fetch recipe ID: {e}")
            return None

    def fetch_nutrition(self, dish_name, normalize_to_100g=True):
        """Fetch detailed nutrition and optionally normalize per 100g."""
        try:
            recipe_id = self.fetch_recipe_id(dish_name)
            print("receipe id", recipe_id)
            if recipe_id is None:
                return {}

            nutrition_url = f"https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json"
            params = {"apiKey": Config.SPOONACULAR_API_KEY}
            response = requests.get(nutrition_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract fields
            calories = float(data["calories"].replace("kcal", "").strip())
            carbs = float(data["carbs"].replace("g", "").strip())
            fat = float(data["fat"].replace("g", "").strip())
            protein = float(data["protein"].replace("g", "").strip())
            serving_weight = float(data.get("serving_weight_grams", 100))  # Default 100g if missing

            # Normalize to 100g if needed
            if normalize_to_100g and serving_weight != 100:
                factor = 100 / serving_weight
                calories = round(calories * factor, 2)
                carbs = round(carbs * factor, 2)
                fat = round(fat * factor, 2)
                protein = round(protein * factor, 2)

            nutrition = {
                "calories": calories,
                "carbohydrates": carbs,
                "fat": fat,
                "protein": protein,
                "serving_weight_grams": serving_weight
            }
            return nutrition

        except Exception as e:
            print(f"⚠️ Failed to fetch detailed nutrition: {e}")
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

def fetch_gi_openfoodfacts(dish_name):
    """Fetch glycemic index estimate dynamically from OpenFoodFacts."""
    try:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": dish_name,
            "json": 1,
            "simple_search": 1,
            "page_size": 1
        }
        response = requests.get(url, params=params)
        print(url)
        json_response = response.json()
        print(json_response)
        response.raise_for_status()
        products = response.json().get("products", [])
        if not products:
            return "Unknown"

        product = products[0]
        categories = product.get("categories_tags", [])

        # If category indicates glycemic index directly
        for cat in categories:
            if "low-glycemic-index" in cat:
                return 40
            if "high-glycemic-index" in cat:
                return 80

        # Otherwise infer based on general category
        if any(keyword in cat for keyword in categories for keyword in ["bread", "pizza", "pasta", "rice", "cereal"]):
            return 70  # High GI
        if any(keyword in cat for keyword in categories for keyword in ["vegetable", "salad", "nuts", "legume"]):
            return 35  # Low GI

        return "Unknown"

    except Exception as e:
        print(f"⚠️ OpenFoodFacts GI fetch failed: {e}")
        return "Unknown"


@app.post("/predict/", response_model=PredictionResponse)
async def predict_food(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Predict food class from image and return nutrition information."""
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
    return {"status": "healthy", "timestamp": datetime.now()}

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
