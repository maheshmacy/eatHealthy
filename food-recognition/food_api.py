import os
import requests
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

glycemic_index_map = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NutritionInfo(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: Optional[float] = None
    sugar: Optional[float] = None
    glycemic_index: Optional[int] = None
    glycemic_load: Optional[float] = None
    glycemic_category: Optional[str] = None
    serving_size_grams: int = 100

    class Config:
        schema_extra = {
            "example": {
                "calories": 52.0,
                "protein": 0.7,
                "carbs": 13.8,
                "fat": 0.4,
                "fiber": 2.4,
                "sugar": 10.3,
                "glycemic_index": 38,
                "glycemic_load": 5.2,
                "glycemic_category": "low",
                "serving_size_grams": 100
            }
        }

class Config:
    MODEL_DIR = Path("models")
    DATA_DIR = Path("food_dataset/data")
    TEMP_DIR = Path("temp")
    UPLOAD_DIR = Path("uploads")
    GI_CSV_PATH = Path("glycemic_index.csv")
    USDA_API_KEY = ""
    USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
    USDA_DETAILS_URL = "https://api.nal.usda.gov/fdc/v1/food"
    NUTRITIONIX_APP_ID = ""
    NUTRITIONIX_API_KEY =""
    NUTRITIONIX_API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    API_TITLE = "Food Recognition API"
    API_DESCRIPTION = """
    API for food image classification and nutrition information.
    
    This API provides endpoints to:
    - Predict food class from an uploaded image
    - Get nutrition information for a specific food
    - List all available food classes
    
    The API uses a deep learning model trained on a dataset of food images
    including prepared dishes, fruits, and beverages.
    """
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    API_KEY_NAME = "X-API-Key"
    API_KEYS = ["test_key", "development_key"]
    
    IMG_SIZE = (224, 224)
    
    MAX_FILE_SIZE = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    CONFIDENCE_THRESHOLD = 0.5
    TOP_K = 3
    
    GI_THRESHOLDS = {
        "low": 55,
        "medium": 70,
        "high": 100
    }

def fetch_nutrition_from_usda(food_name: str) -> Optional[NutritionInfo]:
    try:
        search_params = {
            "query": food_name,
            "api_key": Config.USDA_API_KEY,
            "pageSize": 1
        }
        response = requests.get(Config.USDA_SEARCH_URL, params=search_params)
        if response.status_code != 200:
            logger.error(f"USDA search failed: {response.text}")
            return None
        
        search_data = response.json()
        if not search_data.get("foods"):
            return None
        
        fdc_id = search_data["foods"][0]["fdcId"]

        details_url = f"{Config.USDA_DETAILS_URL}/{fdc_id}"
        response = requests.get(details_url, params={"api_key": Config.USDA_API_KEY})
        if response.status_code != 200:
            logger.error(f"USDA details fetch failed: {response.text}")
            return None
        
        details = response.json()
        
        serving_size = details.get("servingSize", 100)
        
        nutrients = {}
        for n in details.get("foodNutrients", []):
            name = n.get("nutrientName") or (n.get("nutrient") or {}).get("name")
            value = n.get("value") if "value" in n else n.get("amount")
            if name and value is not None:
                nutrients[name.strip().lower()] = value

        calories = nutrients.get("energy", 0)
        protein = nutrients.get("protein", 0)
        carbs = nutrients.get("carbohydrate, by difference", 0)
        fat = nutrients.get("total lipid (fat)", 0)
        fiber = nutrients.get("fiber, total dietary", None)
        sugar = nutrients.get("sugars, total including nlea", None)

        gi = glycemic_index_map.get(food_name.lower())
        gl = round((carbs * gi) / 100, 1) if gi and carbs else None
        category = None
        if gi is not None:
            if gi <= Config.GI_THRESHOLDS['low']:
                category = "low"
            elif gi <= Config.GI_THRESHOLDS['medium']:
                category = "medium"
            else:
                category = "high"

        return NutritionInfo(
            calories=calories,
            protein=protein,
            carbs=carbs,
            fat=fat,
            fiber=fiber,
            sugar=sugar,
            glycemic_index=gi,
            glycemic_load=gl,
            glycemic_category=category,
            serving_size_grams=serving_size
        )

    except Exception as e:
        logger.error(f"Failed to fetch USDA nutrition data: {e}")
        return None

def fetch_nutrition_from_nutritionix(food_name: str) -> Optional[NutritionInfo]:
    try:
        headers = {
            "x-app-id": Config.NUTRITIONIX_APP_ID,
            "x-app-key": Config.NUTRITIONIX_API_KEY,
            "Content-Type": "application/json"
        }
        data = {"query": food_name, "timezone": "US/Eastern"}
        response = requests.post(Config.NUTRITIONIX_API_URL, json=data, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Nutritionix failed for {food_name}: {response.text}")
            return None
        
        item = response.json().get("foods", [{}])[0]
        return NutritionInfo(
            calories=item.get("nf_calories", 0),
            protein=item.get("nf_protein", 0),
            carbs=item.get("nf_total_carbohydrate", 0),
            fat=item.get("nf_total_fat", 0),
            fiber=item.get("nf_dietary_fiber"),
            sugar=item.get("nf_sugars"),
            glycemic_index=glycemic_index_map.get(food_name.lower()),
            glycemic_load=round(
                (item.get("nf_total_carbohydrate", 0) * glycemic_index_map.get(food_name.lower(), 60)) / 100, 2
            ),
            glycemic_category=None,
            serving_size_grams=100
        )
    except Exception as e:
        logger.error(f"Nutritionix fallback failed: {e}")
        return None
    
class SecurityConfig:
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
    
    ALLOWED_HEADERS = [
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods",
        "Access-Control-Allow-Headers"
    ]
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }

class FoodAlternative(BaseModel):
    food_name: str
    confidence: float

    class Config:
        schema_extra = {
            "example": {
                "food_name": "apple",
                "confidence": 0.15
            }
        }

class PredictionResponse(BaseModel):
    food_name: str
    confidence: float
    nutrition: Optional[NutritionInfo] = None
    alternatives: List[FoodAlternative] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "apple",
                "confidence": 0.92,
                "nutrition": {
                    "calories": 52.0,
                    "protein": 0.7,
                    "carbohydrates": 13.8,
                    "fat": 0.4,
                    "fiber": 2.4,
                    "sugar": 10.3,
                    "serving_size_grams": 100
                },
                "glycemic_index": 38,
                "glycemic_load": 5.2,
                "glycemic_category": "low",
                "timestamp": "2025-05-01T12:30:45.123Z"
            }
        }

class PredictionResponseSimple(BaseModel):
    predicted_class: str
    confidence: float
    nutrition: Dict[str, float]
    glycemic_index: Optional[str]
    glycemic_load: Optional[str]
    timestamp: datetime

class FoodListResponse(BaseModel):
    foods: List[str]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "foods": ["apple", "banana", "broccoli", "chicken", "pizza"],
                "count": 5
            }
        }

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    version: str

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-05-01T12:30:45.123Z",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }

app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

api_key_header = APIKeyHeader(name=Config.API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key is None and os.environ.get("ENVIRONMENT", "development") == "development":
        return None
    
    if api_key not in Config.API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    return api_key

app.add_middleware(
    CORSMiddleware,
    allow_origins=SecurityConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=SecurityConfig.ALLOWED_METHODS,
    allow_headers=SecurityConfig.ALLOWED_HEADERS,
    expose_headers=["*"],
    max_age=3600,
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    for header, value in SecurityConfig.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

model = None
class_mapping = {}
nutrition_db = None

try:
    for dir_path in [Config.TEMP_DIR, Config.UPLOAD_DIR]:
        dir_path.mkdir(exist_ok=True)
except Exception as e:
    logger.error(f"Error creating directories: {e}")

def load_model_and_data():
    global model, class_mapping, nutrition_db
    gi_csv_path = Config.GI_CSV_PATH
    if gi_csv_path.exists():
        global glycemic_index_map
        gi_df = pd.read_csv(gi_csv_path)
        glycemic_index_map = {
            row["food_name"].lower(): row["glycemic_index"] for _, row in gi_df.iterrows()
        }
        logger.info(f"Loaded glycemic index table with {len(glycemic_index_map)} entries")
    else:
        logger.warning("Glycemic index CSV not found")
        glycemic_index_map = {}
    try:
        model_dirs = sorted([d for d in Config.MODEL_DIR.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')],
                       key=lambda d: d.stat().st_mtime, reverse=True)
        
        if not model_dirs:
            logger.warning("No model directories found. Using default paths.")
            latest_model_dir = Config.MODEL_DIR
        else:
            latest_model_dir = model_dirs[0]
            logger.info(f"Using latest model directory: {latest_model_dir}")
        
        saved_model_path = latest_model_dir / "saved_model"
        if saved_model_path.exists():
            model_path = saved_model_path
            logger.info(f"Loading SavedModel from {model_path}")
            model = tf.keras.models.load_model(model_path)
        else:
            model_path = latest_model_dir / "final_model.h5"
            logger.info(f"Loading H5 model from {model_path}")
            model = tf.keras.models.load_model(model_path)
        
        logger.info("Model loaded successfully")
        
        class_mapping_path = latest_model_dir / "class_mapping.json"
        if class_mapping_path.exists():
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            logger.info(f"Loaded class mapping with {len(class_mapping)} classes")
        else:
            class_names_path = Config.DATA_DIR / "class_names.json"
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
                class_mapping = {str(i): name for i, name in enumerate(class_names)}
            logger.info(f"Created class mapping from {len(class_names)} class names")
        
        nutrition_db_path = Config.DATA_DIR / "food_nutrition.csv"
        if nutrition_db_path.exists():
            nutrition_db = pd.read_csv(nutrition_db_path)
            logger.info(f"Loaded nutrition database with {len(nutrition_db)} entries")
        else:
            logger.warning("Nutrition database not found")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model and data: {e}")
        return False

def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(Config.IMG_SIZE)
        
        x = np.array(img) / 255.0
        
        x = np.expand_dims(x, axis=0)
        
        return x
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def get_nutrition_info(food_name: str) -> Optional[NutritionInfo]:
    nutrition = fetch_nutrition_from_usda(food_name)
    if (
        not nutrition or 
        (nutrition.calories == 0 and nutrition.carbs == 0 and nutrition.fat == 0 and nutrition.protein == 0)
    ):
        logger.info(f"Falling back to Nutritionix for: {food_name}")
        nutrition = fetch_nutrition_from_nutritionix(food_name)
    return nutrition

async def cleanup_temp_file(file_path: Path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

async def save_upload_file(upload_file: UploadFile) -> Path:
    content = await upload_file.read(Config.MAX_FILE_SIZE + 1)
    
    if len(content) > Config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"upload_{timestamp}{file_ext}"
    
    temp_path = Config.TEMP_DIR / unique_filename
    
    with open(temp_path, 'wb') as f:
        f.write(content)
    
    upload_path = Config.UPLOAD_DIR / unique_filename
    
    with open(upload_path, 'wb') as f:
        f.write(content)
        
    return temp_path

@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "model_loaded": model is not None,
        "version": Config.API_VERSION
    }

@app.post("/predict/", response_model=PredictionResponseSimple, tags=["Prediction"])
async def predict_food(file: UploadFile = File(...), 
                       background_tasks: BackgroundTasks = None,
                       api_key: str = Security(get_api_key)):
    try:
        if model is None:
            load_successful = load_model_and_data()
            if not load_successful:
                raise HTTPException(status_code=500, detail="Model not loaded")
        
        temp_path = await save_upload_file(file)
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        with open(temp_path, 'rb') as f:
            image_data = f.read()
        
        x = preprocess_image(image_data)
        
        predictions = model.predict(x)[0]
        
        top_indices = np.argsort(predictions)[-Config.TOP_K:][::-1]
        top_confidences = predictions[top_indices]
        
        top_classes = [class_mapping[str(idx)] for idx in top_indices]
        
        best_class = top_classes[0]
        best_confidence = float(top_confidences[0])
        
        if best_confidence < Config.CONFIDENCE_THRESHOLD:
            return JSONResponse(
                status_code=200,
                content={
                    "food_name": "unknown",
                    "confidence": best_confidence,
                    "message": "Confidence too low for reliable prediction",
                    "alternatives": [],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        nutrition = get_nutrition_info(best_class)
        
        alternatives = [
            FoodAlternative(food_name=top_classes[i], confidence=float(top_confidences[i]))
            for i in range(1, min(len(top_classes), Config.TOP_K))
        ]
        
        logger.info(
            f"Prediction completed - Class: {best_class}, "
            f"Confidence: {best_confidence:.2f}"
        )
        
        return PredictionResponseSimple(
            predicted_class=best_class,
            confidence=best_confidence,
            nutrition={
                "calories": round(nutrition.calories, 1) if nutrition else 0,
                "carbohydrates": round(nutrition.carbs, 1) if nutrition else 0,
                "fat": round(nutrition.fat, 1) if nutrition else 0,
                "protein": round(nutrition.protein, 1) if nutrition else 0,
                "serving_weight_grams": nutrition.serving_size_grams if nutrition else 100
            },
            glycemic_index=str(nutrition.glycemic_index) if nutrition and nutrition.glycemic_index else None,
            glycemic_load=str(nutrition.glycemic_load) if nutrition and nutrition.glycemic_load else None,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/foods/", response_model=FoodListResponse, tags=["Food Data"])
async def list_foods(
    limit: int = Query(None, description="Maximum number of foods to return"),
    offset: int = Query(0, description="Number of foods to skip"),
    api_key: str = Security(get_api_key)
):
    try:
        if class_mapping == {}:
            load_successful = load_model_and_data()
            if not load_successful:
                raise HTTPException(status_code=500, detail="Model data not loaded")
        
        foods = sorted(set(class_mapping.values()))
        total_count = len(foods)
        
        if limit is not None:
            foods = foods[offset:offset + limit]
        else:
            foods = foods[offset:]
        
        return {"foods": foods, "count": total_count}
    
    except Exception as e:
        logger.error(f"Error listing foods: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/nutrition/{food_name}", response_model=NutritionInfo, tags=["Food Data"])
async def get_food_nutrition(
    food_name: str,
    api_key: str = Security(get_api_key)
):
    try:
        if nutrition_db is None:
            load_successful = load_model_and_data()
            if not load_successful:
                raise HTTPException(status_code=500, detail="Nutrition data not loaded")
        
        nutrition = get_nutrition_info(food_name)
        
        if nutrition is None:
            raise HTTPException(status_code=404, detail=f"Nutrition information not found for {food_name}")
        
        return nutrition
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting nutrition: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/uploads/{filename}", tags=["System"])
async def get_upload(
    filename: str,
    api_key: str = Security(get_api_key)
):
    file_path = Config.UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

app.mount("/static/uploads", StaticFiles(directory=str(Config.UPLOAD_DIR)), name="uploads")

@app.on_event("startup")
async def startup_event():
    load_model_and_data()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "food_api:app", 
        host=Config.HOST, 
        port=Config.PORT,
        reload=True
    )
