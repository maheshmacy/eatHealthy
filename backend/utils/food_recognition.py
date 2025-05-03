import os
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

class NutritionAnalyzer:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.model = None
        self.labels = []
        self.img_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.setup_default_model()
            
        if labels_path and os.path.exists(labels_path):
            self.load_labels(labels_path)
            
    def setup_default_model(self) -> None:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(1000, activation='softmax')(global_avg)
        self.model = tf.keras.Model(inputs=base_model.input, outputs=output)
        
    def load_model(self, model_path: str) -> None:
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.setup_default_model()
    
    def load_labels(self, labels_path: str) -> None:
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Failed to load labels: {e}")
            
    def preprocess_image(self, img_path: str) -> np.ndarray:
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    def analyze_from_path(self, img_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(img_path):
            return [{"error": "Image file not found"}]
            
        if not self.model:
            return [{"error": "Model not loaded"}]
            
        processed_img = self.preprocess_image(img_path)
        
        try:
            predictions = self.model.predict(processed_img)
            top_indices = np.argsort(predictions[0])[::-1][:5]
            
            results = []
            for i in top_indices:
                if i < len(self.labels):
                    food_name = self.labels[i]
                else:
                    food_name = f"Unknown-{i}"
                    
                confidence = float(predictions[0][i])
                results.append({
                    "food_name": food_name,
                    "confidence": confidence,
                    "nutrition": self.get_nutrition_data(food_name)
                })
            return results
        except Exception as e:
            return [{"error": f"Prediction failed: {str(e)}"}]
    
    def analyze_from_array(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return [{"error": "Model not loaded"}]
            
        if img_array.shape[0] != self.img_size[0] or img_array.shape[1] != self.img_size[1]:
            img_array = cv2.resize(img_array, self.img_size)
            
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        try:
            predictions = self.model.predict(img_array)
            top_indices = np.argsort(predictions[0])[::-1][:5]
            
            results = []
            for i in top_indices:
                if i < len(self.labels):
                    food_name = self.labels[i]
                else:
                    food_name = f"Unknown-{i}"
                    
                confidence = float(predictions[0][i])
                results.append({
                    "food_name": food_name,
                    "confidence": confidence,
                    "nutrition": self.get_nutrition_data(food_name)
                })
            return results
        except Exception as e:
            return [{"error": f"Prediction failed: {str(e)}"}]
    
    def get_nutrition_data(self, food_name: str) -> Dict[str, float]:
        nutrition_db = {
            "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
            "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
            "orange": {"calories": 47, "protein": 0.9, "carbs": 12, "fat": 0.1},
            "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
            "burger": {"calories": 354, "protein": 20, "carbs": 29, "fat": 17},
            "salad": {"calories": 152, "protein": 7, "carbs": 11, "fat": 8},
            "pasta": {"calories": 158, "protein": 5.8, "carbs": 31, "fat": 0.9},
            "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
            "sandwich": {"calories": 283, "protein": 14, "carbs": 34, "fat": 9},
            "soup": {"calories": 124, "protein": 6, "carbs": 14, "fat": 4},
        }
        
        normalized_name = food_name.lower().strip()
        for key in nutrition_db:
            if key in normalized_name:
                return nutrition_db[key]
                
        return {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "note": "Nutrition data not available"
        }
        
    def capture_and_analyze(self, camera_id: int = 0) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return {"error": "Failed to open camera"}
                
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {"error": "Failed to capture image"}
                
            results = self.analyze_from_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return {"status": "success", "results": results}
        except Exception as e:
            return {"error": f"Camera capture failed: {str(e)}"}
    
    def estimate_portion_size(self, img_path: str, reference_object: Optional[str] = None) -> Dict[str, Any]:
        try:
            img = cv2.imread(img_path)
            if img is None:
                return {"error": "Image not loaded"}
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {"error": "No contours found"}
                
            largest_contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(largest_contour)
            
            if reference_object == "credit_card":
                reference_area_cm2 = 46.0  # Standard credit card area
                pixel_to_cm2_ratio = reference_area_cm2 / area_pixels
            else:
                # Rough estimate without reference
                pixel_to_cm2_ratio = 0.01
                
            estimated_area_cm2 = area_pixels * pixel_to_cm2_ratio
            
            # Rough volume estimation (assuming average height)
            estimated_volume_cm3 = estimated_area_cm2 * 2
            
            return {
                "estimated_area_cm2": estimated_area_cm2,
                "estimated_volume_cm3": estimated_volume_cm3,
                "estimated_portion": self.volume_to_portion(estimated_volume_cm3)
            }
        except Exception as e:
            return {"error": f"Portion estimation failed: {str(e)}"}
    
    def volume_to_portion(self, volume_cm3: float) -> str:
        if volume_cm3 < 100:
            return "small"
        elif volume_cm3 < 300:
            return "medium"
        else:
            return "large"
