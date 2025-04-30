"""
Food recognition utilities for the GI Personalize app.
"""
import os
import logging
import json
import random
from config import Config
from utils.gi_database import lookup_gi, get_food_nutritional_info

logger = logging.getLogger(__name__)

# In a real application, this would use TensorFlow/PyTorch for food recognition
# For demo purposes, we'll use a simplified mock implementation

# Load food categories database
def load_food_categories():
    try:
        with open(os.path.join('data', 'gi_database.csv'), 'r') as f:
            # Skip header
            next(f)
            food_categories = [line.split(',')[0] for line in f]
        return food_categories
    except Exception as e:
        logger.error(f"Error loading food categories: {str(e)}", exc_info=True)
        # Return some default categories
        return ["rice", "bread", "potato", "apple", "banana", "carrot", "chicken", "beef", "milk", "yogurt"]

# Get the food categories
food_categories = load_food_categories()

def identify_food_in_image(image_path):
    """
    Identify food items in an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        list: List of identified food items with confidence scores
    """
    try:
        logger.info(f"Analyzing image: {image_path}")
        
        # In a real application, you would:
        # 1. Load the image using a library like PIL or OpenCV
        # 2. Preprocess the image (resize, normalize, etc.)
        # 3. Feed it to a trained food recognition model
        # 4. Process the model's output to get food categories and confidence scores
        
        # For this demo, we'll simulate by returning random food items from our database
        num_items = random.randint(1, 3)  # Random number of foods (1-3)
        
        # Get random food items
        results = []
        used_foods = set()  # To avoid duplicates
        
        for _ in range(num_items):
            # Keep selecting until we get a unique food
            while True:
                food_name = random.choice(food_categories)
                if food_name not in used_foods:
                    used_foods.add(food_name)
                    break
                    
            confidence = round(random.uniform(0.7, 0.98), 2)  # Random confidence score
            results.append({
                "name": food_name,
                "confidence": confidence
            })
        
        logger.info(f"Found {len(results)} food items")
        return results
    
    except Exception as e:
        logger.error(f"Error identifying food in image: {str(e)}", exc_info=True)
        # Return a default food item if recognition fails
        return [{"name": "rice", "confidence": 0.8}]

def get_nutritional_info(food_name):
    """
    Get nutritional information for a food item.
    
    Args:
        food_name (str): Name of the food
        
    Returns:
        dict: Nutritional information or None if not found
    """
    # Try to get from our GI database
    nutritional_info = get_food_nutritional_info(food_name)
    
    if nutritional_info:
        return nutritional_info
    
    # Fallback to generic values
    return {
        "gi": lookup_gi(food_name),
        "gl": 10,  # Default glycemic load
        "carbs": 15,
        "protein": 5,
        "fat": 3,
        "fiber": 1,
        "serving_size": 100
    }
