import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import random

def calculate_nutritional_needs(
    weight: float,
    height: float,
    age: int,
    gender: str,
    activity_level: str
) -> Dict[str, float]:
    """
    Calculate daily nutritional needs based on user metrics.
    
    Args:
        weight: Weight in kg
        height: Height in cm
        age: Age in years
        gender: 'male' or 'female'
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very active'
    
    Returns:
        Dictionary containing calories, protein, carbohydrates, and fat requirements
    """
    # Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Apply activity multiplier
    activity_multipliers = {
        'sedentary': 1.2,      # Little or no exercise
        'light': 1.375,        # Light exercise 1-3 days/week
        'moderate': 1.55,      # Moderate exercise 3-5 days/week
        'active': 1.725,       # Hard exercise 6-7 days/week
        'very active': 1.9     # Very hard exercise and physical job
    }
    
    daily_calories = bmr * activity_multipliers.get(activity_level.lower(), 1.55)
    
    # Calculate macronutrient distribution
    # Default macro split: 30% protein, 40% carbs, 30% fat
    protein_calories = daily_calories * 0.3
    carb_calories = daily_calories * 0.4
    fat_calories = daily_calories * 0.3
    
    # Convert calories to grams
    protein_grams = protein_calories / 4  # 4 calories per gram of protein
    carb_grams = carb_calories / 4       # 4 calories per gram of carbs
    fat_grams = fat_calories / 9         # 9 calories per gram of fat
    
    return {
        'calories': round(daily_calories),
        'protein': round(protein_grams),
        'carbohydrates': round(carb_grams),
        'fat': round(fat_grams)
    }

def format_nutrition_data(nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format raw nutrition data into a standardized format.
    
    Args:
        nutrition_data: Raw nutrition data from API
        
    Returns:
        Formatted nutrition data
    """
    formatted_data = {
        'calories': nutrition_data.get('calories', 0),
        'protein': nutrition_data.get('protein', 0),
        'carbohydrates': nutrition_data.get('carbohydrates', 0),
        'fat': nutrition_data.get('fat', 0),
        'fiber': nutrition_data.get('fiber', 0),
        'sugar': nutrition_data.get('sugar', 0),
        'sodium': nutrition_data.get('sodium', 0),
        'cholesterol': nutrition_data.get('cholesterol', 0),
        'vitamins': {}
    }
    
    # Extract vitamins and minerals if available
    for key, value in nutrition_data.items():
        if key.startswith('vitamin_') or key.startswith('mineral_'):
            formatted_data['vitamins'][key] = value
    
    return formatted_data

def generate_meal_plan(
    calorie_target: float,
    diet_preferences: List[str] = None,
    allergies: List[str] = None,
    foods_database: List[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate a daily meal plan based on nutritional targets and preferences.
    
    Args:
        calorie_target: Target calories per day
        diet_preferences: List of dietary preferences (e.g., 'vegetarian', 'keto')
        allergies: List of food allergies to avoid
        foods_database: Optional database of food items to choose from
            
    Returns:
        Dictionary with meal plan structured by meal type
    """
    if diet_preferences is None:
        diet_preferences = []
    
    if allergies is None:
        allergies = []
    
    # Use a simple default food database if none provided
    if foods_database is None:
        foods_database = [
            {
                'id': '1',
                'name': 'Oatmeal',
                'calories': 150,
                'protein': 6,
                'carbohydrates': 27,
                'fat': 2.5,
                'meal_types': ['breakfast'],
                'tags': ['vegetarian', 'vegan', 'dairy-free'],
                'allergens': ['gluten']
            },
            {
                'id': '2',
                'name': 'Scrambled Eggs',
                'calories': 200,
                'protein': 14,
                'carbohydrates': 2,
                'fat': 15,
                'meal_types': ['breakfast'],
                'tags': ['vegetarian', 'gluten-free'],
                'allergens': ['eggs']
            },
            {
                'id': '3',
                'name': 'Chicken Salad',
                'calories': 350,
                'protein': 30,
                'carbohydrates': 10,
                'fat': 20,
                'meal_types': ['lunch', 'dinner'],
                'tags': ['dairy-free', 'gluten-free'],
                'allergens': []
            },
            {
                'id': '4',
                'name': 'Tofu Stir Fry',
                'calories': 300,
                'protein': 15,
                'carbohydrates': 25,
                'fat': 15,
                'meal_types': ['lunch', 'dinner'],
                'tags': ['vegetarian', 'vegan'],
                'allergens': ['soy']
            },
            {
                'id': '5',
                'name': 'Greek Yogurt',
                'calories': 120,
                'protein': 15,
                'carbohydrates': 8,
                'fat': 0,
                'meal_types': ['breakfast', 'snack'],
                'tags': ['vegetarian', 'gluten-free'],
                'allergens': ['dairy']
            },
            {
                'id': '6',
                'name': 'Almonds',
                'calories': 160,
                'protein': 6,
                'carbohydrates': 6,
                'fat': 14,
                'meal_types': ['snack'],
                'tags': ['vegetarian', 'vegan', 'gluten-free', 'dairy-free'],
                'allergens': ['nuts']
            },
            {
                'id': '7',
                'name': 'Salmon Fillet',
                'calories': 250,
                'protein': 30,
                'carbohydrates': 0,
                'fat': 15,
                'meal_types': ['lunch', 'dinner'],
                'tags': ['dairy-free', 'gluten-free'],
                'allergens': ['fish']
            },
            {
                'id': '8',
                'name': 'Quinoa Bowl',
                'calories': 350,
                'protein': 12,
                'carbohydrates': 60,
                'fat': 8,
                'meal_types': ['lunch', 'dinner'],
                'tags': ['vegetarian', 'vegan', 'gluten-free', 'dairy-free'],
                'allergens': []
            },
            {
                'id': '9',
                'name': 'Protein Shake',
                'calories': 200,
                'protein': 25,
                'carbohydrates': 10,
                'fat': 5,
                'meal_types': ['breakfast', 'snack'],
                'tags': ['gluten-free'],
                'allergens': ['dairy']
            },
            {
                'id': '10',
                'name': 'Avocado Toast',
                'calories': 280,
                'protein': 8,
                'carbohydrates': 30,
                'fat': 16,
                'meal_types': ['breakfast', 'lunch'],
                'tags': ['vegetarian', 'vegan'],
                'allergens': ['gluten']
            }
        ]
    
    # Filter foods based on dietary preferences and allergies
    filtered_foods = foods_database.copy()
    
    # Filter out foods that contain allergens
    if allergies:
        filtered_foods = [
            food for food in filtered_foods
            if not any(allergen.lower() in [a.lower() for a in food['allergens']] for allergen in allergies)
        ]
    
    # Filter based on dietary preferences
    if diet_preferences:
        filtered_foods = [
            food for food in filtered_foods
            if any(pref.lower() in [t.lower() for t in food['tags']] for pref in diet_preferences)
        ]
    
    # Generate meal plan
    meal_plan = {
        'breakfast': [],
        'lunch': [],
        'dinner': [],
        'snacks': []
    }
    
    # Allocate calories for each meal
    breakfast_calories = calorie_target * 0.25  # 25% for breakfast
    lunch_calories = calorie_target * 0.30      # 30% for lunch
    dinner_calories = calorie_target * 0.30     # 30% for dinner
    snack_calories = calorie_target * 0.15      # 15% for snacks
    
    # Select foods for breakfast
    breakfast_foods = [food for food in filtered_foods if 'breakfast' in food['meal_types']]
    if breakfast_foods:
        current_calories = 0
        while current_calories < breakfast_calories and breakfast_foods:
            food = random.choice(breakfast_foods)
            meal_plan['breakfast'].append(food)
            current_calories += food['calories']
    
    # Select foods for lunch
    lunch_foods = [food for food in filtered_foods if 'lunch' in food['meal_types']]
    if lunch_foods:
        current_calories = 0
        while current_calories < lunch_calories and lunch_foods:
            food = random.choice(lunch_foods)
            meal_plan['lunch'].append(food)
            current_calories += food['calories']
    
    # Select foods for dinner
    dinner_foods = [food for food in filtered_foods if 'dinner' in food['meal_types']]
    if dinner_foods:
        current_calories = 0
        while current_calories < dinner_calories and dinner_foods:
            food = random.choice(dinner_foods)
            meal_plan['dinner'].append(food)
            current_calories += food['calories']
    
    # Select foods for snacks
    snack_foods = [food for food in filtered_foods if 'snack' in food['meal_types']]
    if snack_foods:
        current_calories = 0
        while current_calories < snack_calories and snack_foods:
            food = random.choice(snack_foods)
            meal_plan['snacks'].append(food)
            current_calories += food['calories']
    
    return meal_plan