from flask import Flask, request, jsonify
import requests
import json
import os
from datetime import datetime, timedelta
import random
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

api = Flask(__name__)
CORS(api)
api.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///nutrition_data.db"
api.config["SQLALCHEMY_TRACK_CHANGES"] = False
db_connection = SQLAlchemy(api)

class MealRecord(db_connection.Model):
    id = db_connection.Column(db_connection.Integer, primary_key=True)
    user_id = db_connection.Column(db_connection.Integer, nullable=False)
    meal_name = db_connection.Column(db_connection.String(100), nullable=False)
    meal_date = db_connection.Column(db_connection.DateTime, default=datetime.utcnow)
    calories = db_connection.Column(db_connection.Float, nullable=False)
    protein = db_connection.Column(db_connection.Float, nullable=False)
    carbs = db_connection.Column(db_connection.Float, nullable=False)
    fats = db_connection.Column(db_connection.Float, nullable=False)

class UserAccount(db_connection.Model):
    id = db_connection.Column(db_connection.Integer, primary_key=True)
    username = db_connection.Column(db_connection.String(50), unique=True, nullable=False)
    password_hash = db_connection.Column(db_connection.String(255), nullable=False)
    email = db_connection.Column(db_connection.String(100), unique=True, nullable=False)
    registration_date = db_connection.Column(db_connection.DateTime, default=datetime.utcnow)

food_api_key = os.environ.get('NUTRITION_API_KEY', 'default_key_for_testing')
food_base_url = "https://api.nutritionservice.com/v1"

def obtain_nutrient_info(dish_title):
    endpoint = f"{food_base_url}/foods/search"
    params = {"query": dish_title, "apiKey": food_api_key}
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if data and 'items' in data and len(data['items']) > 0:
            return data['items'][0]
    return None

def save_collaborative_meal(user_profile, dish_info):
    collab_api = "https://collaborativeservice.com/api/meals"
    headers = {"Authorization": f"Bearer {user_profile['access_token']}", "Content-Type": "application/json"}
    payload = {
        "user_id": user_profile["id"],
        "meal_data": dish_info,
        "timestamp": datetime.utcnow().isoformat()
    }
    response = requests.post(collab_api, headers=headers, json=payload)
    return response.json() if response.status_code == 200 else None

@api.route('/register', methods=['POST'])
def create_user():
    input_data = request.get_json()
    username = input_data.get('username')
    password = input_data.get('password')
    email = input_data.get('email')
    
    if not all([username, password, email]):
        return jsonify({"error": "Missing required fields"}), 400
    
    if UserAccount.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
    
    if UserAccount.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409
    
    password_hash = generate_password_hash(password)
    new_user = UserAccount(username=username, password_hash=password_hash, email=email)
    db_connection.session.add(new_user)
    db_connection.session.commit()
    
    return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201

@api.route('/login', methods=['POST'])
def authenticate_user():
    input_data = request.get_json()
    username = input_data.get('username')
    password = input_data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    user = UserAccount.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401
    
    return jsonify({
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
        "access_token": generate_temp_token(user.id)
    }), 200

def generate_temp_token(user_id):
    return f"temp_token_{user_id}_{int(datetime.utcnow().timestamp())}"

@api.route('/add_meal', methods=['POST'])
def record_meal():
    input_data = request.get_json()
    user_id = input_data.get('user_id')
    meal_name = input_data.get('meal_name')
    
    if not user_id or not meal_name:
        return jsonify({"error": "User ID and meal name required"}), 400
    
    nutrient_data = obtain_nutrient_info(meal_name)
    
    if not nutrient_data:
        return jsonify({"error": "Could not find nutrition data for the meal"}), 404
    
    calories = nutrient_data.get('calories', 0)
    protein = nutrient_data.get('protein', 0)
    carbs = nutrient_data.get('carbohydrates', 0)
    fats = nutrient_data.get('fat', 0)
    
    new_meal = MealRecord(
        user_id=user_id,
        meal_name=meal_name,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fats=fats
    )
    
    db_connection.session.add(new_meal)
    db_connection.session.commit()
    
    user = UserAccount.query.get(user_id)
    if user:
        user_profile = {"id": user.id, "username": user.username, "access_token": generate_temp_token(user.id)}
        collab_result = save_collaborative_meal(user_profile, {
            "meal_id": new_meal.id,
            "meal_name": meal_name,
            "nutrition": {
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fats": fats
            }
        })

    return jsonify({
        "message": "Meal recorded successfully",
        "meal_id": new_meal.id,
        "collaborative_status": "saved" if collab_result else "failed"
    }), 201

@api.route('/meals/<int:user_id>', methods=['GET'])
def get_meal_history(user_id):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    meals = MealRecord.query.filter_by(user_id=user_id).order_by(MealRecord.meal_date.desc()).paginate(page=page, per_page=per_page)
    
    result = []
    for meal in meals.items:
        result.append({
            "id": meal.id,
            "meal_name": meal.meal_name,
            "date": meal.meal_date.isoformat(),
            "nutrition": {
                "calories": meal.calories,
                "protein": meal.protein,
                "carbs": meal.carbs,
                "fats": meal.fats
            }
        })
    
    return jsonify({
        "meals": result,
        "total_pages": meals.pages,
        "current_page": meals.page,
        "total_items": meals.total
    }), 200

@api.route('/stats/<int:user_id>', methods=['GET'])
def get_nutrition_stats(user_id):
    days = request.args.get('days', 7, type=int)
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    meals = MealRecord.query.filter_by(user_id=user_id).filter(
        MealRecord.meal_date.between(start_date, end_date)
    ).all()
    
    if not meals:
        return jsonify({"error": "No meal data found for the specified period"}), 404
    
    total_calories = sum(meal.calories for meal in meals)
    total_protein = sum(meal.protein for meal in meals)
    total_carbs = sum(meal.carbs for meal in meals)
    total_fats = sum(meal.fats for meal in meals)
    
    daily_avg_calories = total_calories / days
    daily_avg_protein = total_protein / days
    daily_avg_carbs = total_carbs / days
    daily_avg_fats = total_fats / days
    
    return jsonify({
        "period_days": days,
        "total_meals": len(meals),
        "total_nutrition": {
            "calories": total_calories,
            "protein": total_protein,
            "carbs": total_carbs,
            "fats": total_fats
        },
        "daily_average": {
            "calories": daily_avg_calories,
            "protein": daily_avg_protein,
            "carbs": daily_avg_carbs,
            "fats": daily_avg_fats
        }
    }), 200

if __name__ == '__main__':
    with api.app_context():
        db_connection.create_all()
    api.run(debug=False, host='0.0.0.0', port=5000)
