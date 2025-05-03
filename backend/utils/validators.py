import re
from typing import Dict, List, Any, Optional, Union, Callable
import datetime

def validate_email(email: str) -> bool:
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> Dict[str, Any]:
    result = {
        "valid": False,
        "errors": []
    }
    
    if len(password) < 8:
        result["errors"].append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        result["errors"].append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        result["errors"].append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        result["errors"].append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result["errors"].append("Password must contain at least one special character")
    
    result["valid"] = len(result["errors"]) == 0
    return result

def validate_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "errors": {}
    }
    
    required_fields = ["username", "email", "password"]
    for field in required_fields:
        if field not in user_data:
            result["valid"] = False
            result["errors"][field] = f"Missing required field: {field}"
    
    if "email" in user_data and not validate_email(user_data["email"]):
        result["valid"] = False
        result["errors"]["email"] = "Invalid email format"
    
    if "password" in user_data:
        pwd_validation = validate_password(user_data["password"])
        if not pwd_validation["valid"]:
            result["valid"] = False
            result["errors"]["password"] = pwd_validation["errors"]
    
    if "username" in user_data and len(user_data["username"]) < 3:
        result["valid"] = False
        result["errors"]["username"] = "Username must be at least 3 characters long"
    
    return result

def validate_food_data(food_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "errors": {}
    }
    
    required_fields = ["name", "category"]
    for field in required_fields:
        if field not in food_data:
            result["valid"] = False
            result["errors"][field] = f"Missing required field: {field}"
    
    numeric_fields = ["calories", "protein", "carbs", "fat"]
    for field in numeric_fields:
        if field in food_data:
            try:
                value = float(food_data[field])
                if value < 0:
                    result["valid"] = False
                    result["errors"][field] = f"{field.capitalize()} cannot be negative"
            except (ValueError, TypeError):
                result["valid"] = False
                result["errors"][field] = f"{field.capitalize()} must be a number"
    
    return result

def validate_date_format(date_str: str) -> bool:
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_log_data(log_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "errors": {}
    }
    
    required_fields = ["user_id", "item_id"]
    for field in required_fields:
        if field not in log_data:
            result["valid"] = False
            result["errors"][field] = f"Missing required field: {field}"
    
    if "quantity" in log_data:
        try:
            quantity = float(log_data["quantity"])
            if quantity <= 0:
                result["valid"] = False
                result["errors"]["quantity"] = "Quantity must be greater than zero"
        except (ValueError, TypeError):
            result["valid"] = False
            result["errors"]["quantity"] = "Quantity must be a number"
    
    if "meal_type" in log_data:
        valid_meal_types = ["breakfast", "lunch", "dinner", "snack"]
        if log_data["meal_type"].lower() not in valid_meal_types:
            result["valid"] = False
            result["errors"]["meal_type"] = f"Meal type must be one of: {', '.join(valid_meal_types)}"
    
    if "date" in log_data and not validate_date_format(log_data["date"]):
        result["valid"] = False
        result["errors"]["date"] = "Date must be in format YYYY-MM-DD"
    
    return result

def apply_validation_rules(data: Dict[str, Any], rules: Dict[str, Callable]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "errors": {}
    }
    
    for field, validation_func in rules.items():
        if field in data:
            is_valid, error_msg = validation_func(data[field])
            if not is_valid:
                result["valid"] = False
                result["errors"][field] = error_msg
    
    return result

def validate_image_format(image_path: str) -> bool:
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(image_path.lower().endswith(ext) for ext in valid_extensions)

def validate_image_dimensions(width: int, height: int) -> bool:
    min_dimension = 100
    max_dimension = 4000
    return (min_dimension <= width <= max_dimension and 
            min_dimension <= height <= max_dimension)

def validate_portion_size(portion_str: str) -> bool:
    valid_portions = ["small", "medium", "large", "extra large"]
    return portion_str.lower() in valid_portions

def validate_nutritional_data(nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "errors": {}
    }
    
    expected_fields = ["calories", "protein", "carbs", "fat"]
    for field in expected_fields:
        if field not in nutrition_data:
            result["valid"] = False
            result["errors"][field] = f"Missing nutritional data: {field}"
            continue
            
        try:
            value = float(nutrition_data[field])
            if value < 0:
                result["valid"] = False
                result["errors"][field] = f"{field.capitalize()} cannot be negative"
        except (ValueError, TypeError):
            result["valid"] = False
            result["errors"][field] = f"{field.capitalize()} must be a number"
    
    return result

def sanitize_input(input_str: str) -> str:
    return re.sub(r'[<>"\'&;]', '', input_str)

def normalize_food_name(food_name: str) -> str:
    return sanitize_input(food_name.lower().strip())
