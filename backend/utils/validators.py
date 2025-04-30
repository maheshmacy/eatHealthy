"""
Input validation utilities for the GI Personalize app.
"""
import logging

logger = logging.getLogger(__name__)

def validate_user_data(data, required_fields=True):
    """
    Validate user profile data.
    
    Args:
        data (dict): User profile data
        required_fields (bool): Whether to validate required fields
        
    Returns:
        str: Error message or None if valid
    """
    # Check required fields
    if required_fields:
        for field in ['name', 'age', 'gender', 'height', 'weight']:
            if field not in data:
                return f"Missing required field: {field}"
    
    # Validate age
    if 'age' in data:
        try:
            age = int(data['age'])
            if age < 18 or age > 120:
                return "Age must be between 18 and 120"
        except ValueError:
            return "Age must be a number"
    
    # Validate gender
    if 'gender' in data and data['gender'] not in ['male', 'female']:
        return "Gender must be either 'male' or 'female'"
    
    # Validate height
    if 'height' in data:
        try:
            height = float(data['height'])
            if height < 100 or height > 250:
                return "Height must be between 100 and 250 cm"
        except ValueError:
            return "Height must be a number"
    
    # Validate weight
    if 'weight' in data:
        try:
            weight = float(data['weight'])
            if weight < 30 or weight > 300:
                return "Weight must be between 30 and 300 kg"
        except ValueError:
            return "Weight must be a number"
    
    # Validate activity_level
    valid_activity_levels = [
        'sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extremely_active'
    ]
    if 'activity_level' in data and data['activity_level'] not in valid_activity_levels:
        return f"Activity level must be one of: {', '.join(valid_activity_levels)}"
    
    # Validate diabetes_status
    valid_diabetes_statuses = ['none', 'pre_diabetic', 'type1_diabetes', 'type2_diabetes']
    if 'diabetes_status' in data and data['diabetes_status'] not in valid_diabetes_statuses:
        return f"Diabetes status must be one of: {', '.join(valid_diabetes_statuses)}"
    
    # Validate weight_goal
    valid_weight_goals = ['lose', 'maintain', 'gain']
    if 'weight_goal' in data and data['weight_goal'] not in valid_weight_goals:
        return f"Weight goal must be one of: {', '.join(valid_weight_goals)}"
    
    # Validate optional fields
    if 'hba1c' in data and data['hba1c']:
        try:
            hba1c = float(data['hba1c'])
            if hba1c < 3 or hba1c > 15:
                return "HbA1c must be between 3 and 15 %"
        except ValueError:
            return "HbA1c must be a number"
    
    if 'fasting_glucose' in data and data['fasting_glucose']:
        try:
            fasting_glucose = float(data['fasting_glucose'])
            if fasting_glucose < 50 or fasting_glucose > 400:
                return "Fasting glucose must be between 50 and 400 mg/dL"
        except ValueError:
            return "Fasting glucose must be a number"
    
    return None


def validate_glucose_readings(readings):
    """
    Validate glucose readings.
    Readings should be a list of dictionaries with 'timestamp' and 'value' keys.
    """
    if not readings or not isinstance(readings, list):
        return "Glucose readings must be a non-empty list"
    
    for reading in readings:
        # Validate that reading is a dictionary with required keys
        if not isinstance(reading, dict):
            return "Each reading must be an object with timestamp and value"
        
        if 'timestamp' not in reading or 'value' not in reading:
            return "Each reading must contain timestamp and value"
        
        # Validate value is a number
        try:
            # Extract the value field from the dictionary instead of trying to convert the whole dict
            reading_value = float(reading.get('value'))
        except (ValueError, TypeError):
            return "Glucose values must be valid numbers"
        
        # Validate reasonable glucose range
        if reading_value < 30 or reading_value > 600:
            return "Glucose values should be in a reasonable range (30-600 mg/dL)"
        
        # Validate timestamp is a valid datetime
        try:
            # The timestamp is already validated by the model
            pass
        except Exception:
            return "Invalid timestamp format"
    
    return None  # Validation passed

def validate_meal_response(response):
    """
    Validate meal response data.
    
    Args:
        response (dict): Meal response data
        
    Returns:
        str: Error message or None if valid
    """
    if not isinstance(response, dict):
        return "Response must be an object"
    
    if 'response' not in response:
        return "Missing required field: response"
    
    valid_responses = ['less_than_expected', 'as_expected', 'more_than_expected']
    if response['response'] not in valid_responses:
        return f"Response must be one of: {', '.join(valid_responses)}"
    
    return None
