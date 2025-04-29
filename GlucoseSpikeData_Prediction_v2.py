#!/usr/bin/env python
# coding: utf-8

# 

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. IMPROVED DATA SIMULATION WITH MORE FEATURES
# ==============================================

def generate_simulated_data(num_participants=30, days_per_participant=3):
    """
    Generates more realistic simulated glucose data with additional features:
    - Activity level
    - Stress level
    - Sleep quality
    - Medication status
    - Time since last meal
    """
    # Define participant categories
    categories = ['Normal', 'Pre-diabetic', 'Diabetic']
    participants_per_category = num_participants // len(categories)

    # Generate participant demographics and categories
    participants = []
    for category in categories:
        for _ in range(participants_per_category):
            pid = len(participants) + 1

            # Add more realistic age distribution per category
            if category == 'Normal':
                age = np.random.randint(22, 55)  # Younger population
            elif category == 'Pre-diabetic':
                age = np.random.randint(35, 65)  # Middle-aged population
            else:  # Diabetic
                age = np.random.randint(45, 75)  # Older population

            participant = {
                'Participant_ID': pid,
                'Age': age,
                'Sex': np.random.choice(['Male', 'Female']),
                'BMI': round(np.random.uniform(18, 35), 1),
                'Category': category,
                # Add new features
                'Takes_Medication': 1 if category == 'Diabetic' and random.random() > 0.2 else 0,
                'Family_History': 1 if category in ['Pre-diabetic', 'Diabetic'] or random.random() > 0.7 else 0
            }
            participants.append(participant)

    # Simulate data
    data = []
    start_date = datetime(2025, 1, 1)

    for participant in participants:
        for day in range(days_per_participant):
            date = start_date + timedelta(days=day)

            # Daily features that remain constant for the day
            sleep_quality = round(np.random.uniform(3, 10), 1)  # Scale of 1-10
            stress_level = round(np.random.uniform(1, 10), 1)   # Scale of 1-10

            # Simulate 3 meals per day with varying timings
            breakfast_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=7 + np.random.uniform(0, 2))
            lunch_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=12 + np.random.uniform(0, 2))
            dinner_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=18 + np.random.uniform(0, 2))

            meal_times = [breakfast_time, lunch_time, dinner_time]

            for meal_idx, meal_time in enumerate(meal_times):
                # Meal composition varies by time of day
                if meal_idx == 0:  # Breakfast
                    carbs = round(np.random.uniform(30, 80), 1)
                    fat = round(np.random.uniform(5, 20), 1)
                    fiber = round(np.random.uniform(2, 8), 1)
                    gi = round(np.random.uniform(50, 80), 0) # Breakfast tends to have higher GI
                    protein = round(np.random.uniform(10, 25), 1)
                elif meal_idx == 1:  # Lunch
                    carbs = round(np.random.uniform(40, 90), 1)
                    fat = round(np.random.uniform(10, 30), 1)
                    fiber = round(np.random.uniform(3, 10), 1)
                    gi = round(np.random.uniform(45, 75), 0)
                    protein = round(np.random.uniform(15, 40), 1)
                else:  # Dinner
                    carbs = round(np.random.uniform(30, 80), 1)
                    fat = round(np.random.uniform(10, 35), 1)
                    fiber = round(np.random.uniform(5, 15), 1)
                    gi = round(np.random.uniform(40, 70), 0)
                    protein = round(np.random.uniform(20, 50), 1)

                glycemic_load = round((gi * carbs) / 100, 1)

                # Activity around this meal
                activity_timing = np.random.choice(['before', 'after', 'none'])
                if activity_timing == 'none':
                    activity_level = round(np.random.uniform(0, 2), 1)  # Sedentary
                elif activity_timing == 'before':
                    activity_level = round(np.random.uniform(3, 8), 1)  # Moderate
                else:
                    activity_level = round(np.random.uniform(2, 7), 1)  # Light to moderate

                # Time since last meal (in hours) - first meal of day has longest gap
                if meal_idx == 0:
                    time_since_last_meal = round(np.random.uniform(8, 12), 1)  # Overnight fast
                else:
                    time_since_last_meal = round(np.random.uniform(3, 6), 1)

                # Simulate CGM readings for 2 hours post-meal at 5-minute intervals
                for i in range(24):  # 2 hours * 12 readings per hour
                    time = meal_time + timedelta(minutes=5 * i)

                    # Enhanced glucose simulation with more features influencing the outcome
                    glucose = simulate_glucose(
                        participant['Category'],
                        carbs,
                        gi,
                        i,
                        activity_level,
                        stress_level,
                        sleep_quality,
                        participant['Takes_Medication'],
                        time_since_last_meal,
                        participant['BMI'],
                        participant['Age'],
                        fiber,
                        fat,
                        protein
                    )

                    # Create entry with all our features
                    entry = {
                        'Participant_ID': participant['Participant_ID'],
                        'Age': participant['Age'],
                        'Sex': participant['Sex'],
                        'BMI': participant['BMI'],
                        'Category': participant['Category'],
                        'Meal_Time': meal_time,
                        'Carbs': carbs,
                        'Fat': fat,
                        'Fiber': fiber,
                        'Protein': protein,
                        'GI': gi,
                        'Glycemic_Load': glycemic_load,
                        'Activity_Level': activity_level,
                        'Activity_Timing': activity_timing,
                        'Stress_Level': stress_level,
                        'Sleep_Quality': sleep_quality,
                        'Takes_Medication': participant['Takes_Medication'],
                        'Family_History': participant['Family_History'],
                        'Time_Since_Last_Meal': time_since_last_meal,
                        'Minutes_After_Meal': i * 5,
                        'Timestamp': time,
                        'Glucose': round(glucose, 1),
                        'Day': day + 1
                    }
                    data.append(entry)

    return pd.DataFrame(data)

def simulate_glucose(category, carbs, gi, time_point, activity_level, stress, sleep, medication, time_since_meal, bmi, age, fiber, fat, protein):
    """
    Enhanced glucose simulation that accounts for multiple factors
    """
    # Base glucose levels by category
    if category == 'Normal':
        base = np.random.uniform(80, 100)
        decay_rate = 0.1  # Faster return to normal
    elif category == 'Pre-diabetic':
        base = np.random.uniform(100, 125)
        decay_rate = 0.08
    else:  # Diabetic
        base = np.random.uniform(126, 180)
        decay_rate = 0.05  # Slower return to normal

    # Calculate the spike based on carbs and GI
    max_effect = (carbs * gi / 100)

    # Calculate the shape of the curve - time in 5 minute increments
    time_in_hours = time_point * (5/60)

    # Glucose curve typically peaks around 30-45 minutes post-meal
    curve_shape = np.exp(-decay_rate * (time_in_hours - 0.6)**2)

    # Initial spike based on meal composition
    spike = max_effect * curve_shape

    # Adjust for other factors

    # Higher fiber reduces spike
    fiber_effect = -fiber * 0.5

    # Fat can slow absorption
    fat_effect = -fat * 0.2 + fat**2 * 0.01  # Non-linear effect

    # Protein can stimulate insulin
    protein_effect = -protein * 0.2 + protein**2 * 0.005

    # Activity reduces glucose (stronger effect for activity after meals)
    if time_in_hours < 0.5:  # Early in digestion
        activity_effect = -activity_level * 2.5
    else:
        activity_effect = -activity_level * 1.5

    # Stress increases glucose
    stress_effect = stress * 1.2

    # Poor sleep increases insulin resistance
    sleep_effect = -(sleep - 5) * 2  # Sleep below 5 increases glucose

    # Medication effect (diabetics)
    medication_effect = -25 * medication if category == 'Diabetic' else 0

    # BMI effect (higher BMI = higher glucose response)
    if bmi < 25:
        bmi_effect = 0
    elif bmi < 30:
        bmi_effect = 5
    else:
        bmi_effect = 10 + (bmi - 30) * 2

    # Age effect (higher age = slightly higher glucose)
    age_effect = (age - 40) * 0.2 if age > 40 else 0

    # Time since last meal (if it's been a long time, liver may release more glucose)
    meal_timing_effect = -5 if time_since_meal < 4 else 0

    # Combine all effects and add some randomness
    total_effect = (spike + fiber_effect + fat_effect + protein_effect + activity_effect +
                   stress_effect + sleep_effect + medication_effect + bmi_effect +
                   age_effect + meal_timing_effect)

    # Add some randomness to simulate individual variability and measurement error
    glucose = base + total_effect + np.random.normal(0, 5)

    # Ensure we don't go below reasonable values
    glucose = max(60, glucose)

    return glucose

# 2. MODEL TRAINING WITH CROSS-VALIDATION
# =======================================

def build_and_evaluate_model(df, time_based=False):
    """
    Build and evaluate a glucose prediction model with cross-validation
    If time_based is True, builds a model that predicts glucose at different time points
    """
    # Encode categorical variables
    label_encoder_sex = LabelEncoder()
    df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])

    label_encoder_category = LabelEncoder()
    df['Category'] = label_encoder_category.fit_transform(df['Category'])

    label_encoder_activity = LabelEncoder()
    df['Activity_Timing'] = label_encoder_activity.fit_transform(df['Activity_Timing'])

    # Define features
    if time_based:
        # Include time point as a feature for time-series prediction
        features = ['Age', 'Sex', 'BMI', 'Category', 'Carbs', 'Fat', 'Fiber', 'Protein',
                    'GI', 'Glycemic_Load', 'Activity_Level', 'Activity_Timing',
                    'Stress_Level', 'Sleep_Quality', 'Takes_Medication',
                    'Family_History', 'Time_Since_Last_Meal', 'Minutes_After_Meal']
    else:
        # Standard model predicting glucose spikes without time component
        features = ['Age', 'Sex', 'BMI', 'Category', 'Carbs', 'Fat', 'Fiber', 'Protein',
                    'GI', 'Glycemic_Load', 'Activity_Level', 'Activity_Timing',
                    'Stress_Level', 'Sleep_Quality', 'Takes_Medication',
                    'Family_History', 'Time_Since_Last_Meal']

    X = df[features]
    y = df['Glucose']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())

    # Train the final model on the full training set
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Cross-Validation RMSE: {cv_rmse:.2f} mg/dL")
    print(f"Test RMSE: {rmse:.2f} mg/dL")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mg/dL")
    print(f"R² Score: {r2:.2f}")

    results = {
        'model': model,
        'features': features,
        'label_encoders': {
            'sex': label_encoder_sex,
            'category': label_encoder_category,
            'activity': label_encoder_activity
        },
        'metrics': {
            'cv_rmse': cv_rmse,
            'test_rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

    return results

# 3. FEATURE IMPORTANCE ANALYSIS
# ==============================

def plot_feature_importance(model_results):
    """
    Visualize feature importance using permutation importance
    """
    model = model_results['model']
    X_test = model_results['X_test']
    y_test = model_results['y_test']
    features = model_results['features']

    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    # Create a DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': [features[i] for i in sorted_idx],
        'Importance': perm_importance.importances_mean[sorted_idx],
        'Std': perm_importance.importances_std[sorted_idx]
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance (Permutation-based)')
    plt.tight_layout()

    return importance_df

# 4. PERSONALIZED MODELS
# ======================

def build_personalized_models(df, num_individuals=5):
    """
    Build personalized models for a few select individuals
    """
    # Get a set of individuals to create personalized models for
    selected_ids = np.random.choice(df['Participant_ID'].unique(), size=num_individuals, replace=False)

    personalized_models = {}

    for pid in selected_ids:
        print(f"\nBuilding personalized model for Participant {pid}")

        # Get this participant's data
        individual_data = df[df['Participant_ID'] == pid]

        # Get general population data (excluding this individual)
        population_data = df[df['Participant_ID'] != pid]

        # Prepare individual data
        label_encoder_sex = LabelEncoder()
        individual_data['Sex'] = label_encoder_sex.fit_transform(individual_data['Sex'])

        label_encoder_category = LabelEncoder()
        individual_data['Category'] = label_encoder_category.fit_transform(individual_data['Category'])

        label_encoder_activity = LabelEncoder()
        individual_data['Activity_Timing'] = label_encoder_activity.fit_transform(individual_data['Activity_Timing'])

        # Features with time component for individual
        features = ['Carbs', 'Fat', 'Fiber', 'Protein', 'GI', 'Glycemic_Load',
                    'Activity_Level', 'Activity_Timing', 'Stress_Level',
                    'Sleep_Quality', 'Time_Since_Last_Meal', 'Minutes_After_Meal']

        X = individual_data[features]
        y = individual_data['Glucose']

        # Split data - use one day for testing, rest for training
        days = individual_data['Day'].unique()
        test_day = np.random.choice(days)

        train_data = individual_data[individual_data['Day'] != test_day]
        test_data = individual_data[individual_data['Day'] == test_day]

        X_train = train_data[features]
        y_train = train_data['Glucose']
        X_test = test_data[features]
        y_test = test_data['Glucose']

        # Train personalized model
        personal_model = RandomForestRegressor(n_estimators=50, random_state=42)
        personal_model.fit(X_train, y_train)

        # Train population model for this individual
        # Get a sample of population data similar in size to the individual's data
        pop_sample = population_data.sample(n=len(individual_data)*3, random_state=42)

        # Prepare population data
        # Need to fit_transform these separately as the values might differ from the individual's data
        pop_sex_encoder = LabelEncoder()
        pop_category_encoder = LabelEncoder()
        pop_activity_encoder = LabelEncoder()

        pop_sample['Sex'] = pop_sex_encoder.fit_transform(pop_sample['Sex'])
        pop_sample['Category'] = pop_category_encoder.fit_transform(pop_sample['Category'])
        pop_sample['Activity_Timing'] = pop_activity_encoder.fit_transform(pop_sample['Activity_Timing'])

        pop_X_train = pop_sample[features]
        pop_y_train = pop_sample['Glucose']

        pop_model = RandomForestRegressor(n_estimators=50, random_state=42)
        pop_model.fit(pop_X_train, pop_y_train)

        # Evaluate both models on the individual's test data
        personal_pred = personal_model.predict(X_test)
        pop_pred = pop_model.predict(X_test)

        personal_rmse = np.sqrt(mean_squared_error(y_test, personal_pred))
        pop_rmse = np.sqrt(mean_squared_error(y_test, pop_pred))

        improvement = (pop_rmse - personal_rmse) / pop_rmse * 100

        print(f"Personal Model RMSE: {personal_rmse:.2f} mg/dL")
        print(f"Population Model RMSE: {pop_rmse:.2f} mg/dL")
        print(f"Improvement: {improvement:.2f}%")

        # Store the personalized model
        personalized_models[pid] = {
            'model': personal_model,
            'features': features,
            'encoders': {
                'sex': label_encoder_sex,
                'category': label_encoder_category,
                'activity': label_encoder_activity
            },
            'metrics': {
                'personal_rmse': personal_rmse,
                'pop_rmse': pop_rmse,
                'improvement': improvement
            }
        }

    return personalized_models

# 5. TIME-SERIES VISUALIZATION
# ============================

def visualize_glucose_curves(df, model_results, num_examples=3):
    """
    Visualize actual vs. predicted glucose curves for a few examples
    """
    # Get unique participant-meal combinations
    participant_meals = df[['Participant_ID', 'Meal_Time']].drop_duplicates()

    # Randomly select a few examples
    selected_indices = np.random.choice(len(participant_meals), size=num_examples, replace=False)
    selected_examples = participant_meals.iloc[selected_indices].values

    # Prepare the plot
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4*num_examples))
    if num_examples == 1:
        axes = [axes]

    model = model_results['model']
    features = model_results['features']

    for i, (pid, meal_time) in enumerate(selected_examples):
        # Get data for this participant and meal
        meal_data = df[(df['Participant_ID'] == pid) & (df['Meal_Time'] == meal_time)]

        # Extract features and make predictions
        X = meal_data[features]
        actual_glucose = meal_data['Glucose'].values
        predicted_glucose = model.predict(X)

        # Plot actual vs. predicted
        time_points = meal_data['Minutes_After_Meal'].values

        axes[i].plot(time_points, actual_glucose, 'b-', label='Actual')
        axes[i].plot(time_points, predicted_glucose, 'r--', label='Predicted')

        # Add meal details
        meal_info = meal_data.iloc[0]
        category = meal_info['Category']
        if isinstance(category, (int, float)):  # If category is encoded
            category = list(model_results['label_encoders']['category'].classes_)[int(category)]

        title = (f"Participant {pid} ({category}) - "
                f"Carbs: {meal_info['Carbs']}g, GI: {meal_info['GI']}, "
                f"Activity: {meal_info['Activity_Level']}/10")

        axes[i].set_title(title)
        axes[i].set_xlabel('Minutes After Meal')
        axes[i].set_ylabel('Glucose (mg/dL)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Add threshold lines
        axes[i].axhline(y=70, color='g', linestyle='-', alpha=0.3)  # Lower normal
        axes[i].axhline(y=140, color='y', linestyle='-', alpha=0.3)  # Upper normal
        axes[i].axhline(y=180, color='r', linestyle='-', alpha=0.3)  # High

    plt.tight_layout()
    return fig

# 6. INFERENCE WITH IMPROVED MODEL
# ================================

def predict_glucose_response(model_results, new_meal_data, time_points=24):
    """
    Predict glucose response over time for a new meal
    """
    model = model_results['model']
    features = model_results['features']

    # Make a copy of the new_meal_data to avoid modifying the original
    meal_data = new_meal_data.copy()

    # Generate predictions for each time point
    predictions = []
    time_series = []

    # Check if we're using the time-based model
    if 'Minutes_After_Meal' in features:
        # Create a DataFrame for each time point (every 5 minutes for 2 hours)
        for i in range(time_points):
            minutes = i * 5
            time_series.append(minutes)

            # Create a copy of the meal data for this time point
            time_data = meal_data.copy()
            time_data['Minutes_After_Meal'] = minutes

            # Convert to DataFrame with all features
            data_point = pd.DataFrame([time_data])

            # Ensure all features are present
            for feature in features:
                if feature not in data_point.columns:
                    data_point[feature] = 0  # Default value

            # Select only the features used by the model
            X_new = data_point[features]

            # Predict glucose
            glucose = model.predict(X_new)[0]
            predictions.append(glucose)
    else:
        # For non-time-based model, just make one prediction
        data_point = pd.DataFrame([meal_data])
        # Ensure all features are present
        for feature in features:
            if feature not in data_point.columns:
                data_point[feature] = 0

        X_new = data_point[features]
        glucose = model.predict(X_new)[0]
        predictions.append(glucose)
        time_series.append(0)


    # Check if we're using the time-based model
    if 'Minutes_After_Meal' in features:
        # Create a DataFrame for each time point (every 5 minutes for 2 hours)
        for i in range(time_points):
            minutes = i * 5
            time_series.append(minutes)

            # Create a copy of the meal data for this time point
            time_data = new_meal_data.copy()
            time_data['Minutes_After_Meal'] = minutes

            # Convert to DataFrame with all features
            data_point = pd.DataFrame([time_data])

            # Ensure all features are present
            for feature in features:
                if feature not in data_point.columns:
                    data_point[feature] = 0  # Default value

            # Select only the features used by the model
            X_new = data_point[features]

            # Predict glucose
            glucose = model.predict(X_new)[0]
            predictions.append(glucose)
    else:
        # For non-time-based model, just make one prediction
        data_point = pd.DataFrame([new_meal_data])
        # Ensure all features are present
        for feature in features:
            if feature not in data_point.columns:
                data_point[feature] = 0

        X_new = data_point[features]
        glucose = model.predict(X_new)[0]
        predictions.append(glucose)
        time_series.append(0)

    # Categorize glucose levels
    def categorize_glucose(value):
        if value < 70:
            return 'Low'
        elif 70 <= value <= 139:
            return 'Normal'
        elif 140 <= value <= 179:
            return 'Elevated'
        else:
            return 'High'

    max_glucose = max(predictions)
    max_glucose_time = time_series[predictions.index(max_glucose)]
    max_glucose_category = categorize_glucose(max_glucose)

    # Return predicted glucose levels and additional info
    results = {
        'time_series': time_series,
        'glucose_predictions': predictions,
        'max_glucose': max_glucose,
        'max_glucose_time': max_glucose_time,
        'glucose_category': max_glucose_category
    }

    return results

def plot_predicted_glucose(prediction_results, meal_info=None):
    """
    Plot predicted glucose response over time
    """
    time_series = prediction_results['time_series']
    glucose_predictions = prediction_results['glucose_predictions']
    max_glucose = prediction_results['max_glucose']
    max_glucose_time = prediction_results['max_glucose_time']

    plt.figure(figsize=(10, 6))
    plt.plot(time_series, glucose_predictions, 'b-', linewidth=2)
    plt.plot(max_glucose_time, max_glucose, 'ro', markersize=8)

    # Add threshold lines
    plt.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Lower Normal (70 mg/dL)')
    plt.axhline(y=140, color='y', linestyle='--', alpha=0.5, label='Upper Normal (140 mg/dL)')
    plt.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='High (180 mg/dL)')

    # Annotation for peak
    plt.annotate(f'Peak: {max_glucose:.1f} mg/dL at {max_glucose_time} min',
                xy=(max_glucose_time, max_glucose),
                xytext=(max_glucose_time+10, max_glucose+10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    plt.title('Predicted Glucose Response After Meal')
    plt.xlabel('Minutes After Meal')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.grid(True, alpha=0.3)

    if meal_info:
        info_text = '\n'.join([f"{k}: {v}" for k, v in meal_info.items() if k not in ['Minutes_After_Meal']])
        plt.figtext(0.02, 0.02, f"Meal Information:\n{info_text}", fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()

    return plt

# 7. MAIN EXECUTION FLOW
# ======================

def run_complete_workflow(num_participants=30, days_per_participant=3):
    """
    Run the complete workflow from data simulation to model evaluation
    """
    print("Step 1: Generating simulated glucose data...")
    df = generate_simulated_data(num_participants, days_per_participant)
    print(f"Generated {len(df)} data points for {num_participants} participants over {days_per_participant} days each.")

    print("\nStep 2: Building and evaluating standard model...")
    standard_model_results = build_and_evaluate_model(df, time_based=False)

    print("\nStep 3: Building and evaluating time-series model...")
    time_model_results = build_and_evaluate_model(df, time_based=True)

    print("\nStep 4: Analyzing feature importance...")
    importance_df = plot_feature_importance(time_model_results)
    print(importance_df.head(10))

    print("\nStep 5: Building personalized models...")
    personalized_models = build_personalized_models(df, num_individuals=3)

    print("\nStep 6: Visualizing glucose curves...")
    fig = visualize_glucose_curves(df, time_model_results, num_examples=3)

    print("\nStep 7: Making predictions with new meal data...")
    # Example meal for testing prediction - using numeric values directly to avoid encoding issues
    new_meal = {
        'Age': 50,
        'Sex': 1,  # Male (assumes encoding where Male=1, Female=0)
        'BMI': 28.0,
        'Category': 1,  # Pre-diabetic (assumes encoding where Normal=0, Pre-diabetic=1, Diabetic=2)
        'Carbs': 70,
        'Fat': 25,
        'Fiber': 7,
        'Protein': 30,
        'GI': 65,
        'Glycemic_Load': 45.5,  # (65 * 70) / 100
        'Activity_Level': 3.0,
        'Activity_Timing': 1,  # 'after' (assumes encoding where before=0, after=1, none=2)
        'Stress_Level': 5.0,
        'Sleep_Quality': 7.0,
        'Takes_Medication': 0,
        'Family_History': 1,
        'Time_Since_Last_Meal': 4.5
    }

    prediction = predict_glucose_response(time_model_results, new_meal)
    plot = plot_predicted_glucose(prediction, new_meal)

    print(f"\nPredicted maximum glucose level: {prediction['max_glucose']:.2f} mg/dL ({prediction['glucose_category']})")
    print(f"Peak occurs at {prediction['max_glucose_time']} minutes after the meal")

    # Save models
    joblib.dump(standard_model_results, 'standard_glucose_model.pkl')
    joblib.dump(time_model_results, 'time_series_glucose_model.pkl')

    print("\nWorkflow complete! Models have been saved.")

    return {
        'data': df,
        'standard_model': standard_model_results,
        'time_model': time_model_results,
        'personalized_models': personalized_models,
        'feature_importance': importance_df,
        'example_prediction': prediction
    }

# Run the entire workflow if executed as a script
if __name__ == "__main__":
    results = run_complete_workflow(num_participants=30, days_per_participant=3)

