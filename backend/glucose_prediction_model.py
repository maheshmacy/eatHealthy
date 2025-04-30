import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class GlucoseResponsePredictor:
    """
    A machine learning model that predicts personalized glucose responses
    based on individual characteristics and meal composition
    """
    
    def __init__(self):
        # Model parameters
        self.feature_stats = {}
        self.model = None
        self.scaler = None
        self.feature_importances = None
        self.target_stats = {}
        self.performance_metrics = {}
        
        # Feature lists
        self.base_features = [
            'person_clinic_bmi',      # BMI
            'person_md_age',          # Age
            'person_md_sex',          # Gender (binary: M=0, F=1)
            'meal_calories',          # Total calories
            'meal_carbohydrate',      # Carbs content
            'meal_fat',               # Fat content
            'meal_protein',           # Protein content
            'meal_fibre',             # Fiber content
            'meal_sugar'              # Sugar content
        ]
        
        self.engineered_features = [
            'carb_to_fiber_ratio',    # Ratio of carbs to fiber
            'fat_to_carb_ratio',      # Ratio of fat to carbs
            'sugar_percentage',       # Percentage of carbs that are sugar
            'carb_percentage',        # Percentage of macros that are carbs
            'fat_percentage',         # Percentage of macros that are fat
            'protein_percentage'      # Percentage of macros that are protein
        ]
        
        self.all_features = self.base_features + self.engineered_features
        self.target_variable = 'meal_iauc'  # Area under insulin curve
    
    def load_and_process_data(self, csv_path):
        """
        Load and preprocess data from a CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the data
            
        Returns:
        --------
        processed_data : pandas.DataFrame
            Processed data ready for training
        """
        print("Loading and processing data...")
        
        # Load data
        try:
            data = pd.read_csv(csv_path)
            print(f"Dataset contains {len(data)} entries")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Check for missing values
        missing_counts = {feature: data[feature].isna().sum() 
                         for feature in self.base_features + [self.target_variable]
                         if data[feature].isna().sum() > 0}
        
        if missing_counts:
            print("Missing value counts:", missing_counts)
        
        # Filter out rows with missing values
        complete_data = data.dropna(subset=self.base_features + [self.target_variable])
        print(f"After removing rows with missing values, {len(complete_data)} entries remain")
        
        # Convert categorical features
        complete_data = complete_data.copy()
        complete_data['person_md_sex'] = complete_data['person_md_sex'].map({'M': 0, 'F': 1})
        
        # Calculate feature statistics
        self.feature_stats = {}
        for feature in self.base_features:
            self.feature_stats[feature] = {
                'min': complete_data[feature].min(),
                'max': complete_data[feature].max(),
                'mean': complete_data[feature].mean(),
                'std': complete_data[feature].std()
            }
        
        # Calculate target statistics
        self.target_stats = {
            'min': complete_data[self.target_variable].min(),
            'max': complete_data[self.target_variable].max(),
            'mean': complete_data[self.target_variable].mean(),
            'std': complete_data[self.target_variable].std()
        }
        
        print("Target statistics:", self.target_stats)
        
        # Engineer additional features
        # Carb-to-fiber ratio
        complete_data['carb_to_fiber_ratio'] = complete_data['meal_carbohydrate'] / (complete_data['meal_fibre'] + 0.1)
        
        # Fat-to-carb ratio
        complete_data['fat_to_carb_ratio'] = complete_data['meal_fat'] / (complete_data['meal_carbohydrate'] + 0.1)
        
        # Sugar percentage in carbs
        complete_data['sugar_percentage'] = (complete_data['meal_sugar'] / (complete_data['meal_carbohydrate'] + 0.1)) * 100
        
        # Total macronutrient content
        complete_data['total_macros'] = (complete_data['meal_carbohydrate'] + 
                                         complete_data['meal_fat'] + 
                                         complete_data['meal_protein'])
        
        # Carb percentage of total macros
        complete_data['carb_percentage'] = (complete_data['meal_carbohydrate'] / 
                                           (complete_data['total_macros'] + 0.1)) * 100
        
        # Fat percentage of total macros
        complete_data['fat_percentage'] = (complete_data['meal_fat'] / 
                                          (complete_data['total_macros'] + 0.1)) * 100
        
        # Protein percentage of total macros
        complete_data['protein_percentage'] = (complete_data['meal_protein'] / 
                                              (complete_data['total_macros'] + 0.1)) * 100
        
        # Calculate statistics for engineered features
        for feature in self.engineered_features:
            self.feature_stats[feature] = {
                'min': complete_data[feature].min(),
                'max': complete_data[feature].max(),
                'mean': complete_data[feature].mean(),
                'std': complete_data[feature].std()
            }
        
        return complete_data
    
    def train_model(self, csv_path, options=None):
        """
        Train the model on the provided data
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the data
        options : dict, optional
            Training options
            
        Returns:
        --------
        training_results : dict
            Training results and model information
        """
        if options is None:
            options = {}
        
        try:
            # Process data
            processed_data = self.load_and_process_data(csv_path)
            if processed_data is None:
                return None
            
            # Split data into features and target
            X = processed_data[self.all_features]
            y = processed_data[self.target_variable]
            
            # Split into training and testing sets
            train_ratio = options.get('train_ratio', 0.8)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-train_ratio, random_state=42
            )
            
            print(f"Training set: {len(X_train)} entries")
            print(f"Testing set: {len(X_test)} entries")
            
            # Scale features (important for many algorithms)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train the random forest model
            print("Training random forest model...")
            model_params = {
                'n_estimators': options.get('n_estimators', 100),
                'max_depth': options.get('max_depth', 10),
                'min_samples_split': options.get('min_samples_split', 2),
                'random_state': 42
            }
            
            self.model = RandomForestRegressor(**model_params)
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate performance metrics
            self.performance_metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            print("Model performance metrics:", self.performance_metrics)
            
            # Calculate feature importance
            importance = self.model.feature_importances_
            
            # Create feature importance dict
            self.feature_importances = {
                feature: importance for feature, importance in zip(self.all_features, importance)
            }
            
            # Sort features by importance
            sorted_features = sorted(
                self.feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            print("Feature importance (top 5):")
            for feature, importance in sorted_features[:5]:
                print(f"{feature}: {importance:.4f}")
            
            return {
                'model_info': {
                    'n_estimators': self.model.n_estimators,
                    'performance_metrics': self.performance_metrics,
                    'top_features': sorted_features[:5]
                }
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_glucose_response(self, person, meal):
        """
        Predict glucose response for a new person and meal
        
        Parameters:
        -----------
        person : dict
            Person's characteristics (bmi, age, sex)
        meal : dict
            Meal's composition (calories, carbs, fat, protein, fiber, sugar)
            
        Returns:
        --------
        prediction : dict
            Prediction and explanation
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Extract person and meal features
        bmi = person.get('bmi')
        age = person.get('age')
        sex = 0 if person.get('sex') == 'M' else 1
        
        calories = meal.get('calories')
        carbs = meal.get('carbs')
        fat = meal.get('fat')
        protein = meal.get('protein')
        fiber = meal.get('fiber')
        sugar = meal.get('sugar')
        
        # Create a new data point with basic features
        new_data = {
            'person_clinic_bmi': bmi,
            'person_md_age': age,
            'person_md_sex': sex,
            'meal_calories': calories,
            'meal_carbohydrate': carbs,
            'meal_fat': fat,
            'meal_protein': protein,
            'meal_fibre': fiber,
            'meal_sugar': sugar
        }
        
        # Engineer additional features
        total_macros = carbs + fat + protein
        
        new_data['carb_to_fiber_ratio'] = carbs / (fiber + 0.1)
        new_data['fat_to_carb_ratio'] = fat / (carbs + 0.1)
        new_data['sugar_percentage'] = (sugar / (carbs + 0.1)) * 100
        new_data['carb_percentage'] = (carbs / (total_macros + 0.1)) * 100
        new_data['fat_percentage'] = (fat / (total_macros + 0.1)) * 100
        new_data['protein_percentage'] = (protein / (total_macros + 0.1)) * 100
        
        # Create input array for prediction
        input_data = np.array([[new_data[feature] for feature in self.all_features]])
        
        # Scale the input data
        input_data_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_data_scaled)[0]
        
        # Determine risk level based on prediction
        if prediction <= self.target_stats['mean'] - 0.5 * self.target_stats['std']:
            risk_level = "Low"
        elif prediction <= self.target_stats['mean'] + 0.5 * self.target_stats['std']:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate percentile
        z_score = (prediction - self.target_stats['mean']) / self.target_stats['std']
        from scipy.stats import norm
        percentile = norm.cdf(z_score) * 100
        
        # Generate explanation based on feature importance
        sorted_importance = sorted(
            [(f, self.model.feature_importances_[i]) 
             for i, f in enumerate(self.all_features)],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [f[0] for f in sorted_importance[:3]]
        explanation = f"This prediction is based primarily on {', '.join(top_features)}."
        
        return {
            'iauc_prediction': prediction,
            'risk_level': risk_level,
            'percentile': round(percentile),
            'explanation': explanation,
            # Add additional details that might be useful for the app
            'mean_population_response': self.target_stats['mean'],
            'std_population_response': self.target_stats['std']
        }
    
    def save_model(self, filename):
        """
        Save the trained model to a file
        
        Parameters:
        -----------
        filename : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'target_stats': self.target_stats,
            'feature_importances': self.feature_importances,
            'performance_metrics': self.performance_metrics,
            'all_features': self.all_features,
            'base_features': self.base_features,
            'engineered_features': self.engineered_features
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        filename : str
            Path to the saved model
        """
        try:
            model_data = joblib.load(filename)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_stats = model_data['feature_stats']
            self.target_stats = model_data['target_stats']
            self.feature_importances = model_data.get('feature_importances')
            self.performance_metrics = model_data.get('performance_metrics')
            self.all_features = model_data.get('all_features', self.all_features)
            self.base_features = model_data.get('base_features', self.base_features)
            self.engineered_features = model_data.get('engineered_features', self.engineered_features)
            
            print(f"Model loaded successfully from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Example usage function
def demonstrate_usage():
    """
    Demonstrate how to use the GlucoseResponsePredictor
    """
    # Create the predictor
    predictor = GlucoseResponsePredictor()
    
    try:
        # Train the model
        print("Training the model...")
        training_result = predictor.train_model('sample_meals.csv', {
            'n_estimators': 100,
            'max_depth': 10
        })
        
        print("Training completed:", training_result)
        
        # Example predictions
        test_cases = [
            {
                'person': {'bmi': 25.0, 'age': 35, 'sex': 'M'},
                'meal': {'calories': 500, 'carbs': 60, 'fat': 15, 'protein': 25, 'fiber': 5, 'sugar': 20},
                'description': "Average BMI male with balanced meal"
            },
            {
                'person': {'bmi': 32.0, 'age': 55, 'sex': 'F'},
                'meal': {'calories': 500, 'carbs': 60, 'fat': 15, 'protein': 25, 'fiber': 5, 'sugar': 20},
                'description': "Higher BMI older female with same balanced meal"
            },
            {
                'person': {'bmi': 25.0, 'age': 35, 'sex': 'M'},
                'meal': {'calories': 500, 'carbs': 20, 'fat': 35, 'protein': 30, 'fiber': 8, 'sugar': 5},
                'description': "Average BMI male with low-carb meal"
            },
            {
                'person': {'bmi': 25.0, 'age': 35, 'sex': 'M'},
                'meal': {'calories': 500, 'carbs': 100, 'fat': 5, 'protein': 15, 'fiber': 3, 'sugar': 40},
                'description': "Average BMI male with high-carb meal"
            }
        ]
        
        print("\nExample predictions:")
        for test_case in test_cases:
            prediction = predictor.predict_glucose_response(test_case['person'], test_case['meal'])
            print(f"\n{test_case['description']}:")
            print(f"- Predicted IAUC: {prediction['iauc_prediction']:.2f}")
            print(f"- Risk Level: {prediction['risk_level']}")
            print(f"- Percentile: {prediction['percentile']}%")
            print(f"- Explanation: {prediction['explanation']}")
        
        # Save and load model
        predictor.save_model("glucose_model.joblib")
        
        # Load model in a new predictor
        new_predictor = GlucoseResponsePredictor()
        new_predictor.load_model("glucose_model.joblib")
        
        # Verify loaded model works
        verification_prediction = new_predictor.predict_glucose_response(
            test_cases[0]['person'],
            test_cases[0]['meal']
        )
        
        print("\nVerification prediction after loading saved model:")
        print(f"- Predicted IAUC: {verification_prediction['iauc_prediction']:.2f}")
        print(f"- Risk Level: {verification_prediction['risk_level']}")
        
        return "Model demonstration completed successfully!"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error in demonstration: {e}"

if __name__ == "__main__":
    demonstrate_usage()
