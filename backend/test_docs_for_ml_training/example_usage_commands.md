
# Generate sample data and test with a specific model file
python test_model.py --model-path ./models/standard_glucose_model.pkl --generate-samples

# Test with a user-specific model
python test_model.py --user-id 12345abc-def6-789 --generate-samples

# Use custom meal data and user profile
python test_model.py --user-id 12345abc-def6-789 --meal-json my_meals.json --user-profile-json my_profile.json

# Generate sample data, save to custom output file
python test_model.py --model-path ./models/standard_glucose_model.pkl --generate-samples --output test_results.json
