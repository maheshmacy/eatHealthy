# EATSMART-AI: Personalized Glycemic Index Tracking

This is a modular implementation of the EATSMART-AI Streamlit frontend, with fixed image loading and Plotly visualization issues.

## Project Structure

The application has been split into several modules for better organization and easier maintenance:

1. **app.py** - Main application entry point
2. **api_client.py** - API client for interacting with the backend
3. **helpers.py** - Helper functions for data processing and visualization
4. **ui_components.py** - UI components for rendering the application pages

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install streamlit pandas numpy plotly pillow requests
```

3. Place all the Python files in the same directory
4. Run the application:

```bash
streamlit run app.py
```

## Integration with Backend

This frontend is designed to work with the Flask backend that provides the following API endpoints:

- `/users` - User management
- `/food/analyze-food` - Food analysis
- `/meals/{user_id}` - Meal history
- `/meals/{user_id}/{meal_id}` - Meal details
- `/meals/{user_id}/stats` - Meal statistics
- `/meals/{user_id}/{meal_id}/feedback` - Meal feedback

Make sure the backend server is running on `http://localhost:5000` or update the `API_ENDPOINT` variable in `api_client.py` to point to the correct URL.

## Usage

1. **User Registration**: Create a new user profile or log in with an existing user ID
2. **Food Analysis**: Upload food images for analysis
3. **Meal History**: View and filter your meal history
4. **Statistics**: View statistics and insights about your glucose responses

## Production Deployment Notes

For a production environment, consider the following improvements:

1. **Image Handling**: Implement a proper API endpoint in the backend to serve images, rather than relying on file paths
2. **Authentication**: Add proper authentication and session management
3. **Error Handling**: Enhance error handling and user feedback
4. **Caching**: Implement caching for better performance
5. **Responsive Design**: Further enhance the UI for various screen sizes

## Troubleshooting

- If the application cannot connect to the backend, check that the Flask server is running
- If images are not displaying correctly, ensure the placeholder image generation is working
- If Plotly charts are not rendering, check for any Plotly property errors in the console