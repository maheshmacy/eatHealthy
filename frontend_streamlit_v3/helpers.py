import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_placeholder_image_base64(width=150, height=100, text="Meal Image"):
    """Create a placeholder image for meals that fail to load."""
    # Create a new image with a light background
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        # Fall back to default font
        font = None
    
    # Draw text
    text_width, text_height = 0, 0
    try:
        if font:
            # Draw text in center of image
            position = ((width - 60) // 2, (height - 10) // 2)
            draw.text(position, text, fill=(100, 100, 100), font=font)
    except Exception as e:
        logger.error(f"Error creating placeholder text: {e}")
        pass
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def display_meal_image(meal, width=150):
    """
    Display meal image using the API endpoint approach for production.
    Falls back to placeholder if image can't be loaded.
    """
    try:
        image_path = meal.get('image_path')
        
        if image_path and isinstance(image_path, str):
            # Extract the filename from the full path
            filename = os.path.basename(image_path)
            
            # Get the image URL from API endpoint
            from api_client import get_image_url
            image_url = get_image_url(filename)
            
            # Log the URL being requested (for debugging)
            logger.info(f"Loading meal image from: {image_url}")
            
            # Try to display the image from the API endpoint
            try:
                st.image(image_url, width=width)
                return  # Return early if successful
            except Exception as e:
                logger.error(f"Failed to load image from {image_url}: {e}")
                # Fall through to placeholder
        
        # Fall back to placeholder with food names if available
        food_names = ""
        if 'foods' in meal and isinstance(meal['foods'], list):
            food_names = ", ".join(meal.get('foods', ['Meal']))
        elif 'food_items' in meal and isinstance(meal['food_items'], list):
            food_names = ", ".join([item.get('food_name', 'Food') for item in meal['food_items']])
        else:
            food_names = "Meal"
        
        # Create placeholder image and display it
        img_base64 = create_placeholder_image_base64(text=food_names)
        st.image(f"data:image/png;base64,{img_base64}", width=width)
    
    except Exception as e:
        logger.error(f"Error in display_meal_image: {e}")
        # Final fallback - show a simple colored box if all else fails
        st.markdown(
            f"""
            <div style="
                width: {width}px;
                height: {width}px;
                background-color: #f0f0f0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                font-size: 12px;
                text-align: center;
                border-radius: 5px;
            ">
                Food Image
            </div>
            """, 
            unsafe_allow_html=True
        )

def create_glucose_graph(glucose_prediction):
    """Create interactive glucose prediction graph"""
    # Check if we have valid prediction data
    if not glucose_prediction:
        return None
    
    # Check if we have time series data specifically
    has_time_series = 'time_series' in glucose_prediction and glucose_prediction['time_series']
    
    # If no time series but we have predicted_glucose, create a simple single point visualization
    if not has_time_series and 'predicted_glucose' in glucose_prediction:
        # Create a simple point plot showing just the predicted glucose value
        fig = go.Figure()
        
        # Add single point marker
        predicted_glucose = glucose_prediction['predicted_glucose']
        fig.add_trace(go.Scatter(
            x=[0],  # Just showing the prediction at the start
            y=[predicted_glucose],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Predicted Glucose',
            hovertemplate='Predicted Peak Glucose: %{y:.1f} mg/dL'
        ))
        
        # Add category ranges as horizontal bands
        fig.add_shape(type="rect", x0=-10, x1=10, y0=180, y1=400,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=140, y1=180,
                      fillcolor="rgba(255,165,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=70, y1=140,
                      fillcolor="rgba(0,128,0,0.1)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=-10, x1=10, y0=0, y1=70,
                      fillcolor="rgba(255,0,0,0.1)", line=dict(width=0), layer="below")
        
        # Add category text labels
        fig.add_annotation(x=8, y=190, text="High", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=8, y=160, text="Elevated", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=8, y=105, text="Normal", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=8, y=35, text="Low", showarrow=False, font=dict(color="red"))
        
        # Update layout
        fig.update_layout(
            title='Predicted Peak Glucose',
            yaxis_title='Blood Glucose (mg/dL)',
            xaxis_visible=False,
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Set y-axis range to show all categories
        fig.update_yaxes(range=[0, 250])
        
        return fig
    
    # If we have proper time series data, create the full graph
    elif has_time_series:
        time_series = glucose_prediction['time_series']
        
        # If the time series is a list of dictionaries
        if isinstance(time_series, list) and len(time_series) > 0 and isinstance(time_series[0], dict):
            df = pd.DataFrame(time_series)
            x = df.get('minute', [])
            y = df.get('glucose', [])
        else:
            # If it's just a list of values
            x = list(range(0, len(time_series) * 5, 5))
            y = time_series
        
        # If we still don't have valid x,y data, return None
        if not x or not y:
            return None
            
        # Create glucose range areas
        fig = go.Figure()
        
        # Add glucose ranges (colored areas)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[180] * len(x) + [400] * len(x),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='High',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[140] * len(x) + [180] * len(x),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.1)',
            line=dict(color='rgba(255, 165, 0, 0)'),
            name='Elevated',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[70] * len(x) + [140] * len(x),
            fill='toself',
            fillcolor='rgba(0, 128, 0, 0.1)',
            line=dict(color='rgba(0, 128, 0, 0)'),
            name='Normal',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[0] * len(x) + [70] * len(x),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Low',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add predicted glucose curve
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            name='Predicted Glucose',
            hovertemplate='Time: %{x} min<br>Glucose: %{y:.1f} mg/dL'
        ))
        
        # Add peak marker if available
        max_glucose_time = glucose_prediction.get('max_glucose_time', 0)
        if max_glucose_time > 0 and max_glucose_time in x:
            max_index = x.index(max_glucose_time)
            max_glucose = y[max_index] if max_index < len(y) else None
            
            if max_glucose is not None:
                fig.add_trace(go.Scatter(
                    x=[max_glucose_time],
                    y=[max_glucose],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='star'),
                    name='Peak Glucose',
                    hovertemplate='Peak Time: %{x} min<br>Peak Glucose: %{y:.1f} mg/dL'
                ))
        
        # Update layout
        fig.update_layout(
            title='Predicted Glucose Response',
            xaxis_title='Time (minutes after meal)',
            yaxis_title='Blood Glucose (mg/dL)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Add range lines
        max_x = max(x)
        fig.add_shape(type="line", x0=0, y0=70, x1=max_x, y1=70, 
                    line=dict(color="green", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=140, x1=max_x, y1=140, 
                    line=dict(color="orange", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=180, x1=max_x, y1=180, 
                    line=dict(color="red", width=1, dash="dash"))
        
        return fig
    
    # If neither condition is met, return None
    return None

def get_nutrient_chart(nutrients):
    """Create a bar chart for nutritional information"""
    if not nutrients:
        return None
    
    # Create data for bar chart
    nutrient_names = []
    nutrient_values = []
    
    for key, value in nutrients.items():
        if key not in ['calories']:  # Excluding calories as it's typically much higher than others
            nutrient_names.append(key.capitalize())
            nutrient_values.append(value)
    
    # Create bar chart
    fig = px.bar(
        x=nutrient_names,
        y=nutrient_values,
        title="Nutritional Content (g)",
        labels={"x": "Nutrient", "y": "Amount (g)"},
        color=nutrient_names,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Grams",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def format_timestamp(timestamp):
    """Format ISO timestamp to a readable date/time"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except:
        return timestamp

def format_date(date_str):
    """Format date string to readable format"""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%b %d, %Y")
    except:
        return date_str

# CSS for styling
CUSTOM_CSS = """
<style>
    /* Card styling for meal history */
    div.stContainer {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 10px;
        margin-bottom: 15px !important;
        background-color: #ffffff;
        transition: all 0.3s ease;
    }

    div.stContainer:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #3e8e41;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Custom glucose category styling */
    .glucose-normal {
        color: #4CAF50;
        font-weight: bold;
    }

    .glucose-elevated {
        color: #FF9800;
        font-weight: bold;
    }

    .glucose-high {
        color: #F44336;
        font-weight: bold;
    }

    .glucose-low {
        color: #F44336;
        font-weight: bold;
    }

    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 500;
    }

    /* Improve headers */
    h2, h3 {
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #4CAF50;
    }

    /* Improve form styling */
    div[data-testid="stForm"] {
        border-radius: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        margin-bottom: 20px;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    div[data-testid="stMetricValue"] {
        font-weight: bold;
    }
    
    /* Image container styling */
    .image-container {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        background-color: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
"""