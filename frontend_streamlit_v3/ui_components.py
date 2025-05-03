import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import random
import base64
from io import BytesIO

def display_header(title: str) -> None:
    """
    Display a consistent header with title.
    
    Args:
        title: The page title
    """
    st.title(title)
    st.markdown("---")

def create_sidebar() -> None:
    """
    Create the sidebar navigation menu.
    """
    st.sidebar.title("Nutrition Tracker")
    
    # Display user info if available
    if st.session_state.user_profile:
        if "name" in st.session_state.user_profile:
            st.sidebar.write(f"Welcome, {st.session_state.user_profile['name']}!")
        st.sidebar.markdown("---")
    
    # Navigation options
    navigation_options = ["Dashboard", "Food Search", "Meal Planner", "Profile", "Logout"]
    
    for option in navigation_options:
        if st.sidebar.button(option):
            st.session_state.current_tab = option
            # Clear any temporary states when changing tabs
            if option != "Meal Planner":
                if "selected_food" in st.session_state:
                    del st.session_state.selected_food
            if option != "Food Search":
                if "food_details" in st.session_state:
                    del st.session_state.food_details
            st.experimental_rerun()
    
    # Display info about the app
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This nutrition tracking app helps you monitor your food intake "
        "and provides personalized nutrition recommendations based on your "
        "profile and goals."
    )
    
    # Display the current date/time
    from datetime import datetime
    now = datetime.now()
    st.sidebar.markdown(f"Current date: {now.strftime('%Y-%m-%d')}")

def render_nutrition_chart(nutrition_data: Dict[str, float], chart_type: str = "pie") -> None:
    """
    Render a chart showing nutritional composition.
    
    Args:
        nutrition_data: Dictionary with nutrition values
        chart_type: Type of chart ('pie' or 'bar')
    """
    if chart_type == "pie":
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Extract macronutrient data
        labels = ['Protein', 'Carbs', 'Fat']
        sizes = [
            nutrition_data.get('protein', 0),
            nutrition_data.get('carbohydrates', 0),
            nutrition_data.get('fat', 0)
        ]
        
        # Convert to percentages
        total = sum(sizes)
        if total > 0:
            sizes = [s / total * 100 for s in sizes]
        
        colors = ['#4CAF50', '#FFC107', '#F44336']
        explode = (0.1, 0, 0)  # explode protein slice
        
        ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            shadow=True, 
            startangle=90
        )
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        st.pyplot(fig)
    
    elif chart_type == "bar":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract macronutrient data
        nutrients = ['Protein', 'Carbs', 'Fat']
        values = [
            nutrition_data.get('protein', 0),
            nutrition_data.get('carbohydrates', 0),
            nutrition_data.get('fat', 0)
        ]
        
        # Calculate calories from each macronutrient
        calorie_values = [
            values[0] * 4,  # 4 calories per gram of protein
            values[1] * 4,  # 4 calories per gram of carbs
            values[2] * 9   # 9 calories per gram of fat
        ]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Nutrient': nutrients,
            'Grams': values,
            'Calories': calorie_values
        })
        
        # Plot
        bar_plot = sns.barplot(x='Nutrient', y='Grams', data=df, ax=ax)
        
        # Add calorie labels on top of bars
        for i, p in enumerate(bar_plot.patches):
            bar_plot.annotate(
                f"{calorie_values[i]:.0f} kcal",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        ax.set_title('Macronutrient Distribution (in grams)')
        ax.set_ylabel('Grams')
        st.pyplot(fig)

def display_food_card(food: Dict[str, Any], on_add=None, on_view=None) -> None:
    """
    Display a food item as a card.
    
    Args:
        food: Food item data
        on_add: Callback function when "Add to Meal" is clicked
        on_view: Callback function when "View Details" is clicked
    """
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(food.get('name', 'Unknown Food'))
        st.write(f"**Calories:** {food.get('calories', 0)} kcal")
        st.write(f"**Protein:** {food.get('protein', 0)}g | **Carbs:** {food.get('carbohydrates', 0)}g | **Fat:** {food.get('fat', 0)}g")
        
        # Display tags if available
        if 'tags' in food and food['tags']:
            tags_html = ' '.join([f'<span style="background-color: #e0e0e0; padding: 2px 8px; border-radius: 10px; margin-right: 5px;">{tag}</span>' for tag in food['tags']])
            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
    
    with col2:
        if on_add:
            if st.button("Add to Meal", key=f"add_{food.get('id', random.randint(1, 10000))}"):
                on_add(food)
        
        if on_view:
            if st.button("View Details", key=f"view_{food.get('id', random.randint(1, 10000))}"):
                on_view(food)
    
    st.markdown("---")

def get_image_as_base64(image_path: str) -> str:
    """
    Convert an image to base64 string for embedding in HTML.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def display_meal_summary(meal_data: Dict[str, Any]) -> None:
    """
    Display a summary of a meal with nutritional information.
    
    Args:
        meal_data: Meal data including name, time, and nutrition info
    """
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(meal_data.get('name', 'Meal'))
        st.write(f"**Date:** {meal_data.get('date', 'Unknown')}")
        st.write(f"**Type:** {meal_data.get('type', 'Unknown')}")
        
        if 'items' in meal_data and meal_data['items']:
            st.write("**Items:**")
            for item in meal_data['items']:
                st.write(f"- {item.get('name', 'Unknown Food')} ({item.get('serving_size', 1)} servings)")
    
    with col2:
        if 'nutrition' in meal_data:
            nutrition = meal_data['nutrition']
            st.write("**Nutrition Summary:**")
            st.write(f"Calories: {nutrition.get('calories', 0)} kcal")
            st.write(f"Protein: {nutrition.get('protein', 0)}g")
            st.write(f"Carbs: {nutrition.get('carbohydrates', 0)}g")
            st.write(f"Fat: {nutrition.get('fat', 0)}g")
            
            # Mini pie chart for macronutrient distribution
            fig, ax = plt.subplots(figsize=(3, 3))
            labels = ['Protein', 'Carbs', 'Fat']
            sizes = [
                nutrition.get('protein', 0),
                nutrition.get('carbohydrates', 0),
                nutrition.get('fat', 0)
            ]
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            # Convert to percentages for the pie chart
            total = sum(sizes)
            if total > 0:
                sizes = [s / total * 100 for s in sizes]
                
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
    
    st.markdown("---")

def progress_bar_with_target(
    value: float,
    target: float,
    label: str,
    color: str = "blue",
    show_percentage: bool = True
) -> None:
    """
    Display a progress bar with a target indicator.
    
    Args:
        value: Current value
        target: Target value
        label: Label for the progress bar
        color: Color of the progress bar
        show_percentage: Whether to show percentage of target
    """
    # Calculate percentage of target
    percentage = min(100, (value / target) * 100) if target > 0 else 0
    
    # Create the label with current/target values
    if show_percentage:
        display_label = f"{label}: {value:.1f}/{target:.1f} ({percentage:.1f}%)"
    else:
        display_label = f"{label}: {value:.1f}/{target:.1f}"
    
    # Create CSS for customizing the progress bar
    progress_css = f"""
    <style>
        .custom-progress-container {{
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .custom-progress-bar {{
            height: 24px;
            background-color: {color};
            border-radius: 5px;
            width: {percentage}%;
            position: relative;
        }}
        .custom-progress-label {{
            position: absolute;
            top: 0;
            left: 10px;
            line-height: 24px;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }}
    </style>
    """
    
    # Create the HTML for the progress bar
    progress_html = f"""
    {progress_css}
    <div class="custom-progress-container">
        <div class="custom-progress-bar">
            <div class="custom-progress-label">{display_label}</div>
        </div>
    </div>
    """
    
    # Display the custom progress bar
    st.markdown(progress_html, unsafe_allow_html=True)

def create_date_range_selector(
    default_days: int = 7,
    key_prefix: str = "date_range"
) -> Tuple[datetime, datetime]:
    """
    Create a date range selector with preset options.
    
    Args:
        default_days: Default number of days to look back
        key_prefix: Prefix for session state keys
    
    Returns:
        Tuple of (start_date, end_date)
    """
    from datetime import datetime, timedelta
    
    # Define preset options
    presets = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "This year": (datetime.now() - datetime(datetime.now().year, 1, 1)).days + 1,
        "Custom": 0
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_preset = st.selectbox(
            "Time period",
            list(presets.keys()),
            index=0,
            key=f"{key_prefix}_preset"
        )
    
    today = datetime.now().date()
    
    if selected_preset == "Custom":
        with col2:
            # If Custom is selected, show date input fields
            start_col, end_col = st.columns(2)
            with start_col:
                start_date = st.date_input(
                    "Start date",
                    today - timedelta(days=default_days),
                    key=f"{key_prefix}_start"
                )
            with end_col:
                end_date = st.date_input(
                    "End date",
                    today,
                    key=f"{key_prefix}_end"
                )
    else:
        # Calculate dates based on preset
        days = presets[selected_preset]
        start_date = today - timedelta(days=days)
        end_date = today
    
    return start_date, end_date
