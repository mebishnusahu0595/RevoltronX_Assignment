import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import requests

# Load and preprocess data
# Enhanced data loading with synthetic data generation
# Add missing import at the top
import requests

# In load_data() function, fix the feature column name mismatch
def load_data():
    # Update column mapping to match new dataset
    column_mapping = {
        'Name': 'Restaurant',
        'Type': 'Cuisine',
        'Price_Range': 'Price_Level',
        'No of Reviews': 'Reviews_Count',
        'Reviews': 'Rating',
        'Street Address': 'Address',
        'Comments': 'Description'
    }
    
    df = pd.read_csv("data/restaurants.csv")
    df = df.rename(columns={k:v for k,v in column_mapping.items() if k in df.columns})

    # Ensure essential columns exist
    for col in ['Restaurant', 'Cuisine', 'Location', 'Price_Level']:
        if col not in df.columns:
            df[col] = 'Unknown' if col != 'Price_Level' else '$$'

    # Handle numeric columns
    df['Reviews_Count'] = pd.to_numeric(df.get('Reviews_Count', 0), errors='coerce').fillna(0)
    df['Rating'] = pd.to_numeric(df.get('Rating', 0), errors='coerce').fillna(0).clip(0, 5)

    # Create combined features
    # Change "Features" to "Enhanced_Features" to match SmartRecommender usage
    df["Enhanced_Features"] = (  # Changed from df["Features"]
        df["Cuisine"].fillna('').astype(str).str.strip() + " " +
        df["Location"].fillna('').astype(str).str.strip() + " " +
        df["Price_Level"].fillna('').astype(str).str.strip() + " " +
        df.get("Description", "").fillna('').astype(str).str.strip()
    )
    
    # Replace empty features with fallback text
    df["Enhanced_Features"] = df["Enhanced_Features"].replace('', 'general restaurant')
    
    # Add synthetic dietary preferences with proper column validation
    if 'Dietary_Preference' not in df.columns:
        df['Dietary_Preference'] = random.choices(
            ['Vegetarian', 'Vegan', 'Omnivore', 'Gluten-Free'],
            weights=[0.2, 0.1, 0.6, 0.1],
            k=len(df)
        )
    
    return df

# In SmartRecommender class, add API key
class SmartRecommender:
    def __init__(self, df):
        self.weather_api_key = "47d4a56a7961ebae12a18ceb9a56ae36"  # Replaced placeholder
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.sia = SentimentIntensityAnalyzer()
        
        # Add error handling for empty features
        if df["Enhanced_Features"].str.len().sum() == 0:
            raise ValueError("Feature matrix cannot be created from empty text data")
            
        self.weather_api_key = "YOUR_OPENWEATHER_API_KEY"
        
        # Train hybrid recommendation model
        X = self.vectorizer.fit_transform(df["Enhanced_Features"])
        y = df["Rating"]
        self.model = RandomForestRegressor()
        self.model.fit(X, y)
    
    def get_weather(self, location):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url).json()
            return {
                'temp': response['main']['temp'],
                'weather': response['weather'][0]['main'].lower()
            }
        except:
            return {'temp': 25, 'weather': 'clear'}

# Enhanced recommendation function with context awareness
def recommend_restaurants(restaurant_name, user_age, dietary_pref, location, top_n=5):
    global df, recommender
    try:
        # Contextual features
        weather = recommender.get_weather(location)
        sentiment = recommender.sia.polarity_scores(restaurant_name)['compound']
        
        # Get base predictions
        features = recommender.vectorizer.transform(df["Enhanced_Features"])
        df['Predicted_Rating'] = recommender.model.predict(features)
        
        # Apply contextual weights
        age_weight = 0.8 if user_age < 30 else 1.2  # Prefer affordable for younger users
        weather_weights = {
            'rain': {'Soup': 1.5, 'Comfort Food': 1.3},
            'cold': {'Hot Beverages': 1.4, 'Stews': 1.4}
        }
        
        # Fix missing Final_Score assignment
        df['Final_Score'] = (
            0.5 * df['Predicted_Rating'] +
            0.2 * (df['Rating'] * sentiment) +
            0.15 * age_weight +
            0.15 * df['Enhanced_Features'].apply(lambda x: sum(
                weather_weights.get(weather['weather'], {}).get(word, 1) 
                for word in x.split()
            ))
        )
        
        # Personalization filters
        # Add fallback for dietary preference filter
        if 'Dietary_Preference' not in df.columns:
            df['Dietary_Preference'] = 'Omnivore'  # Default value
        
        # Modified filtering logic with fallbacks
        filtered = df[
            (df['Dietary_Preference'] == dietary_pref) &
            (df['Location'] == location)
        ]
        
        # Fallback if no matches: broaden location search
        if len(filtered) == 0:
            filtered = df[df['Dietary_Preference'] == dietary_pref]
            
        # Final fallback: show top rated regardless of filters
        if len(filtered) == 0:
            filtered = df.sort_values('Rating', ascending=False)
        
        return filtered.sort_values('Final_Score', ascending=False).head(top_n)[["Restaurant", "Cuisine", "Location", "Rating", "Price_Level"]]
    
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()

# Enhanced Streamlit UI with user demographics
def main():
    st.title("🍽️ Smart Restaurant Recommender")
    
    # User context inputs
    with st.sidebar:
        st.header("User Profile")
        user_age = st.slider("Your Age", 18, 100, 25)
        dietary_pref = st.selectbox("Dietary Preference", ['Vegetarian', 'Vegan', 'Omnivore', 'Gluten-Free'])
        location = st.selectbox("Current Location", df['Location'].unique())
        cuisine_pref = st.multiselect("Preferred Cuisines", df['Cuisine'].unique())
    
    # Main interface
    st.header("Find Your Perfect Match")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.selectbox("Select a restaurant you like:", df['Restaurant'].unique())
    
    if st.button("Get Smart Recommendations"):
        with st.spinner("Analyzing multiple factors..."):
            recommendations = recommend_restaurants(user_input, user_age, dietary_pref, location)
            
            if not recommendations.empty:
                st.success("Top Recommendations:")
                for idx, row in recommendations.iterrows():
                    with st.expander(f"🌟 {row['Restaurant']} ({row['Cuisine']})"):
                        # In Streamlit display section:
                        st.markdown(f"""
                        - **Location**: {row['Location']}
                        - **Rating**: {row['Rating']}/5
                        - **Price Range**: {row.get('Price_Level', 'Not Available')}
                        - **Reviews**: {row.get('Reviews_Count', 0)} reviews
                        """)
            else:
                st.error("Restaurant not found. Please try another selection.")

# Initialize enhanced components
df = load_data()
recommender = SmartRecommender(df)

if __name__ == "__main__":
    main()