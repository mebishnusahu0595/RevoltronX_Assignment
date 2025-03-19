# RevoltronX_Assignment

# 🍽 Smart Restaurant Recommender

## 📌 Overview
The **Smart Restaurant Recommender** is a personalized restaurant recommendation system that utilizes **machine learning, natural language processing (NLP), and weather-based personalization** to suggest the best restaurants based on user preferences. It is built using **Streamlit**, **Scikit-learn**, and **Pandas**.

## 🚀 Features
- **Hybrid Recommendation System:** Uses **TF-IDF** vectorization for similarity analysis and **RandomForestRegressor** for rating prediction.
- **Personalized Recommendations:** Takes into account **age, weather, and dietary preferences** to enhance user experience.
- **Sentiment Analysis:** Analyzes restaurant reviews to assess overall customer sentiment.
- **Weather Integration:** Adapts recommendations based on real-time weather data from **OpenWeather API**.
- **Interactive UI:** Streamlit-powered intuitive interface for seamless user interaction.
- **Google Maps Integration:** Clickable links to view restaurant locations directly on Google Maps.

## 🛠 Tech Stack
- **Frontend:** Streamlit (Python)
- **Backend:** Scikit-learn, Pandas, NumPy
- **APIs:** OpenWeather API for weather-based personalization
- **ML Techniques:** TF-IDF, Sentiment Analysis, RandomForest Regression

## 📦 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/smart-restaurant-recommender.git
   cd smart-restaurant-recommender
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## ⚙️ Usage
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Enter your preferences in the sidebar.
3. View recommended restaurants with predicted ratings and weather-aware suggestions.

## 🔑 Configuration
### API Key Setup
- Replace `YOUR_OPENWEATHER_API_KEY` in the script with your OpenWeather API key.

## 📊 Example Output
- **Restaurant Name:** The Spice House  
- **Predicted Rating:** 4.5 ⭐  
- **Location:** Downtown City Center  
- **Dietary Options:** Vegan, Gluten-Free  
- **Google Maps Link:** [📍 View on Google Maps](https://www.google.com/maps/search/The+Spice+House+Downtown+City+Center)  

## 🛠 Future Enhancements
- Add **restaurant images** using an external API.
- Integrate **real-time customer reviews** for improved recommendations.
- Implement **multi-language support** for a wider audience.

## 🤝 Contributing
Feel free to fork the repository, make changes, and submit a pull request. Contributions are always welcome! 🚀

## 📜 License
This project is licensed under the MIT License.

---

🌟 **Enjoy personalized restaurant recommendations!** 🍽

