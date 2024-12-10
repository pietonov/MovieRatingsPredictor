######################### Import ##############################
import streamlit as st
import joblib
import os



st.set_page_config(page_title="Movie Ratings Predictor")

# Header Section
st.title("Movie Ratings Predictor")
st.image(
    "INPUT/MovieRatingsPredictor.jpg", 
    caption="AI-generated: Predicting Movies Ratings before You Watched It", 
    width=800
)


######################### Load Models #########################
@st.cache_resource
def load_models():
    try:
        glm_path = 'OUTPUT/glm_model.pkl'
        if not os.path.exists(glm_path):
            raise FileNotFoundError("Model files not found.")
        glm_model = joblib.load(glm_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise
    return glm_model

glm_full = load_models()



