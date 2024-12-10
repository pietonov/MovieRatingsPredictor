######################### Import ##############################
import streamlit as st


st.set_page_config(page_title="Movie Ratings Predictor")

# Header Section
st.title("Movie Ratings Predictor")
st.image(
    "INPUT/MovieRatingsPredictor.jpg", 
    caption="AI-generated: Predicting Movies Ratings before You Watched It", 
    width=800
)


######################### Sidebar Navigation ##################
# Contact Information
st.sidebar.title("Contact & Support")
st.sidebar.markdown("""
For professional inquiries or feedback:  
- **Email**: support@movieratingspredictor.com  
- **Phone**: +1 (555) 987-6543   
""")
