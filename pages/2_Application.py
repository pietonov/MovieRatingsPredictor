import streamlit as st
import os
import joblib
import pandas as pd

# Header Section
st.title("Movie Ratings Predictor")


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

actor_stats = pd.read_csv('OUTPUT/actor_stats.csv')
director_stats = pd.read_csv('OUTPUT/director_stats.csv')

actors_list = actor_stats['actors_list'].unique().tolist()
directors_list = director_stats['directors_list'].unique().tolist()



######################### GLM Model #########################

st.write("### Input Features")

with st.form("movie_form"):
    # Select actors
    selected_actors = st.multiselect(
        "Select Actors",
        options=actors_list,
        default=None
    )

    # Select directors
    selected_directors = st.multiselect(
        "Select Directors",
        options=directors_list,
        default=None
    )

    # Submit button for the form
    submit_button = st.form_submit_button(label="Submit")


# Display user inputs only after submission
if submit_button:
    st.write("Selected Actors:", selected_actors)
    st.write("Selected Directors:", selected_directors)

# # User Input Form
# with st.form("prediction_form"):
#     st.header("Input Flight Details")
#     distance = st.number_input("Distance (miles):", min_value=0, value=500)
#     large_airport = st.selectbox("Large Airport:", [1, 0], format_func=lambda x: "Yes" if x else "No")
#     has_passengers = st.selectbox("Has Passengers:", [1, 0], format_func=lambda x: "Yes" if x else "No")
#     passengers = st.number_input("Number of Passengers:", min_value=0, value=150, disabled=not has_passengers)
#     is_winter = st.selectbox("Winter Season:", [1, 0], format_func=lambda x: "Yes" if x else "No")
#     unique_carrier = st.selectbox("Carrier:", unique_carriers)
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     input_data = {
#         'DISTANCE': [distance],
#         'LARGE_AIRPORT': [large_airport],
#         'HAS_PASSENGERS': [has_passengers],
#         'PASSENGERS': [0 if not has_passengers else passengers],
#         'IS_WINTER': [is_winter],
#         'UNIQUE_CARRIER': [unique_carrier]
#     }
#     input_df = pd.DataFrame(input_data)

#     try:
#         glm_pred = glm_full.predict(input_df)
#         st.success(f"GLM Predicted Ground Time: {glm_pred.iloc[0]:.2f} minutes")
#     except Exception as e:
#         st.error(f"Error during GLM prediction: {e}")

