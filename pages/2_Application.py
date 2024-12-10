import streamlit as st
import os
import joblib
import pandas as pd

# Header Section
st.title("Movie Ratings Predictor")

######################### Load and Cache Data #########################
@st.cache_resource
def load_data_and_models():
    try:
        # Load GLM model
        glm_path = 'OUTPUT/glm_model.pkl'
        if not os.path.exists(glm_path):
            raise FileNotFoundError("Model file not found.")
        glm_model = joblib.load(glm_path)

        # Load actor and director statistics
        actor_stats = pd.read_csv('OUTPUT/actor_stats.csv')
        director_stats = pd.read_csv('OUTPUT/director_stats.csv')

        # Sort actors and directors by name
        actors_list = sorted(actor_stats['actors_list'].unique().tolist())
        directors_list = sorted(director_stats['directors_list'].unique().tolist())

    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        raise

    return glm_model, actor_stats, director_stats, actors_list, directors_list


# Load data and models
glm_full, actor_stats, director_stats, actors_list, directors_list = load_data_and_models()

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

# Display user inputs and make prediction only after submission
if submit_button:
    # Calculate average statistics for selected actors and directors
    avg_actor_score = actor_stats[actor_stats['actors_list'].isin(selected_actors)]['avg_critics_vote'].mean()
    avg_director_score = director_stats[director_stats['directors_list'].isin(selected_directors)]['avg_critics_vote'].mean()

    # Handle cases where no actors or directors are selected
    if pd.isna(avg_actor_score):
        avg_actor_score = actor_stats['avg_critics_vote'].mean()
    if pd.isna(avg_director_score):
        avg_director_score = director_stats['avg_critics_vote'].mean()

    # Prepare input data for the model
    input_data = pd.DataFrame({
        'actor_avg_critics_vote': [avg_actor_score],
        'director_avg_critics_vote': [avg_director_score]
    })

    # Make prediction
    try:
        predicted_rating = glm_full.predict(input_data)
        st.write(f"### Predicted Movie Rating: {predicted_rating[0]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
