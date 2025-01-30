import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

# Load your dataset
@st.cache_data
def load_data():
    # Replace this with the path to your dataset
    df = pd.read_csv("movies_with_sexual_score.csv")
    return df

# Categorize nudity score into categories
def categorize_nudity_score(score):
    """Classify the nudity score into categories."""
    if score <= 1:
        return 'None'
    elif 2 <= score <= 4:
        return 'Mild'
    elif 5 <= score <= 9:
        return 'Moderate'
    else:
        return 'Severe'

# Generate movie recommendations
def generate_recommendations(selected_movies, genre_filter, avoid_nudity, data):
    # Normalise titles for consistent matching
    data['normalized_title'] = data['title'].str.lower().str.strip()

    # Filter data based on the selected genre
    filtered_data = data.copy()  # Keep a copy of the original dataset
    if genre_filter:
        filtered_data = filtered_data[filtered_data['genres'].str.contains(genre_filter, case=False, na=False)]

    # If no movies are selected, just recommend based on genre and sexual content
    if len(selected_movies) == 0:
        st.write("No movies selected. Showing recommendations based on genre and sexual content preferences.")
        
        # NLP-based recommendations using TF-IDF
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(filtered_data['keywords'].fillna(''))

        # Check if tfidf_matrix has valid data
        if tfidf_matrix.shape[0] == 0:
            st.warning("No valid data to compute recommendations.")
            return pd.DataFrame()  # Return empty DataFrame if no valid data

        # Calculate similarity scores (using the entire dataset since no movies were selected)
        similarity_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)
        mean_similarity = similarity_scores.mean(axis=0)

        # Add similarity scores to the dataset
        filtered_data['similarity'] = mean_similarity

        # Sort by similarity to get recommendations
        recommendations = filtered_data.sort_values(by='similarity', ascending=False).head(100)
    else:
        # If movies are selected, proceed as normal
        existing_movies = selected_movies  # Use selected movies directly

        # NLP-based recommendations using TF-IDF
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(filtered_data['keywords'].fillna(''))

        # Check if tfidf_matrix has valid data
        if tfidf_matrix.shape[0] == 0:
            st.warning("No valid data to compute recommendations.")
            return pd.DataFrame()  # Return empty DataFrame if no valid data

        # Calculate similarity scores
        selected_indices = filtered_data[filtered_data['normalized_title'].isin([movie.lower().strip() for movie in existing_movies])].index
        selected_vectors = tfidf_matrix[selected_indices]
        if selected_vectors.shape[0] == 0:
            st.warning("No valid recommendation for the selected genre. Please select another genre or search the genre without selecting any movies in Step 1")
            return pd.DataFrame()  # Return empty DataFrame if no valid vectors

        similarity_scores = cosine_similarity(selected_vectors, tfidf_matrix)
        mean_similarity = similarity_scores.mean(axis=0)

        # Add similarity scores to the dataset
        filtered_data['similarity'] = mean_similarity

        # Sort by similarity to get recommendations
        recommendations = filtered_data.sort_values(by='similarity', ascending=False).head(100)

    # Apply the nudity filter to the recommendations if requested
    if avoid_nudity:
        recommendations = recommendations[recommendations['NLP_nudity_score'] <= 2]

    # Add Sexual Nudity Category to the recommendations DataFrame
    recommendations['Sexual_Nudity_Category'] = recommendations['NLP_nudity_score'].apply(categorize_nudity_score)

    return recommendations.head(100)  # Return top 100 recommendations


# Streamlit UI
def main():

# Customised Title with CSS
    st.markdown(
        """
    <style>
        /* Set the background color of the app */
        .stApp {
            background-color: #EBEBEB; /* Dark Grey */
            color: black;
        }

        /* Make sure text stays readable */
        h1, h2, h3, h4, h5, h6, label {
            color: #FFFAFB;
        }

        p {
            color: black;
        }

        .recommendation-title {
            font-size: 44px;
            font-weight: bold;
            color: #4A90E2;  /* Optional: Change colour */
        }
        .recommendation-info {
            font-size: 28px;
            color: white;  /* Optional: Change colour */
        }

    </style>
        """,
        unsafe_allow_html=True,
    )

    # Wrap your app content inside the container div
    st.markdown('<div class="app-container">', unsafe_allow_html=True)


    # Display your logo using the URL
    logo_url = "https://noodle.digitalfutures.com/studentuploads/innocine-new-high-resolution-logo-transparent.png"
    st.image(logo_url, width=300)


    # Load data
    data = load_data()

# Step 1: Select Movies Section
    # Step 1: Select 3 movies you like from a diverse and popular list (Optional)
    st.write("#### Select up to 3 movies you like, to get our recommendations (Optional)")

    # List of movies
    options = ['Avatar', 'The Dark Knight', 'Inception', 'Interstellar', 'The Lord of the Rings: The Return of the King', 
               'Fight Club', 'Harry Potter and the Goblet of Fire', 'John Wick', 'Shrek', 'Monster']


    # Use st.pills for selecting multiple movies
    selected_movies = st.pills("", options, selection_mode="multi")
    # Limit to 3 selections if more than 3 are selected
    if len(selected_movies) > 3:
        st.warning("You can select up to 3 movies only. Extra selections will be ignored.",  icon="⚠️")
    selected_movies = selected_movies[:3]

    # Step 2: Select genre
    st.write("#### Select your preferred genre")
    all_genres = set(", ".join(data['genres'].dropna().unique()).split(", "))
    genre_filter = st.selectbox("Genre:", ["All"] + list(all_genres))
    genre_filter = None if genre_filter == "All" else genre_filter

    # Step 3: Minimal Sexual & Nudity Scenes
    st.write("#### Filter for minimal/no sexual & nudity scenes")
    avoid_nudity = st.checkbox("Show only movies with minimal/no sexual & nudity scenes")

    # Generate recommendations
    if st.button("Get Recommendations"):
        recommendations = generate_recommendations(selected_movies, genre_filter, avoid_nudity, data)

        if recommendations.empty:
            st.write("Sorry, no recommendations could be generated based on the selected movies and filters.")
        else:
            st.write("### Recommendations:")
            for _, row in recommendations.iterrows():
                st.write(f"#### **{row['title']} ({row['year']})**")
                st.write(f"##### Runtime: {row['runtime']} minutes | Language: {row['original_language']}")
                st.write(f"##### Genres: {row['genres']}")
                st.write(f"##### Overview: {row['overview']}")
                st.write(f"##### Directors: {row['directors']} | Cast: {row['cast']}")
                st.write(f"##### **Sexual Nudity Comments:** {row['sex_nudity_summary']}")
                st.write(f"##### **Sexual Nudity Category:** {row['Sexual_Nudity_Category']}")
                st.write("---")



    # Close the container div at the end of your app
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

