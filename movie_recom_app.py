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

    # Check if movies were selected
    if len(selected_movies) == 0:
        st.warning("Please select at least one movie to proceed.")
        return pd.DataFrame()  # Return empty DataFrame if no movies selected

    # Filter data based on the selected genre
    filtered_data = data.copy()  # Keep a copy of the original dataset
    if genre_filter:
        filtered_data = filtered_data[filtered_data['genres'].str.contains(genre_filter, case=False, na=False)]

    # Ensure selected movies are in the dataset
    existing_movies = [movie for movie in selected_movies if movie.lower().strip() in filtered_data['normalized_title'].values]
    if len(existing_movies) == 0:
        st.warning("None of the selected movies were found in the database. Please select valid movies.")
        return pd.DataFrame()  # Return empty DataFrame if no valid movies selected

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
        st.warning("No valid vectors for the selected movies.")
        return pd.DataFrame()  # Return empty DataFrame if no valid vectors
    
    similarity_scores = cosine_similarity(selected_vectors, tfidf_matrix)
    mean_similarity = similarity_scores.mean(axis=0)

    # Add similarity scores to the dataset
    filtered_data['similarity'] = mean_similarity

    # Sort by similarity to get recommendations
    recommendations = filtered_data.sort_values(by='similarity', ascending=False).head(20)

    # Apply the nudity filter to the recommendations if requested
    if avoid_nudity:
        recommendations = recommendations[recommendations['NLP_nudity_score'] <= 2]

    # Add Sexual Nudity Category to the recommendations DataFrame
    recommendations['Sexual_Nudity_Category'] = recommendations['NLP_nudity_score'].apply(categorize_nudity_score)

    return recommendations.head(20)  # Return top 20 recommendations



# Streamlit UI
def main():
    st.title("Movie Recommender System")
    st.subheader("Find your next favourite movie!")

    # Load data
    data = load_data()

    # Step 1: Select 3 movies you like from a diverse and popular list
    st.write("### Step 1: Select 1 movie you like to get recommendations")

    # List of movies
    options = ['Avatar', 'The Dark Knight', 'Inception', 'Interstellar', 'The Lord of the Rings: The Return of the King', 
               'Fight Club', 'Harry Potter and the Goblet of Fire', 'John Wick', 'Shrek', 'Monster']

    # Use st.pills for selecting multiple movies
    selected_movies = st.pills("Select Movies you like", options, selection_mode="multi")
    # Limit to 3 selections if more than 3 are selected
    if len(selected_movies) > 3:
        st.warning("You can select up to 3 movies only. Extra selections will be ignored.")
    selected_movies = selected_movies[:3]
    # Display selected movies
    st.markdown(f"Your selected movies: {selected_movies}.")

    # Step 2: Select genre
    st.write("### Step 2: Select your preferred genre")
    all_genres = set(", ".join(data['genres'].dropna().unique()).split(", "))
    genre_filter = st.selectbox("Genre:", ["All"] + list(all_genres))
    genre_filter = None if genre_filter == "All" else genre_filter

    # Step 3: Minimal Sexual & Nudity Scenes
    st.write("### Step 3: Filter for minimal/no sexual & nudity scenes")
    avoid_nudity = st.checkbox("Show only movies with minimal/no sexual & nudity scenes")

    # Generate recommendations
    if st.button("Get Recommendations"):
        if len(selected_movies) < 3:
            st.warning("Please select at least 3 movie to proceed.")
        else:
            recommendations = generate_recommendations(selected_movies, genre_filter, avoid_nudity, data)

            if recommendations.empty:
                st.write("No recommendations could be generated based on the selected movies and filters.")
            else:
                st.write("### Recommendations:")
                for _, row in recommendations.iterrows():
                    st.write(f"**{row['title']} ({row['year']})**")
                    st.write(f"Runtime: {row['runtime']} minutes | Language: {row['original_language']}")
                    st.write(f"Genres: {row['genres']}")
                    st.write(f"Overview: {row['overview']}")
                    st.write(f"Directors: {row['directors']} | Cast: {row['cast']}")
                    st.write(f"**Sexual Nudity Comments:** {row['sex_nudity_summary']}")
                    st.write(f"**Sexual Nudity Category:** {row['Sexual_Nudity_Category']}")
                    st.write("---")


if __name__ == "__main__":
    main()
