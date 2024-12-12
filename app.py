import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Constants
DATA_PATH = "data/Job Listings.csv"
MODEL_PATH = "models/tfidf_model.pkl"

# Load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.dropna(subset=["job_description"], inplace=True)
        data["job_description"] = data["job_description"].fillna("").astype(str)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load model
@st.cache_resource
def load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Job recommendation logic
def get_recommendations(user_query, vectorizer, data):
    try:
        tfidf_matrix = vectorizer.transform(data["job_description"])
        query_vector = vectorizer.transform([user_query])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        data["similarity_score"] = similarity_scores
        
        top_recommendations = (
            data.nlargest(5, "similarity_score")
            .loc[:, ["Cleaned Job Title", "Category", "country", "average_hourly_rate", "link"]]
        )
        return top_recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

# Streamlit interface
def main():
    st.title("Job Finder: Personalized Recommendations")
    st.write("Find your next job by entering a job title or description below.")

    # Load resources
    data = load_data(DATA_PATH)
    vectorizer = load_model(MODEL_PATH)

    if data.empty or vectorizer is None:
        st.error("Unable to load required data or model. Please check your files.")
        return

    st.write(f"Data successfully loaded with {len(data)} job postings.")

    # User input
    user_query = st.text_input("Enter Job Title/Description:", "")

    if user_query:
        recommendations = get_recommendations(user_query, vectorizer, data)

        if not recommendations.empty:
            st.subheader("Top Job Recommendations")
            for _, job in recommendations.iterrows():
                st.markdown(
                    f"""
                    **{job['Cleaned Job Title']}**  
                    *Category:* {job['Category']}  
                    *Location:* {job['country']}  
                    *Hourly Rate:* ${job['average_hourly_rate']}  
                    [View Job Posting]({job['link']})  
                    ---
                    """
                )
        else:
            st.warning("No matching jobs found. Try refining your search.")

if __name__ == "__main__":
    main()
