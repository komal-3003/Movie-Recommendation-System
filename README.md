# Movie Recommendation System: AI-Based Content Recommender
## Project Objective
The objective of this project is to build an intelligent movie recommendation system that suggests similar movies based on user input. The system uses TF-IDF similarity and TMDB API integration to deliver personalized and data-driven movie recommendations along with detailed movie insights.

## Project Description
- This project is an interactive movie recommendation web application built using Streamlit (frontend) and FastAPI (backend).
- It takes a movie name as input and returns:
- Movie details (title, overview, release date, poster)
- AI-based similar movie recommendations
- The system combines machine learning (TF-IDF similarity) with real-time TMDB API data to enhance recommendation quality and user experience.
- The UI is designed to be simple, fast, and visually appealing with movie posters and scores.

## Key Performance Indicators (KPIs)
- Recommendation Accuracy (Similarity Score)
- Response Time of API
- Number of Movies Processed in Dataset
- API Success Rate (TMDB + Backend)
- User Search Queries Handled

## Additional metrics include:
- Top recommended movies per search
- Similarity score distribution
- Movie metadata completeness (poster, overview, rating)
- API latency monitoring
- Dataset coverage of popular movies
  
## Dataset used
- <a href="https://github.com/komal-3003/HR-Analytics-Dashboard/blob/main/HR_Analytics.csv">Dataset</a>
## Project Process
## 1️.Data Collection
The dataset includes movie-related information such as:
- Movie titles
- Overview and descriptions
- Metadata (genres, release dates, ratings)
- TF-IDF features for similarity calculation
  
## 2.Data Cleaning & Preparation
To ensure accurate recommendations:
- Removed missing and duplicate movie entries
- Normalized movie titles for matching
- Converted text data into TF-IDF vectors
- Created index mapping for fast lookup
- Stored processed data using pickle files

## 3.Data Analysis
- Applied TF-IDF Vectorization to compute similarity between movies
- Used cosine similarity to rank similar movies
- Extracted top-N recommendations based on similarity score
- Integrated TMDB API for enriched movie metadata
  
## 4️.Dashboard Development
The backend handles:
- Movie search requests
- Recommendation generation
- TMDB API integration
- Parallel async processing for faster responses
- Structured API responses using Pydantic models

## 5.Frontend Development (Streamlit)
The dashboard was built using interactive UI components:
- Search bar for movie input
- Movie detail display section
- Recommendation grid with posters and scores
- Responsive 5-column movie layout
- Fallback images for missing posters
## Project Insights
- TF-IDF based similarity provides fast and reliable recommendations
- TMDB API significantly improves user experience with rich metadata
- Parallel async requests improve performance of recommendation enrichment
- Proper fallback handling prevents missing image/UI issues
- Most popular movies generate richer and more accurate recommendations
  
## User Interface
- <a href="https://github.com/komal-3003/Movie-Recommendation-System/blob/main/Screenshot%202026-05-17%20223152.png"> View User Interface </a>
## Conclusion
This project demonstrates how machine learning and real-time APIs can be combined to build a powerful movie recommendation system. It provides personalized suggestions with enriched movie details, helping users discover new movies easily while ensuring fast and interactive performance.
