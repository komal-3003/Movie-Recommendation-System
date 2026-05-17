import requests
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = "http://127.0.0.1:8002"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# =============================
# STYLE
# =============================
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 1200px; }
.movie-title { font-size: 0.9rem; height: 2.2rem; overflow: hidden; }
.card { border: 1px solid #eee; padding: 10px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# =============================
# API CALL
# =============================
def get_api(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        if r.status_code != 200:
            return None, r.text
        return r.json(), None
    except Exception as e:
        return None, str(e)

# =============================
# SAFE IMAGE
# =============================
def safe_image(url):
    return url if url else "https://via.placeholder.com/300x450?text=No+Image"

# =============================
# GRID DISPLAY
# =============================
def show_movies(recs):
    if not recs:
        st.warning("No recommendations found.")
        return

    cols = st.columns(5)

    for i, r in enumerate(recs):
        with cols[i % 5]:

            tmdb = r.get("tmdb") or {}

            poster = safe_image(tmdb.get("poster_url"))

            st.image(poster, use_container_width=True)

            title = r.get("title", "Unknown Title")
            score = r.get("score", 0)

            st.markdown(f"**{title}**")
            st.caption(f"⭐ Score: {score:.2f}")

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommendation System")
st.write("Search a movie and get AI-based recommendations")

# =============================
# INPUT
# =============================
movie = st.text_input("Enter Movie Name", placeholder="e.g. Batman, Avengers")

# =============================
# ACTION
# =============================
if st.button("Get Recommendations"):

    if not movie:
        st.warning("Please enter a movie name")
        st.stop()

    with st.spinner("Fetching recommendations..."):

        data, err = get_api("/recommend", {"query": movie})

        if err or not data:
            st.error(f"Error: {err}")
            st.stop()

        # =============================
        # MOVIE DETAILS
        # =============================
        st.subheader("🎥 Movie Details")

        movie_details = data["movie_details"]

        col1, col2 = st.columns([1, 2])

        with col1:
            poster = safe_image(movie_details.get("poster_url"))
            st.image(poster, use_container_width=True)

        with col2:
            st.markdown(f"### {movie_details.get('title', 'N/A')}")
            st.write(movie_details.get("overview", "No overview available"))
            st.write("📅 Release Date:", movie_details.get("release_date", "N/A"))

        # =============================
        # RECOMMENDATIONS
        # =============================
        st.divider()
        st.subheader("🍿 Recommended Movies")

        show_movies(data.get("recommendations", []))