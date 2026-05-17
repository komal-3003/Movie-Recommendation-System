import os
import pickle
import asyncio
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Add it to .env file")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf-matrix.pkl")

# =========================
# GLOBALS
# =========================
df: Optional[pd.DataFrame] = None
tfidf_matrix: Any = None
indices_obj: Any = None
TITLE_TO_IDX: Dict[str, int] = {}

# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    recommendations: List[TFIDFRecItem]


# =========================
# UTILS
# =========================
def norm_title(t: str) -> str:
    return str(t).strip().lower()


def img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


# =========================
# TMDB CALLS
# =========================
async def tmdb_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params["api_key"] = TMDB_API_KEY

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{TMDB_BASE}{endpoint}", params=params)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)

    return r.json()


async def tmdb_search_first(query: str):
    data = await tmdb_get("/search/movie", {
        "query": query,
        "language": "en-US",
        "include_adult": "false"
    })

    results = data.get("results", [])
    return results[0] if results else None


async def tmdb_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})

    return TMDBMovieDetails(
        tmdb_id=data["id"],
        title=data.get("title"),
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=img_url(data.get("poster_path")),
        backdrop_url=img_url(data.get("backdrop_path")),
        genres=data.get("genres", [])
    )


# =========================
# ENRICH RECOMMENDATIONS (NEW FIX)
# =========================
async def enrich_movie_card(title: str) -> Optional[TMDBMovieCard]:
    try:
        data = await tmdb_search_first(title)
        if not data:
            return None

        return TMDBMovieCard(
            tmdb_id=data["id"],
            title=data.get("title"),
            poster_url=img_url(data.get("poster_path")),
            release_date=data.get("release_date"),
            vote_average=data.get("vote_average"),
        )
    except:
        return None


# =========================
# TF-IDF LOGIC
# =========================
def build_index_map():
    global TITLE_TO_IDX
    TITLE_TO_IDX = {norm_title(k): v for k, v in indices_obj.items()}


def get_index(title: str) -> int:
    key = norm_title(title)
    if key not in TITLE_TO_IDX:
        raise HTTPException(status_code=404, detail="Movie not found in dataset")
    return TITLE_TO_IDX[key]


def recommend(title: str, top_n: int = 10):
    idx = get_index(title)

    vec = tfidf_matrix[idx]
    scores = (tfidf_matrix @ vec.T).toarray().ravel()

    ranked = np.argsort(-scores)

    results = []
    for i in ranked:
        if i == idx:
            continue

        try:
            movie_title = df.iloc[i]["title"]
        except:
            continue

        results.append((movie_title, float(scores[i])))

        if len(results) == top_n:
            break

    return results


# =========================
# LOAD DATA
# =========================
def load_data():
    global df, tfidf_matrix, indices_obj

    df = pickle.load(open(DF_PATH, "rb"))
    indices_obj = pickle.load(open(INDICES_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_MATRIX_PATH, "rb"))

    build_index_map()


@app.on_event("startup")
async def startup():
    load_data()


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"message": "Movie Recommendation API Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend", response_model=SearchResponse)
async def recommend_movie(query: str):

    # 1. Find best TMDB match
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie_id = best["id"]
    details = await tmdb_details(movie_id)

    # 2. TF-IDF recommendations
    recs = []

    try:
        results = recommend(details.title, top_n=10)

        # 3. FAST parallel enrichment (IMPORTANT FIX)
        tasks = [enrich_movie_card(title) for title, _ in results]
        tmdb_cards = await asyncio.gather(*tasks)

        for (title, score), card in zip(results, tmdb_cards):
            recs.append(
                TFIDFRecItem(
                    title=title,
                    score=score,
                    tmdb=card
                )
            )

    except:
        recs = []

    return SearchResponse(
        query=query,
        movie_details=details,
        recommendations=recs
    )