from fastapi import FastAPI, Query, Request
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="TF-IDF + SBERT Course Search API")

# ===== Load Models and Data =====
courses = pd.read_pickle(r"D:\Jupyter_Notebooks\Graduation Project\courses_with_vectors.pkl")

with open(r"D:\Jupyter_Notebooks\Graduation Project\tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(r"D:\Jupyter_Notebooks\Graduation Project\tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# ✅ Only load the precomputed SBERT embeddings
sbert_embeddings = np.load(r"D:\Jupyter_Notebooks\Graduation Project\sbert_embeddings.npy")

# ⚠️ Remove this line entirely:
# sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ===== Search Function =====
def search_courses_tfidf(query: str, top_n=5):
    if not isinstance(query, str) or not query.strip():
        return {"error": "Invalid or empty query."}

    query_vector = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        course = courses.iloc[idx]
        results.append({
            "course_name": course["course_name"],
            "description": course["description"],
            "url": course["url"],
            "similarity": float(sim_scores[idx])
        })
    return results

# ===== SBERT Recommendation Function =====
def recommend_by_course(enrolled_courses, top_n=5):
    if not isinstance(enrolled_courses, list):
        enrolled_courses = [enrolled_courses]

    lower_enrolled = [c.lower() for c in enrolled_courses]
    matched = courses[courses['course_name'].str.lower().isin(lower_enrolled)]

    if matched.empty:
        return {"error": "No matching enrolled courses found."}

    indices = matched.index.tolist()
    query_emb = sbert_embeddings[indices].mean(axis=0).reshape(1, -1)

    sim_scores = cosine_similarity(query_emb, sbert_embeddings).flatten()
    sim_scores[indices] = -1  # Exclude enrolled

    top_indices = sim_scores.argsort()[::-1][:top_n]
    recommendations = []
    for idx in top_indices:
        course = courses.iloc[idx]
        recommendations.append({
            "course_name": course["course_name"],
            "description": course["description"],
            "url": course["url"],
            "similarity": float(sim_scores[idx])
        })
    return recommendations

# ===== API Endpoints =====
@app.get("/")
def home():
    return {"message": "Welcome to the Course Search & Recommendation API!"}

@app.post("/search")
async def search(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        return {"error": "No query provided."}
    return {"results": search_courses_tfidf(query)}

@app.post("/recommend")
async def recommend(request: Request):
    data = await request.json()
    enrolled_courses = data.get("enrolled_courses", [])
    return {"recommendations": recommend_by_course(enrolled_courses)}
