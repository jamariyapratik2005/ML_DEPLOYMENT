"""
ML Deployment - Assignment 4
Instagram Like Counter API using FastAPI

Endpoints:
  GET  /                -> API status message (Q1)
  POST /predict-likes   -> Calculate total likes (Q2 + Q4)
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Instagram Like Counter API")


# =============================================
# Q4: Pydantic Model for input validation
# Enforces required fields with proper types
# =============================================
class LikesInput(BaseModel):
    current_likes: int
    new_likes: int


# =============================================
# Q1: Root Endpoint
# =============================================
@app.get("/")
def root():
    return {"message": "Instagram Like Counter API running"}


# =============================================
# Q2 + Q4: Predict Likes Endpoint
# - Accepts current_likes and new_likes
# - Returns updated total likes
# - Returns 400 error if fields are missing (Q4)
#   (Pydantic automatically validates required fields)
# =============================================
@app.post("/predict-likes")
def predict_likes(data: LikesInput):
    total_likes = data.current_likes + data.new_likes

    return {
        "current_likes": data.current_likes,
        "new_likes": data.new_likes,
        "total_likes": total_likes,
        "message": f"🎉 Post updated! Total likes: {total_likes}"
    }


# =============================================
# Run the app
# =============================================
if __name__ == "__main__":
    import uvicorn
    print("\n--- Instagram Like Counter API (FastAPI) ---")
    print("Endpoints:")
    print("  GET  http://127.0.0.1:8000/")
    print("  POST http://127.0.0.1:8000/predict-likes")
    print("  Docs http://127.0.0.1:8000/docs")
    print("-" * 45)
    uvicorn.run(app, host="127.0.0.1", port=8000)
