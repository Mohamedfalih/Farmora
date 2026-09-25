from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.cluster import KMeans

app = FastAPI()

# Allow CORS for Spring Boot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy dataset for training K-Means
# Features: [XP, Sustainability_Score]
X_train = np.array([
    [10, 10], [20, 20], [30, 30], # Beginner
    [40, 40], [50, 50], [60, 60], # Intermediate
    [70, 70], [80, 80], [90, 80], # Advanced
    [90, 95], [100, 100], [100, 90] # Expert
])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train)

# Mappings based on dummy data distribution
cluster_names = {
    0: "BEGINNER",
    1: "INTERMEDIATE",
    2: "ADVANCED",
    3: "EXPERT"
}

recommendations = {
    "BEGINNER": "Soil Health Basics",
    "INTERMEDIATE": "Water Conservation 101",
    "ADVANCED": "Organic Pest Control",
    "EXPERT": "Advanced Sustainable Farming"
}

class FarmerData(BaseModel):
    xp: int
    score: int

@app.post("/predict")
def predict_cluster(data: FarmerData):
    # Predict cluster
    cluster_id = kmeans.predict([[data.xp, data.score]])[0]
    
    # K-Means cluster IDs might change ordering between runs, 
    # but for this MVP, we map it back dynamically or just use simple logic.
    # To ensure stable demo behavior, we will assign manually based on score/xp limits.
    if data.score < 30:
        cluster = "BEGINNER"
    elif data.score < 60:
        cluster = "INTERMEDIATE"
    elif data.score < 80:
        cluster = "ADVANCED"
    else:
        cluster = "EXPERT"
        
    return {
        "cluster": cluster,
        "recommendedModule": recommendations[cluster]
    }
