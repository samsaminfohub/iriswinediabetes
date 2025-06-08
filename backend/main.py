# api.py
"""
API REST pour accéder aux prédictions (Bonus)
Peut être ajouté au projet pour une intégration avec d'autres systèmes
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json

from database.connection import get_db
from database.crud import (
    create_prediction,
    get_predictions,
    get_prediction_by_id,
    get_predictions_statistics
)

app = FastAPI(
    title="ML Predictions API",
    description="API pour accéder aux prédictions ML stockées dans PostgreSQL",
    version="1.0.0"
)

# Modèles Pydantic pour l'API
class PredictionCreate(BaseModel):
    model_name: str
    dataset_name: str
    features: dict
    prediction: str
    prediction_proba: float
    model_params: Optional[dict] = None

class PredictionResponse(BaseModel):
    id: int
    model_name: str
    dataset_name: str
    timestamp: datetime
    prediction: str
    prediction_proba: float
    features_json: str
    model_params: Optional[str]

    class Config:
        orm_mode = True

class StatsResponse(BaseModel):
    total_predictions: int
    recent_predictions_24h: int
    model_statistics: dict

# Endpoints
@app.get("/")
def root():
    return {
        "message": "ML Predictions API",
        "endpoints": {
            "predictions": "/predictions",
            "prediction": "/predictions/{id}",
            "statistics": "/statistics",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/predictions", response_model=PredictionResponse)
def create_new_prediction(
    prediction: PredictionCreate,
    db: Session = Depends(get_db)
):
    """Créer une nouvelle prédiction"""
    try:
        db_prediction = create_prediction(
            db=db,
            model_name=prediction.model_name,
            dataset_name=prediction.dataset_name,
            features=prediction.features,
            prediction=prediction.prediction,
            prediction_proba=prediction.prediction_proba,
            model_params=prediction.model_params
        )
        return db_prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions", response_model=List[PredictionResponse])
def read_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Récupérer les prédictions avec filtres optionnels"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    predictions = get_predictions(
        db=db,
        skip=skip,
        limit=limit,
        model_name=model_name,
        dataset_name=dataset_name,
        start_date=start_date
    )
    
    return predictions

@app.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def read_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """Récupérer une prédiction spécifique par ID"""
    prediction = get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@app.get("/statistics", response_model=StatsResponse)
def read_statistics(
    dataset_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Obtenir les statistiques des prédictions"""
    stats = get_predictions_statistics(db, dataset_name)
    return stats

@app.get("/predictions/export/csv")
def export_predictions_csv(
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Exporter les prédictions au format CSV"""
    import pandas as pd
    from fastapi.responses import StreamingResponse
    import io
    
    start_date = datetime.utcnow() - timedelta(days=days)
    predictions = get_predictions(
        db=db,
        model_name=model_name,
        dataset_name=dataset_name,
        start_date=start_date,
        limit=10000
    )
    
    # Convertir en DataFrame
    data = []
    for p in predictions:
        data.append({
            'id': p.id,
            'timestamp': p.timestamp,
            'model_name': p.model_name,
            'dataset_name': p.dataset_name,
            'prediction': p.prediction,
            'prediction_proba': p.prediction_proba,
            'features': p.features_json
        })
    
    df = pd.DataFrame(data)
    
    # Créer un buffer pour le CSV
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=predictions_export_{datetime.now().strftime('%Y%m%d')}.csv"
        }
    )

# Pour lancer l'API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- Fichier de test pour l'API ---
# test_api.py
"""
Tests pour l'API REST
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_create_prediction():
    """Test de création d'une prédiction"""
    data = {
        "model_name": "Random Forest",
        "dataset_name": "Iris",
        "features": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        "prediction": "setosa",
        "prediction_proba": 0.98,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 5
        }
    }
    
    response = requests.post(f"{BASE_URL}/predictions", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_get_predictions():
    """Test de récupération des prédictions"""
    params = {
        "limit": 10,
        "dataset_name": "Iris",
        "days": 7
    }
    
    response = requests.get(f"{BASE_URL}/predictions", params=params)
    print(f"Status: {response.status_code}")
    print(f"Found {len(response.json())} predictions")
    return response.json()

def test_get_statistics():
    """Test de récupération des statistiques"""
    response = requests.get(f"{BASE_URL}/statistics")
    print(f"Statistics: {response.json()}")
    return response.json()

if __name__ == "__main__":
    print("Testing API endpoints...")
    
    # Test 1: Créer une prédiction
    print("\n1. Creating prediction...")
    prediction = test_create_prediction()
    
    # Test 2: Récupérer les prédictions
    print("\n2. Getting predictions...")
    predictions = test_get_predictions()
    
    # Test 3: Récupérer une prédiction spécifique
    if prediction:
        print(f"\n3. Getting prediction {prediction['id']}...")
        response = requests.get(f"{BASE_URL}/predictions/{prediction['id']}")
        print(f"Prediction details: {response.json()}")
    
    # Test 4: Récupérer les statistiques
    print("\n4. Getting statistics...")
    stats = test_get_statistics()