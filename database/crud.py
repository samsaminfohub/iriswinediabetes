"""
Opérations CRUD pour les prédictions
"""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import json
from .models import Prediction

def create_prediction(
    db: Session,
    model_name: str,
    dataset_name: str,
    features: dict,
    prediction: str,
    prediction_proba: float,
    model_params: dict = None
) -> Prediction:
    """Créer une nouvelle prédiction dans la base de données"""
    
    # Créer l'objet prédiction
    db_prediction = Prediction(
        model_name=model_name,
        dataset_name=dataset_name,
        prediction=prediction,
        prediction_proba=prediction_proba,
        features_json=json.dumps(features),
        model_params=json.dumps(model_params) if model_params else None
    )
    
    # Mapper les features selon leur index
    for i, (key, value) in enumerate(features.items(), 1):
        if i <= 8:  # On a 8 colonnes de features génériques
            setattr(db_prediction, f'feature_{i}', float(value) if value is not None else None)
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Prediction]:
    """Récupérer les prédictions avec filtres optionnels"""
    
    query = db.query(Prediction)
    
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    
    if dataset_name:
        query = query.filter(Prediction.dataset_name == dataset_name)
    
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)
    
    return query.order_by(Prediction.timestamp.desc()).offset(skip).limit(limit).all()

def get_prediction_by_id(db: Session, prediction_id: int) -> Optional[Prediction]:
    """Récupérer une prédiction par son ID"""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()

def get_predictions_statistics(db: Session, dataset_name: Optional[str] = None):
    """Obtenir des statistiques sur les prédictions"""
    query = db.query(Prediction)
    
    if dataset_name:
        query = query.filter(Prediction.dataset_name == dataset_name)
    
    total_predictions = query.count()
    
    # Statistiques par modèle
    model_stats = {}
    for model in db.query(Prediction.model_name).distinct():
        model_name = model[0]
        model_query = query.filter(Prediction.model_name == model_name)
        model_stats[model_name] = {
            'count': model_query.count(),
            'avg_proba': db.query(func.avg(Prediction.prediction_proba)).filter(
                Prediction.model_name == model_name
            ).scalar()
        }
    
    # Prédictions des dernières 24h
    last_24h = datetime.utcnow() - timedelta(hours=24)
    recent_predictions = query.filter(Prediction.timestamp >= last_24h).count()
    
    return {
        'total_predictions': total_predictions,
        'recent_predictions_24h': recent_predictions,
        'model_statistics': model_stats
    }

def delete_old_predictions(db: Session, days: int = 30):
    """Supprimer les prédictions plus anciennes que X jours"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    deleted = db.query(Prediction).filter(Prediction.timestamp < cutoff_date).delete()
    db.commit()
    return deleted