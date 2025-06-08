"""
Modèles SQLAlchemy pour stocker les prédictions
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    dataset_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Features communes (selon le dataset)
    feature_1 = Column(Float)
    feature_2 = Column(Float)
    feature_3 = Column(Float)
    feature_4 = Column(Float)
    feature_5 = Column(Float)
    feature_6 = Column(Float)
    feature_7 = Column(Float)
    feature_8 = Column(Float)
    
    # Résultat de la prédiction
    prediction = Column(String(100))
    prediction_proba = Column(Float)
    
    # Métadonnées
    features_json = Column(Text)  # Stockage JSON des features complètes
    model_params = Column(Text)   # Paramètres du modèle utilisé
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, model={self.model_name}, dataset={self.dataset_name}, prediction={self.prediction})>"
