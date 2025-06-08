"""
Configuration de la connexion à PostgreSQL
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Configuration de la base de données
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://username:password@localhost:5432/predictions_db"
)

# Création de l'engine
engine = create_engine(DATABASE_URL, echo=True)

# Création de la session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialise la base de données"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Générateur de session pour les requêtes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()