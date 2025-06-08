# Iris Wine Diabetes ML Project with PostgreSQL Integration

Ce projet étend l'application Streamlit/MLflow existante en ajoutant une fonctionnalité complète de stockage et consultation des prédictions dans PostgreSQL.

## 🚀 Nouvelles Fonctionnalités

### 1. **Stockage des Prédictions**
- Sauvegarde automatique de chaque prédiction dans PostgreSQL
- Stockage des features, résultats, probabilités et métadonnées
- Horodatage automatique de chaque prédiction

### 2. **Consultation des Prédictions**
- Interface pour visualiser l'historique des prédictions
- Filtres par modèle, dataset, et période
- Export des données en CSV

### 3. **Statistiques et Analytics**
- Dashboard avec métriques en temps réel
- Graphiques de tendances et timeline
- Statistiques par modèle et dataset

### 4. **Gestion des Données**
- Suppression automatique des anciennes prédictions
- Export/Import des données
- Interface pgAdmin pour la gestion directe

## 📋 Prérequis

- Docker et Docker Compose
- Python 3.9+ (si exécution locale)
- 2GB de RAM minimum

## 🛠️ Installation

### Option 1: Avec Docker (Recommandé)

1. **Cloner le repository**
```bash
git clone https://github.com/samsaminfohub/iriswinediabetes.git
cd iriswinediabetes
```

2. **Ajouter les nouveaux fichiers**
Créez les fichiers suivants dans votre projet:
- `database/models.py`
- `database/connection.py`
- `database/crud.py`
- `database/__init__.py` (fichier vide)
- `app_with_db.py`
- `docker-compose.yml`
- `Dockerfile`
- `Dockerfile.mlflow`
- `.env`
- `init.sql`

3. **Configuration de l'environnement**
```bash
cp .env.example .env
# Éditez .env avec vos propres valeurs
```

4. **Démarrer les services**
```bash
docker-compose up -d
```

5. **Vérifier que tout fonctionne**
```bash
docker-compose ps
```

### Option 2: Installation Locale

1. **Installer PostgreSQL**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql
```

2. **Créer la base de données**
```sql
CREATE DATABASE predictions_db;
CREATE USER admin WITH PASSWORD 'admin123';
GRANT ALL PRIVILEGES ON DATABASE predictions_db TO admin;
```

3. **Installer les dépendances Python**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**
```bash
export DATABASE_URL=postgresql://admin:admin123@localhost:5432/predictions_db
export MLFLOW_TRACKING_URI=http://localhost:5000
```

5. **Initialiser la base de données**
```bash
python -c "from database.connection import init_db; init_db()"
```

6. **Démarrer l'application**
```bash
# Terminal 1 - MLflow
mlflow server --backend-store-uri $DATABASE_URL --default-artifact-root ./mlruns

# Terminal 2 - Streamlit
streamlit run app_with_db.py
```

## 📱 Utilisation

### 1. **Accès aux Services**
- **Streamlit App**: http://localhost:8501
- **MLflow UI**: http://localhost:5000
- **pgAdmin**: http://localhost:5050
  - Email: admin@admin.com
  - Password: admin123

### 2. **Workflow Typique**

1. **Entraîner un Modèle**
   - Sélectionnez un dataset (Iris, Wine, ou Diabetes)
   - Choisissez un algorithme
   - Cliquez sur "Train Model"

2. **Faire des Prédictions**
   - Entrez les valeurs des features
   - Cliquez sur "Predict & Save to Database"
   - La prédiction est automatiquement sauvegardée

3. **Consulter l'Historique**
   - Allez dans l'onglet "View Predictions"
   - Utilisez les filtres pour affiner la recherche
   - Exportez les données si nécessaire

4. **Analyser les Statistiques**
   - Onglet "Statistics" pour les métriques
   - Visualisez les tendances et performances

## 🗄️ Structure de la Base de Données

### Table `predictions`
| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER | Identifiant unique |
| model_name | VARCHAR(100) | Nom du modèle utilisé |
| dataset_name | VARCHAR(100) | Dataset (Iris/Wine/Diabetes) |
| timestamp | TIMESTAMP | Date/heure de la prédiction |
| feature_1..8 | FLOAT | Valeurs des features |
| prediction | VARCHAR(100) | Classe prédite |
| prediction_proba | FLOAT | Probabilité de la prédiction |
| features_json | TEXT | Features complètes en JSON |
| model_params | TEXT | Paramètres du modèle |

## 🔧 Configuration Avancée

### Variables d'Environnement
```bash
# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=predictions_db

# Application
DATABASE_URL=postgresql://admin:admin123@postgres:5432/predictions_db
MLFLOW_TRACKING_URI=http://mlflow:5000

# pgAdmin
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=admin123
```

### Personnalisation
- Modifiez `app_with_db.py` pour ajouter de nouveaux modèles
- Étendez `database/models.py` pour stocker plus d'informations
- Ajoutez des visualisations dans l'onglet Statistics

## 🐛 Dépannage

### Problème de connexion à PostgreSQL
```bash
# Vérifier les logs
docker-compose logs postgres

# Recréer les volumes
docker-compose down -v
docker-compose up -d
```

### Erreur d'import de modules
```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Performance lente
- Augmentez la mémoire allouée à Docker
- Créez des index supplémentaires dans PostgreSQL
- Limitez le nombre de prédictions affichées

## 📊 Exemples d'Utilisation

### Requête SQL pour obtenir les meilleures prédictions
```sql
SELECT * FROM predictions 
WHERE prediction_proba > 0.95 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Export programmé des données
```python
from database.crud import get_predictions
import pandas as pd

# Récupérer toutes les prédictions du dernier mois
predictions = get_predictions(
    db=session,
    start_date=datetime.now() - timedelta(days=30)
)

# Convertir en DataFrame et exporter
df = pd.DataFrame([p.__dict__ for p in predictions])
df.to_csv('predictions_monthly.csv', index=False)
```

## 🚀 Améliorations Futures

1. **API REST** pour accéder aux prédictions
2. **Notifications** en temps réel pour les prédictions importantes
3. **Comparaison** automatique des modèles
4. **Batch predictions** pour traiter plusieurs échantillons
5. **Authentification** et gestion des utilisateurs

## 📝 License

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📧 Contact

Pour toute question ou suggestion, ouvrez une issue sur GitHub.