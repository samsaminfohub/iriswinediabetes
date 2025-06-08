# 🧠 Projet ML avec Base de Données MySQL

## Vue d'ensemble

Ce projet  `iriswinediabetes` qui intègre l'analyse de datasets de machine learning (Iris, Wine, Diabetes) avec un système complet d'enregistrement des résultats dans une base de données MySQL. Il combine Streamlit pour l'interface utilisateur, MLflow pour le tracking des expériences, et MySQL pour la persistance des données.

## ✨ Fonctionnalités

- **Interface Streamlit interactive** pour l'analyse des données
- **Support de 3 datasets** : Iris, Wine, et Diabetes (Pima Indians)
- **Modèles ML multiples** : Random Forest, Logistic Regression, SVM
- **Enregistrement automatique** des statistiques et résultats en base MySQL
- **Tracking MLflow** intégré
- **Visualisations avancées** avec Seaborn et Matplotlib
- **Utilitaires d'administration** de la base de données
- **Requêtes SQL prêtes à l'emploi** pour l'analyse
- **Export des données** en multiple formats (CSV, JSON, Excel)
- **Génération de rapports HTML**

## 🗄️ Structure de la Base de Données

### Tables principales

#### `statistiques`
Stocke les statistiques descriptives des datasets :
```sql
- id (INT, AUTO_INCREMENT)
- dataset_name (VARCHAR(100))
- colonne (VARCHAR(100))
- moyenne, ecart_type, minimum, maximum (FLOAT)
- mediane, q1, q3 (FLOAT)
- date_creation (TIMESTAMP)
```

#### `resultats_ml`
Stocke les résultats des modèles de machine learning :
```sql
- id (INT, AUTO_INCREMENT)
- dataset_name, model_name (VARCHAR(100))
- accuracy, precision_macro, recall_macro, f1_score_macro (FLOAT)
- confusion_matrix, hyperparameters (TEXT)
- date_creation (TIMESTAMP)
```

#### `experiences_mlflow`
Stocke les expériences MLflow :
```sql
- id (INT, AUTO_INCREMENT)
- run_id (VARCHAR(255), UNIQUE)
- experiment_name, dataset_name, model_name (VARCHAR(100))
- accuracy (FLOAT)
- parameters, metrics (TEXT)
- date_creation (TIMESTAMP)
```

## 🛠️ Installation

### Prérequis
- Python 3.9+
- Docker et Docker Compose
- MySQL 8.0+ (ou utiliser le container Docker)

### Installation locale

1. **Cloner le projet**
```bash
git clone <repository_url>
cd iriswinediabetes-mysql
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**
```bash
cp .env.example .env
# Éditer .env avec vos paramètres de base de données
```

5. **Initialiser la base de données**
```bash
# Option 1: Utiliser Docker Compose
docker-compose up -d mysql

# Option 2: Base MySQL existante
mysql -u root -p < init.sql
```

### Installation avec Docker

1. **Lancer tous les services**
```bash
docker-compose up -d
```

2. **Accéder aux interfaces**
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- MySQL: localhost:3306

## 🚀 Utilisation

### Interface Streamlit

1. **Démarrer l'application**
```bash
streamlit run app.py
```

2. **Naviguer dans l'interface**
- Sélectionner un dataset (Iris, Wine, Diabetes)
- Choisir un modèle ML
- Ajuster les hyperparamètres
- Visualiser les statistiques
- Entraîner le modèle
- Enregistrer les résultats


## 📊 Exemples de Requêtes SQL

### Performances des modèles par dataset
```sql
SELECT 
    dataset_name,
    model_name,
    ROUND(AVG(accuracy), 4) as accuracy_moyenne,
    COUNT(*) as nb_executions
FROM resultats_ml
GROUP BY dataset_name, model_name
ORDER BY dataset_name, accuracy_moyenne DESC;
```

### Meilleur modèle par dataset
```sql
SELECT 
    dataset_name,
    model_name,
    ROUND(accuracy, 4) as accuracy,
    date_creation
FROM resultats_ml r1
WHERE accuracy = (
    SELECT MAX(accuracy) 
    FROM resultats_ml r2 
    WHERE r2.dataset_name = r1.dataset_name
)
ORDER BY dataset_name;
```

### Évolution des performances dans le temps
```sql
SELECT 
    dataset_name,
    model_name,
    DATE(date_creation) as jour,
    ROUND(accuracy, 4) as accuracy,
    ROUND(AVG(accuracy) OVER (
        PARTITION BY dataset_name, model_name 
        ORDER BY date_creation 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 4) as moyenne_mobile
FROM resultats_ml
ORDER BY dataset_name, model_name, date_creation;
```

## 🔧 Configuration

### Variables d'environnement (.env)
```env
# Configuration MySQL
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=mlflow_results

# Configuration MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlflow-artifacts
```

### Configuration Docker
Le fichier `docker-compose.yml` configure :
- **MySQL** : Base de données principale
- **MLflow** : Serveur de tracking
- **Streamlit** : Interface utilisateur

## 📈 Analyse et Monitoring

### Métriques suivies
- **Accuracy** : Précision générale du modèle
- **Precision (Macro)** : Précision moyenne par classe
- **Recall (Macro)** : Rappel moyen par classe
- **F1-Score (Macro)** : Score F1 moyen par classe
- **Matrice de confusion** : Détail des prédictions

### Statistiques descriptives
Pour chaque colonne numérique :
- Moyenne, écart-type
- Minimum, maximum
- Médiane, quartiles (Q1, Q3)

### Rapports automatiques
- **Tableau de bord HTML** avec toutes les métriques
- **Graphiques de tendances** des performances
- **Export des données** pour analyse externe


## 🎯 Cas d'Usage

### 1. Comparaison de modèles
```python
# Via Streamlit : tester différents modèles sur le même dataset
# Via SQL : analyser les performances historiques
```

### 2. Optimisation d'hyperparamètres
```python
# Tester différentes configurations
# Tracker avec MLflow
# Analyser les résultats en base
```

### 3. Analyse de tendances
```python
# Voir l'évolution des performances dans le temps
# Identifier les régressions
# Optimiser les processus ML
```

### 4. Rapports de performance
```python
# Génération automatique de rapports
# Export pour présentations
# Monitoring continu
```

## 🔍 Dépannage

### Problèmes courants

#### Erreur de connexion MySQL
```bash
# Vérifier que MySQL est démarré
docker-compose ps

# Tester la connexion
python database_utils.py test
```

#### Problèmes de permissions
```bash
# Vérifier les variables d'environnement
cat .env

# Tester avec un utilisateur différent
mysql -u root -p -e "SHOW GRANTS FOR 'mlflow_user'@'%';"
```

#### Données corrompues
```bash
# Vérifier l'intégrité
python database_utils.py status

# Nettoyer si nécessaire
python database_utils.py cleanup --days 0 --dry-run
```

## 📚 Ressources Supplémentaires

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Datasets
- **Iris** : Classification de fleurs (4 features, 3 classes)
- **Wine** : Classification de vins (13 features, 3 classes)  
- **Diabetes** : Prédiction du diabète (8 features, 2 classes)