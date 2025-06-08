import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv


# Charger les variables d'environnement
load_dotenv()

# Configuration de la base de données
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "root_password"),
    'database': os.getenv("DB_NAME", "mlflow_results")
}

class DatabaseManager:
    """Gestionnaire de base de données pour enregistrer les résultats"""
    
    def __init__(self, config):
        self.config = config
        self.init_database()
    
    def get_connection(self):
        """Établir une connexion à la base de données"""
        return mysql.connector.connect(**self.config)
    
    def init_database(self):
        """Initialiser les tables de la base de données"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Table pour les statistiques descriptives
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistiques (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    dataset_name VARCHAR(100),
                    colonne VARCHAR(100),
                    moyenne FLOAT,
                    ecart_type FLOAT,
                    minimum FLOAT,
                    maximum FLOAT,
                    mediane FLOAT,
                    q1 FLOAT,
                    q3 FLOAT,
                    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table pour les résultats des modèles ML
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resultats_ml (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    dataset_name VARCHAR(100),
                    model_name VARCHAR(100),
                    accuracy FLOAT,
                    precision_macro FLOAT,
                    recall_macro FLOAT,
                    f1_score_macro FLOAT,
                    confusion_matrix TEXT,
                    hyperparameters TEXT,
                    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table pour les expériences MLflow
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences_mlflow (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(255),
                    experiment_name VARCHAR(100),
                    dataset_name VARCHAR(100),
                    model_name VARCHAR(100),
                    accuracy FLOAT,
                    parameters TEXT,
                    metrics TEXT,
                    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Base de données initialisée avec succès")
            
        except mysql.connector.Error as err:
            print(f"❌ Erreur lors de l'initialisation de la base de données: {err}")
    
    def enregistrer_statistiques(self, dataset_name, stats_df):
        """Enregistrer les statistiques descriptives dans la base de données"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for colonne in stats_df.columns:
                stats = stats_df[colonne]
                cursor.execute("""
                    INSERT INTO statistiques 
                    (dataset_name, colonne, moyenne, ecart_type, minimum, maximum, mediane, q1, q3)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    dataset_name,
                    colonne,
                    float(stats['mean']) if pd.notnull(stats['mean']) else None,
                    float(stats['std']) if pd.notnull(stats['std']) else None,
                    float(stats['min']) if pd.notnull(stats['min']) else None,
                    float(stats['max']) if pd.notnull(stats['max']) else None,
                    float(stats['50%']) if pd.notnull(stats['50%']) else None,
                    float(stats['25%']) if pd.notnull(stats['25%']) else None,
                    float(stats['75%']) if pd.notnull(stats['75%']) else None
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except mysql.connector.Error as err:
            print(f"❌ Erreur lors de l'enregistrement des statistiques: {err}")
            return False
    
    def enregistrer_resultats_ml(self, dataset_name, model_name, metrics, hyperparams=None):
        """Enregistrer les résultats des modèles ML"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO resultats_ml 
                (dataset_name, model_name, accuracy, precision_macro, recall_macro, 
                 f1_score_macro, confusion_matrix, hyperparameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                dataset_name,
                model_name,
                metrics.get('accuracy'),
                metrics.get('precision_macro'),
                metrics.get('recall_macro'),
                metrics.get('f1_score_macro'),
                str(metrics.get('confusion_matrix')),
                str(hyperparams) if hyperparams else None
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except mysql.connector.Error as err:
            print(f"❌ Erreur lors de l'enregistrement des résultats ML: {err}")
            return False
    
    def enregistrer_experience_mlflow(self, run_id, experiment_name, dataset_name, 
                                    model_name, accuracy, parameters, metrics):
        """Enregistrer une expérience MLflow"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiences_mlflow 
                (run_id, experiment_name, dataset_name, model_name, accuracy, parameters, metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                run_id,
                experiment_name,
                dataset_name,
                model_name,
                accuracy,
                str(parameters),
                str(metrics)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except mysql.connector.Error as err:
            print(f"❌ Erreur lors de l'enregistrement de l'expérience MLflow: {err}")
            return False

def load_dataset(dataset_name):
    """Charger le dataset sélectionné"""
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_names'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.target_names
    
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_names'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.target_names
    
    elif dataset_name == "Diabetes":
        # Simuler le dataset Pima Indians Diabetes
        np.random.seed(42)
        n_samples = 768
        
        # Générer des données synthétiques similaires au dataset Pima Indians
        data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(70, 20, n_samples),
            'SkinThickness': np.random.normal(20, 15, n_samples),
            'Insulin': np.random.exponential(100, n_samples),
            'BMI': np.random.normal(32, 7, n_samples),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
            'Age': np.random.gamma(2, 15, n_samples).astype(int)
        }
        
        df = pd.DataFrame(data)
        # Créer une variable cible basée sur des seuils réalistes
        df['target'] = ((df['Glucose'] > 140) | 
                       (df['BMI'] > 35) | 
                       (df['Age'] > 50)).astype(int)
        df['target_names'] = df['target'].map({0: 'No Diabetes', 1: 'Diabetes'})
        
        return df, ['No Diabetes', 'Diabetes']

def train_model(X, y, model_name, hyperparams=None):
    """Entraîner un modèle de machine learning"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "Random Forest":
        if hyperparams:
            model = RandomForestClassifier(**hyperparams, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    elif model_name == "Logistic Regression":
        if hyperparams:
            model = LogisticRegression(**hyperparams, random_state=42, max_iter=1000)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
    
    elif model_name == "SVM":
        if hyperparams:
            model = SVC(**hyperparams, random_state=42)
        else:
            model = SVC(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_score_macro': report['macro avg']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    return model, metrics, X_test, y_test, y_pred

def main():
    st.set_page_config(page_title="ML Analysis avec MySQL", layout="wide")
    
    st.title("🧠 Analyse ML avec Base de Données MySQL")
    st.markdown("Analyse des datasets Iris, Wine et Diabetes avec enregistrement en base de données")
    
    # Initialiser le gestionnaire de base de données
    try:
        db_manager = DatabaseManager(DB_CONFIG)
        st.success("✅ Connexion à la base de données établie")
    except Exception as e:
        st.error(f"❌ Erreur de connexion à la base de données: {e}")
        st.stop()
    
    # Sidebar pour la sélection
    st.sidebar.header("Configuration")
    
    # Sélection du dataset
    dataset_name = st.sidebar.selectbox(
        "Choisir un dataset",
        ["Iris", "Wine", "Diabetes"]
    )
    
    # Sélection du modèle
    model_name = st.sidebar.selectbox(
        "Choisir un modèle",
        ["Random Forest", "Logistic Regression", "SVM"]
    )
    
    # Hyperparamètres
    st.sidebar.subheader("Hyperparamètres")
    hyperparams = {}
    
    if model_name == "Random Forest":
        hyperparams['n_estimators'] = st.sidebar.slider("Nombre d'arbres", 10, 200, 100)
        hyperparams['max_depth'] = st.sidebar.slider("Profondeur max", 1, 20, 10)
    
    elif model_name == "Logistic Regression":
        hyperparams['C'] = st.sidebar.slider("Régularisation C", 0.01, 10.0, 1.0)
    
    elif model_name == "SVM":
        hyperparams['C'] = st.sidebar.slider("Paramètre C", 0.01, 10.0, 1.0)
        hyperparams['kernel'] = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly'])
    
    # Charger les données
    df, target_names = load_dataset(dataset_name)
    
    # Affichage des données
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 Dataset: {dataset_name}")
        st.write(f"**Forme du dataset:** {df.shape}")
        st.write(f"**Classes:** {', '.join(target_names)}")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("📈 Statistiques descriptives")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe()
        st.dataframe(stats_df)
        
        # Enregistrer les statistiques en base
        if st.button("💾 Enregistrer les statistiques"):
            if db_manager.enregistrer_statistiques(dataset_name, stats_df):
                st.success("✅ Statistiques enregistrées en base de données")
            else:
                st.error("❌ Erreur lors de l'enregistrement")
    
    # Visualisations
    st.subheader("📊 Visualisations")
    
    if len(numeric_cols) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Heatmap de corrélation
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Matrice de corrélation')
        
        # Distribution de la variable cible
        target_counts = df['target_names'].value_counts()
        axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Distribution des classes')
        
        st.pyplot(fig)
    
    # Entraînement du modèle
    st.subheader("🤖 Entraînement du modèle")
    
    if st.button("🚀 Entraîner le modèle"):
        with st.spinner("Entraînement en cours..."):
            # Préparer les données
            X = df[numeric_cols].drop(['target'], axis=1, errors='ignore')
            y = df['target']
            
            # Démarrer une expérience MLflow
            mlflow.set_experiment(f"{dataset_name}_Classification")
            
            with mlflow.start_run() as run:
                # Entraîner le modèle
                model, metrics, X_test, y_test, y_pred = train_model(X, y, model_name, hyperparams)
                
                # Logger avec MLflow
                mlflow.log_params(hyperparams)
                mlflow.log_metrics({
                    'accuracy': metrics['accuracy'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro'],
                    'f1_score_macro': metrics['f1_score_macro']
                })
                mlflow.sklearn.log_model(model, "model")
                
                # Afficher les résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Métriques de performance")
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    st.metric("Precision (Macro)", f"{metrics['precision_macro']:.3f}")
                    st.metric("Recall (Macro)", f"{metrics['recall_macro']:.3f}")
                    st.metric("F1-Score (Macro)", f"{metrics['f1_score_macro']:.3f}")
                
                with col2:
                    st.subheader("🔥 Matrice de confusion")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                              cmap='Blues', ax=ax)
                    ax.set_title('Matrice de confusion')
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Valeurs réelles')
                    st.pyplot(fig)
                
                # Enregistrer en base de données
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Enregistrer les résultats ML"):
                        if db_manager.enregistrer_resultats_ml(dataset_name, model_name, metrics, hyperparams):
                            st.success("✅ Résultats ML enregistrés")
                        else:
                            st.error("❌ Erreur lors de l'enregistrement")
                
                with col2:
                    if st.button("📝 Enregistrer l'expérience MLflow"):
                        if db_manager.enregistrer_experience_mlflow(
                            run.info.run_id,
                            f"{dataset_name}_Classification",
                            dataset_name,
                            model_name,
                            metrics['accuracy'],
                            hyperparams,
                            metrics
                        ):
                            st.success("✅ Expérience MLflow enregistrée")
                        else:
                            st.error("❌ Erreur lors de l'enregistrement")

if __name__ == "__main__":
    main()