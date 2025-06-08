# app_with_db.py
"""
Application Streamlit avec intégration PostgreSQL pour iris, wine et diabetes
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px

# Import des modules de base de données
from database.connection import get_db, init_db
from database.models import Prediction
from database.crud import (
    create_prediction,
    get_predictions,
    get_prediction_by_id,
    get_predictions_statistics,
    delete_old_predictions
)

# Configuration de la page
st.set_page_config(
    page_title="ML Models with PostgreSQL",
    page_icon="🤖",
    layout="wide"
)

# Initialiser la base de données
init_db()

# Fonctions utilitaires
def load_dataset(dataset_name):
    """Charger le dataset sélectionné"""
    if dataset_name == "Iris":
        data = datasets.load_iris()
        feature_names = data.feature_names
        target_names = data.target_names.tolist()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
        feature_names = data.feature_names
        target_names = data.target_names.tolist()
    else:  # Diabetes
        data = datasets.load_diabetes()
        # Pour diabetes, on fait une classification binaire
        feature_names = data.feature_names
        target_names = ["No Diabetes", "Diabetes"]
        # Convertir en classification binaire
        data.target = (data.target > np.median(data.target)).astype(int)
    
    return data, feature_names, target_names

def train_model(model_type, X_train, y_train):
    """Entraîner le modèle sélectionné"""
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    else:  # SVM
        model = SVC(probability=True)
    
    model.fit(X_train, y_train)
    return model

def save_prediction_to_db(model_name, dataset_name, features_dict, prediction, prediction_proba, model_params=None):
    """Sauvegarder une prédiction dans la base de données"""
    db = next(get_db())
    try:
        pred = create_prediction(
            db=db,
            model_name=model_name,
            dataset_name=dataset_name,
            features=features_dict,
            prediction=str(prediction),
            prediction_proba=float(prediction_proba),
            model_params=model_params
        )
        return pred.id
    finally:
        db.close()

# Interface principale
st.title("🤖 Machine Learning Models with PostgreSQL Integration")
st.markdown("Train models on Iris, Wine, or Diabetes datasets and store predictions in PostgreSQL")

# Sidebar pour la configuration
with st.sidebar:
    st.header("Configuration")
    
    # Sélection du dataset
    dataset_name = st.selectbox(
        "Select Dataset",
        ["Iris", "Wine", "Diabetes"]
    )
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "SVM"]
    )
    
    # Paramètres de train/test split
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    # MLflow tracking
    use_mlflow = st.checkbox("Use MLflow Tracking", value=True)

# Tabs pour différentes fonctionnalités
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Train & Predict", "📊 View Predictions", "📈 Statistics", "🗑️ Manage Data"])

with tab1:
    # Charger et préparer les données
    data, feature_names, target_names = load_dataset(dataset_name)
    X = data.data
    y = data.target
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Features:** {len(feature_names)}")
        st.write(f"**Samples:** {len(X)}")
        st.write(f"**Classes:** {len(target_names)}")
        st.write(f"**Training samples:** {len(X_train)}")
        st.write(f"**Test samples:** {len(X_test)}")
    
    with col2:
        st.subheader("Feature Names")
        for i, name in enumerate(feature_names):
            st.write(f"{i+1}. {name}")
    
    # Bouton pour entraîner le modèle
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # MLflow tracking
            if use_mlflow:
                mlflow.set_experiment(f"{dataset_name}_{model_type}")
                with mlflow.start_run():
                    # Entraîner le modèle
                    model = train_model(model_type, X_train_scaled, y_train)
                    
                    # Prédictions
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Log dans MLflow
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("test_size", test_size)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.sklearn.log_model(model, "model")
            else:
                model = train_model(model_type, X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
            
            # Stocker le modèle en session
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names
            st.session_state['target_names'] = target_names
            st.session_state['model_params'] = {
                'model_type': model_type,
                'accuracy': accuracy,
                'test_size': test_size
            }
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
            
            # Afficher le rapport de classification
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, target_names=target_names))
    
    # Section pour faire des prédictions
    if 'model' in st.session_state:
        st.subheader("Make a Prediction")
        
        # Inputs pour les features
        input_features = {}
        cols = st.columns(2)
        for i, feature_name in enumerate(feature_names):
            with cols[i % 2]:
                input_features[feature_name] = st.number_input(
                    feature_name,
                    value=float(X[0][i]),
                    format="%.4f",
                    key=f"input_{i}"
                )
        
        if st.button("Predict & Save to Database"):
            # Préparer les données pour la prédiction
            input_data = np.array(list(input_features.values())).reshape(1, -1)
            input_scaled = st.session_state['scaler'].transform(input_data)
            
            # Faire la prédiction
            prediction = st.session_state['model'].predict(input_scaled)[0]
            prediction_proba = st.session_state['model'].predict_proba(input_scaled)[0].max()
            
            # Afficher le résultat
            predicted_class = st.session_state['target_names'][prediction]
            st.success(f"Prediction: **{predicted_class}** (Confidence: {prediction_proba:.2%})")
            
            # Sauvegarder dans la base de données
            pred_id = save_prediction_to_db(
                model_name=model_type,
                dataset_name=dataset_name,
                features_dict=input_features,
                prediction=predicted_class,
                prediction_proba=prediction_proba,
                model_params=st.session_state['model_params']
            )
            
            st.info(f"Prediction saved to database with ID: {pred_id}")

with tab2:
    st.subheader("📊 View Stored Predictions")
    
    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_model = st.selectbox(
            "Filter by Model",
            ["All"] + ["Logistic Regression", "Random Forest", "SVM"],
            key="filter_model"
        )
    with col2:
        filter_dataset = st.selectbox(
            "Filter by Dataset",
            ["All"] + ["Iris", "Wine", "Diabetes"],
            key="filter_dataset"
        )
    with col3:
        filter_days = st.number_input(
            "Last N days",
            min_value=1,
            max_value=365,
            value=7,
            key="filter_days"
        )
    with col4:
        limit = st.number_input(
            "Max results",
            min_value=10,
            max_value=1000,
            value=100,
            key="limit"
        )
    
    # Récupérer les prédictions
    db = next(get_db())
    start_date = datetime.utcnow() - timedelta(days=filter_days)
    
    predictions = get_predictions(
        db,
        model_name=filter_model if filter_model != "All" else None,
        dataset_name=filter_dataset if filter_dataset != "All" else None,
        start_date=start_date,
        limit=limit
    )
    db.close()
    
    if predictions:
        # Convertir en DataFrame
        df_predictions = pd.DataFrame([{
            'ID': p.id,
            'Timestamp': p.timestamp,
            'Model': p.model_name,
            'Dataset': p.dataset_name,
            'Prediction': p.prediction,
            'Confidence': f"{p.prediction_proba:.2%}",
            'Features': p.features_json
        } for p in predictions])
        
        # Afficher le tableau
        st.dataframe(df_predictions.drop('Features', axis=1), use_container_width=True)
        
        # Option pour voir les détails
        if st.checkbox("Show detailed features"):
            selected_id = st.selectbox("Select prediction ID", df_predictions['ID'].tolist())
            selected_pred = df_predictions[df_predictions['ID'] == selected_id].iloc[0]
            
            st.json(json.loads(selected_pred['Features']))
    else:
        st.info("No predictions found with the selected filters.")

with tab3:
    st.subheader("📈 Prediction Statistics")
    
    # Sélection du dataset pour les statistiques
    stat_dataset = st.selectbox(
        "Select Dataset for Statistics",
        ["All", "Iris", "Wine", "Diabetes"],
        key="stat_dataset"
    )
    
    db = next(get_db())
    
    # Obtenir les statistiques de base
    stats = get_predictions_statistics(
        db,
        dataset_name=stat_dataset if stat_dataset != "All" else None
    )
    
    # Afficher les métriques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    with col2:
        st.metric("Last 24h", stats['recent_predictions_24h'])
    with col3:
        st.metric("Models Used", len(stats['model_statistics']))
    
    # Graphique des prédictions par modèle
    if stats['model_statistics']:
        fig_models = go.Figure(data=[
            go.Bar(
                x=list(stats['model_statistics'].keys()),
                y=[v['count'] for v in stats['model_statistics'].values()],
                text=[v['count'] for v in stats['model_statistics'].values()],
                textposition='auto',
            )
        ])
        fig_models.update_layout(
            title="Predictions by Model",
            xaxis_title="Model",
            yaxis_title="Number of Predictions"
        )
        st.plotly_chart(fig_models, use_container_width=True)
    
    # Timeline des prédictions
    if stat_dataset != "All":
        # Récupérer toutes les prédictions pour le timeline
        all_predictions = get_predictions(
            db,
            dataset_name=stat_dataset,
            limit=1000
        )
        
        if all_predictions:
            df_timeline = pd.DataFrame([{
                'timestamp': p.timestamp,
                'model': p.model_name,
                'prediction': p.prediction
            } for p in all_predictions])
            
            # Grouper par jour
            df_timeline['date'] = pd.to_datetime(df_timeline['timestamp']).dt.date
            daily_counts = df_timeline.groupby(['date', 'model']).size().reset_index(name='count')
            
            fig_timeline = px.line(
                daily_counts,
                x='date',
                y='count',
                color='model',
                title='Predictions Timeline',
                labels={'count': 'Number of Predictions', 'date': 'Date'}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    db.close()

with tab4:
    st.subheader("🗑️ Data Management")
    
    st.warning("⚠️ Be careful with these operations as they permanently delete data!")
    
    # Supprimer les anciennes prédictions
    st.write("### Delete Old Predictions")
    days_to_keep = st.number_input(
        "Delete predictions older than (days)",
        min_value=1,
        max_value=365,
        value=30
    )
    
    if st.button("Delete Old Predictions", type="secondary"):
        db = next(get_db())
        deleted_count = delete_old_predictions(db, days=days_to_keep)
        db.close()
        st.success(f"Deleted {deleted_count} predictions older than {days_to_keep} days.")
    
    # Export des données
    st.write("### Export Data")
    if st.button("Export All Predictions to CSV"):
        db = next(get_db())
        all_predictions = get_predictions(db, limit=10000)
        db.close()
        
        if all_predictions:
            df_export = pd.DataFrame([{
                'id': p.id,
                'timestamp': p.timestamp,
                'model_name': p.model_name,
                'dataset_name': p.dataset_name,
                'prediction': p.prediction,
                'prediction_proba': p.prediction_proba,
                'features': p.features_json,
                'model_params': p.model_params
            } for p in all_predictions])
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions to export.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, MLflow, and PostgreSQL")