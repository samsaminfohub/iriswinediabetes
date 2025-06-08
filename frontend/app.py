import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BACKEND_URL = "http://backend:8000"

def selectbox_without_default(label, options):
    options = [''] + options
    format_func = lambda x: 'Select one option' if x == '' else x
    return st.selectbox(label, options, format_func=format_func)

def main():
    st.title("Machine Learning Experimentation Platform")
    
    # Get available datasets
    try:
        datasets = requests.get(f"{BACKEND_URL}/datasets").json()["datasets"]
    except:
        st.error("Backend service unavailable")
        st.stop()
    
    # Dataset selection
    dataset = selectbox_without_default("Choose a dataset", datasets)
    if not dataset:
        st.stop()
    
    # Get available models for selected dataset
    models = requests.get(f"{BACKEND_URL}/models/{dataset}").json()["models"]
    model = selectbox_without_default("Choose a model", models)
    if not model:
        st.stop()
    
    # Get available features
    features = requests.get(f"{BACKEND_URL}/features/{dataset}").json()["features"]
    selected_features = st.multiselect("Select features", features)
    
    if not selected_features:
        st.warning("Please select at least one feature")
        st.stop()
    
    # MLflow tracking option
    track_with_mlflow = st.checkbox("Track with MLflow", value=True)
    
    if st.button("Run Experiment"):
        payload = {
            "dataset": dataset,
            "model": model,
            "features": selected_features,
            "track_with_mlflow": track_with_mlflow
        }
        
        with st.spinner("Running experiment..."):
            response = requests.post(f"{BACKEND_URL}/run-experiment", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success("Experiment completed successfully!")
                
                # Display metrics
                st.subheader("Results")
                metrics = result["metrics"]
                st.write(f"Train {list(metrics.keys())[0]}: {metrics[list(metrics.keys())[0]]}")
                st.write(f"Test {list(metrics.keys())[1]}: {metrics[list(metrics.keys())[1]]}")
                
                # Show metrics visualization
                metrics_df = pd.DataFrame({
                    "Type": ["Train", "Test"],
                    "Score": [metrics[list(metrics.keys())[0]], metrics[list(metrics.keys())[1]]],
                    "Metric": [list(metrics.keys())[0], list(metrics.keys())[1]]
                })
                
                fig = px.bar(metrics_df, x="Type", y="Score", 
                             title=f"Model Performance ({dataset}, {model})",
                             color="Type")
                st.plotly_chart(fig)
                
                st.markdown(f"""
                **Experiment Details:**
                - Dataset: {result['dataset']}
                - Model: {result['model']}
                - Features: {', '.join(selected_features)}
                """)
            else:
                st.error(f"Error running experiment: {response.text}")

if __name__ == "__main__":
    main()