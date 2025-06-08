from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ExperimentRequest(BaseModel):
    dataset: str
    model: str
    features: List[str]
    track_with_mlflow: bool = True

# Available datasets and models
DATA = {
    "iris": load_iris,
    "wine": load_wine,
    "diabetes": load_diabetes
}

PROBLEMS = {
    "iris": "classification",
    "wine": "classification",
    "diabetes": "regression"
}

MODELS = {
    "classification": {
        "KNN": KNeighborsClassifier,
        "SVM": SVC
    },
    "regression": {
        "LR": LinearRegression,
        "RFR": RandomForestRegressor
    }
}

@app.get("/datasets")
async def get_datasets():
    return {"datasets": list(DATA.keys())}

@app.get("/models/{dataset}")
async def get_models(dataset: str):
    problem_type = PROBLEMS.get(dataset)
    if not problem_type:
        return {"error": "Dataset not found"}
    return {"models": list(MODELS[problem_type].keys())}

@app.get("/features/{dataset}")
async def get_features(dataset: str):
    if dataset not in DATA:
        return {"error": "Dataset not found"}
    data = DATA[dataset](as_frame=True)
    df = data['data']
    return {"features": df.columns.tolist()}

@app.post("/run-experiment")
async def run_experiment(request: ExperimentRequest):
    # Load data
    data = DATA[request.dataset](as_frame=True)
    df = data['data']
    df['target'] = data['target']
    
    # Prepare features and target
    X = df[request.features].copy()
    y = df['target'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Get model
    problem_type = PROBLEMS[request.dataset]
    model_class = MODELS[problem_type][request.model]
    model = model_class()
    
    # MLflow tracking
    if request.track_with_mlflow:
        mlflow.set_experiment(request.dataset)
        mlflow.start_run()
        mlflow.log_param('model', request.model)
        mlflow.log_param('features', request.features)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    if problem_type == "classification":
        metric_name = "f1_score"
        metric_train = f1_score(y_train, preds_train, average='micro')
        metric_test = f1_score(y_test, preds_test, average='micro')
    else:
        metric_name = "r2_score"
        metric_train = r2_score(y_train, preds_train)
        metric_test = r2_score(y_test, preds_test)
    
    # Log metrics
    if request.track_with_mlflow:
        mlflow.log_metric(f"{metric_name}_train", metric_train)
        mlflow.log_metric(f"{metric_name}_test", metric_test)
        mlflow.end_run()
    
    return {
        "metrics": {
            f"{metric_name}_train": round(metric_train, 3),
            f"{metric_name}_test": round(metric_test, 3)
        },
        "model": request.model,
        "dataset": request.dataset
    }