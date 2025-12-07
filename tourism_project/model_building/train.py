# for data manipulation
import pandas as pd
import numpy as np
import os
import joblib

# MLOps Tools
import mlflow
from huggingface_hub import HfApi, create_repo, HfFolder

# Scikit-learn components
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Evaluate performance (Need to ensure classification_report is imported)
from sklearn.metrics import classification_report, roc_auc_score
#Need to ensure RepositoryNotFoundError is imported from huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Model Algorithm
from xgboost import XGBClassifier

#For tracking the experiment
from pyngrok import ngrok
import subprocess
import atexit # For cleanup

# CONFIGURATION AND INITIALIZATION
HF_USER_ID = "SriniGS"
MODEL_REPO_ID = "SriniGS/tourism-package-prediction-v2"
EXPERIMENT_NAME = "tourism-package-prediction-experiment"
MODEL_FILENAME = "xgboost_model_pipeline.joblib"

# 1. Set the ML Flow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

# 3. Initialize the API by using the HF token.
api = HfApi(token=os.getenv("HF_TOKEN"))
print("MLflow and Hugging Face API initialized.")

# DATA LOADING
# 4. Read the csv files from Hugging Face repo

Xtrain_path = f"hf://datasets/SriniGS/tourism-package-prediction-v2/Xtrain.csv"
Xtest_path = f"hf://datasets/SriniGS/tourism-package-prediction-v2/Xtest.csv"
ytrain_path = f"hf://datasets/SriniGS/tourism-package-prediction-v2/ytrain.csv"
ytest_path = f"hf://datasets/SriniGS/tourism-package-prediction-v2/ytest.csv"

# 5. Load the dataset
try:
    X_train = pd.read_csv(Xtrain_path)
    X_test = pd.read_csv(Xtest_path)
    y_train = pd.read_csv(ytrain_path).iloc[:, 0]
    y_test = pd.read_csv(ytest_path).iloc[:, 0]
    print("Train/Test data loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading data from HF: {e}. Check HF_USER_ID and repo contents.")
    # Exit or use a fallback mechanism if data loading fails
    exit(1)

# PREPROCESSING AND PIPELINE DEFINITION
# 6. Seperate the numerical and categorical variables
numerical_cols = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]

# 7. Preprocess the data using make_column_transformer
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
)

# 8. Initialize a base model (XGBoost)
# Set the class weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# CORRECTED: scale_pos_weight is REMOVED from the base model initialization
# because the param_grid will test its values.
base_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# 9. Create a hyper parameter grid for fine tuning
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__max_depth': [3, 5],
    # The grid explicitly tests both the default (1) and the calculated balanced weight.
    'xgbclassifier__scale_pos_weight': [1, class_weight]
}

# 10. Use model pipeline to preprocess the data and build the model
model_pipeline = make_pipeline(preprocessor,base_model)

#4. FINE TUNING THE MODEL AND MLFLOW LOGGING

# 1. Start the ML Flow run (Main Run)
with mlflow.start_run():
    mlflow.set_tag("model_type", "XGBoost")    
    mlflow.log_param("cross_validation_folds", 5)
    mlflow.log_param("classification_threshold", 0.45) # Log the custom threshold

    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores (Nested Runs)
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        # Start a NESTED run for each combination
        with mlflow.start_run(nested=True, run_name=f"Trial_{i}"):
            # Log the parameters and the mean score for this specific CV trial
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_roc_auc", results['mean_test_score'][i])
            mlflow.log_metric("std_test_roc_auc", results['std_test_score'][i])
            print("-" * 50)
            print(f"Nested Run for Trial {i} logged.")
            print("-" * 50)
            print(f"Parameters: {results['params'][i]}")
            print(f"Mean Test Score: {results['mean_test_score'][i]}")
            print(f"Standard Deviation: {results['std_test_score'][i]}")
            print("-" * 50)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)
    print("-" * 50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV ROC AUC: {grid_search.best_score_}")
    print("-" * 50)
    print("Main Run logged.")
    print("-" * 50)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Define custom classification threshold
    classification_threshold = 0.45

    # Predictions
    # NOTE: Using the correct variable names X_train, X_test
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "train_roc_auc": roc_auc_score(y_train, y_pred_train_proba), # Log ROC AUC directly
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_roc_auc": roc_auc_score(y_test, y_pred_test_proba) # Log ROC AUC directly
    })
    #print log_mertics
    print("-" * 50)
    print("Metrics logged for the best model:")
    print("-" * 50)
    print(f"Parameters: {grid_search.best_params_}")
    print(f"Best CV ROC AUC: {grid_search.best_score_}")
    print("-" * 50)

    print("-" * 50)
    print("-- Train Metrics logged for the best model: --")
    print("-" * 50)

    print(f"Train Accuracy: {train_report['accuracy']}")
    print(f"Train Precision: {train_report['1']['precision']}")
    print(f"Train Recall: {train_report['1']['recall']}")
    print(f"Train F1-Score: {train_report['1']['f1-score']}")
    print(f"Train ROC AUC: {roc_auc_score(y_train, y_pred_train_proba)}")

    print("-" * 50)
    print("-- Test Metrics logged for the best model: --")
    print("-" * 50)
    print(f"Test Accuracy: {test_report['accuracy']}")
    print(f"Test Precision: {test_report['1']['precision']}")
    print(f"Test Recall: {test_report['1']['recall']}")
    print(f"Test F1-Score: {test_report['1']['f1-score']}")
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred_test_proba)}")
    print("-" * 50)

    # Save the model locally (using consistent filename)
    model_path = MODEL_FILENAME # Defined as "xgboost_model_pipeline.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact to MLflow (using built-in function)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="WellnessTourismXGBoostModel"
    )
    print(f"Model saved and logged to MLflow.")


# 5. HUGGING FACE DEPLOYMENT

repo_id = MODEL_REPO_ID # Using the correct variable for our project: SriniGS/tourism-package-prediction-v2
repo_type = "model"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Space '{repo_id}' created.")

# Upload the final model artifact (using consistent filename)
api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME, # Uploading the .joblib file
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Best XGBoost model pipeline deployed."
)
print(f"Successfully uploaded model to Hugging Face repo: {repo_id}")
