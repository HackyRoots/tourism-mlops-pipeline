# train.py
# Model building with experiment tracking: tunes an XGBoost classifier, logs everything to MLflow,
# evaluates the best model and saves it into tourism_project/deployment/ so the workflow can commit it.
import os
import urllib.request

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "tourism_project/deployment/best_tourism_model_v1.joblib"
THRESHOLD = 0.5

# ---- MLflow set-up: use the tracking server started by the workflow, else fall back to a local folder ----
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
try:
    urllib.request.urlopen(tracking_uri, timeout=3)
except Exception:
    print(f"No MLflow server reachable at {tracking_uri}; logging to the local ./mlruns folder instead")
    tracking_uri = "file:./mlruns"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism-package-prediction")
print("MLflow tracking URI:", tracking_uri)

# ---- Load the train / test splits produced by the data-prep job ----
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print("Train:", Xtrain.shape, " Test:", Xtest.shape)

# ---- Preprocessing: scale numeric columns, one-hot encode categorical columns ----
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical_features),
)

# ---- Model and hyperparameter grid ----
# Only ~19% of customers buy the package, so the positive class is up-weighted.
# The grid covers deeper trees because shallow trees (depth 3-5) scored clearly lower in cross-validation.
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [6, 8, 10],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [0.5, 1],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ---- Tune the model and log every tried combination to MLflow ----
with mlflow.start_run(run_name="xgboost-grid-search"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # One child run per parameter combination (parameters + cross-validated F1)
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    # Log the best parameters and the evaluation of the best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scale_pos_weight", round(float(class_weight), 3))
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    scores = {}
    for name, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        proba = best_model.predict_proba(X)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        scores[name] = {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
        }
        for metric, value in scores[name].items():
            mlflow.log_metric(f"{name}_{metric}", value)

        print(f"\n=== {name.upper()} classification report (threshold {THRESHOLD}) ===")
        print(classification_report(y, pred, target_names=["No purchase", "Purchase"]))

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated F1:", round(grid_search.best_score_, 4))
    print("\nMetric summary:")
    print(pd.DataFrame(scores).round(4).T.to_string())

    # ---- Save the best model and attach it to the MLflow run ----
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")
