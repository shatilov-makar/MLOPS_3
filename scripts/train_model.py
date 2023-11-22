from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pkl
import mlflow
import os
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_percentage_error

os.environ["MLFLOW_REGISTRY_URI"] = "/home/makar/MLops_3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

train_df = pd.read_csv('/home/makar/MLops_3/datasets/data_train.csv')

X_train, y_train = train_df.drop(['num_sold'], axis=1), train_df[['num_sold']]

rf_regressor = RandomForestRegressor(
    n_estimators=50, random_state=0)

with mlflow.start_run():
    mlflow.sklearn.log_model(rf_regressor, artifact_path="rf_regressor", registered_model_name="rf_regressor")
    mlflow.log_artifact(local_path="/home/makar/MLops_3/scripts/train_model.py", artifact_path="train_model code")
    mlflow.end_run()

rf_regressor.fit(X_train, y_train)
with open('/home/makar/MLops_3/models/rf_regressor.pickle', 'wb') as f:
    pkl.dump(rf_regressor, f)
