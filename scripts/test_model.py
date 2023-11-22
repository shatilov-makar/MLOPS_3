from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pkl
import mlflow
import os
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_percentage_error

os.environ["MLFLOW_REGISTRY_URI"] = "/home/makar/MLops_3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")


test_df = pd.read_csv('/home/makar/MLops_3/datasets/data_test.csv')

X_test, y_test = test_df.drop(['num_sold'], axis=1), test_df[['num_sold']]

with open('/home/makar/MLops_3/models/rf_regressor.pickle', 'rb') as f:
    model = pkl.load(f)

y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)
with mlflow.start_run():
    mlflow.log_metric('MAPE', mape)
    mlflow.log_artifact(local_path="/home/makar/MLops_3/scripts/test_model.py", artifact_path="test_model code")
    mlflow.end_run()
print("score=", mape)
