import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os

os.environ["MLFLOW_REGISTRY_URI"] = "/home/makar/MLops_3/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

# df_full = pd.read_csv(
#     'https://raw.githubusercontent.com/shatilov-makar/MLops/main/train%20(2).csv', delimiter=',')

#df_full.to_csv('/home/makar/MLops_3/datasets/data.csv')

with mlflow.start_run():
    df_full = pd.read_csv(
        'https://raw.githubusercontent.com/shatilov-makar/MLops/main/train%20(2).csv', delimiter=',')
    mlflow.log_artifact(local_path="/home/makar/MLops_3/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

df_full.to_csv('/home/makar/MLops_3/datasets/data.csv', index='Row_id')
