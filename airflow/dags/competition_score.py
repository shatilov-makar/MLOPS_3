from airflow import DAG
import airflow
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt


args = {
    'owner': 'makar',
    'start_date':datetime(2023, 11, 21),
    'retries':1,
    'retry_delays':dt.timedelta(minutes=1),
    'depends_on_past':False
    }

with DAG(
    dag_id='competition',
    default_args=args,
    schedule_interval=None,
    tags=['competition', 'score']
) as dag:
    get_data = BashOperator(task_id='get_data', bash_command="python3 /home/makar/MLops_3/scripts/get_data.py", dag=dag)
    process_data = BashOperator(task_id='process_data', bash_command="python3 /home/makar/MLops_3/scripts/process_data.py", dag=dag)
    train_test_split_data = BashOperator(task_id='train_test_splits', bash_command="python3 /home/makar/MLops_3/scripts/train_test_split.py", dag=dag)
    train_model = BashOperator(task_id='train_model', bash_command="python3 /home/makar/MLops_3/scripts/train_model.py", dag=dag)
    test_model = BashOperator(task_id='test_model', bash_command="python3 /home/makar/MLops_3/scripts/test_model.py", dag=dag)

    get_data >> process_data >> train_test_split_data >> train_model >> test_model


