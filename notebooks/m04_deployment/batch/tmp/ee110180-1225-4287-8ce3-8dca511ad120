import sys

import uuid
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import mlflow


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    return df


def prepare_dictionaries(df: pd.DataFrame):
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    
    numerical = ['trip_distance']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id, experiment_id):
    logged_model = f's3://mlflow-artifacts-mzph/{experiment_id}/{run_id}/artifacts/model'
    
    model = mlflow.pyfunc.load_model(logged_model)
    
    return model


def prepare_result(df, y_pred, run_id):  
    df_result = df[['ride_id', 'lpep_pickup_datetime', 'PULocationID', 'DOLocationID']].copy()
    df_result['predicted_duration'] = y_pred
    df_result['actual_duration'] = df['duration']
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    return df_result


@task
def apply_model(input_file, output_file, run_id, experiment_id='2'):
    
    logger = get_run_logger()

    logger.info(f'Reading data from {input_file}...')
    df = read_dataframe(input_file)

    logger.info(f'Preparing data...')
    dicts = prepare_dictionaries(df)
    
    logger.info(f'Loading model run id: {run_id}...')
    model = load_model(run_id, experiment_id)

    logger.info(f'Applying model...')
    y_pred = model.predict(dicts)
    
    logger.info(f'Preparing results...')
    df_result = prepare_result(df, y_pred, run_id)
    
    logger.info(f'Saving results in {output_file}...')
    df_result.to_parquet(output_file, index=False)
    
    return output_file


def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month
    
    input_file = f's3://nyc-tlc/trip data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f's3://prefect-deploy-taxi-duration/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet'
    
    return input_file, output_file


@flow
def ride_duration_prediction(taxi_type: str, 
                             run_id: str,
                             run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)
    
    EXPERIMENT_ID = '2'

    apply_model(input_file, output_file, run_id, EXPERIMENT_ID)
    

def run():
    # data path
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    run_id = sys.argv[4]  # 'ce361da39ca847d7b993f11d7b1dbec6'
    
    ride_duration_prediction(taxi_type=taxi_type, 
                             run_id=run_id, 
                             run_date=datetime(year=year, month=month, day=1))

if __name__ == '__main__':
    run()