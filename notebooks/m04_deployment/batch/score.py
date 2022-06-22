import sys

import uuid
import pandas as pd

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


def apply_model(input_file, output_file, run_id, experiment_id='2'):

    print(f'Reading data from {input_file}...')
    df = read_dataframe(input_file)

    dicts = prepare_dictionaries(df)
    
    print(f'Loading and applying model run id: {run_id}...')
    model = load_model(run_id, experiment_id)

    y_pred = model.predict(dicts)
    
    print(f'Saving results in {output_file}...')
    df_result = prepare_result(df, y_pred, run_id)
    
    df_result.to_parquet(output_file, index=False)
    
    return None

def run():
    
    # data path
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3

    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    # model path
    RUN_ID = 'ce361da39ca847d7b993f11d7b1dbec6'
    EXPERIMENT_ID = '2'

    apply_model(input_file, output_file, RUN_ID, EXPERIMENT_ID)

if __name__ == '__main__':
    run()