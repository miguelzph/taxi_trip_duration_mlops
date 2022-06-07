from datetime import datetime
from dateutil.relativedelta import relativedelta

import pickle    

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow,task
from prefect import get_run_logger

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    
    logger = get_run_logger()
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    
    date_correct_format = datetime.strptime(date, "%Y-%m-%d")
    
    
    train_date = date_correct_format + relativedelta(months=-2) # 2 months
    train_month = str(train_date.month).zfill(2)
    train_year = str(train_date.year)
    
    val_date = date_correct_format + relativedelta(months=-1) # 1 months
    val_month = str(val_date.month).zfill(2)
    val_year = str(val_date.year)
    
    train_path = f'datasets/fhv_tripdata_{train_year}-{train_month}.parquet'
    val_path = f'datasets/fhv_tripdata_{val_year}-{val_month}.parquet'
    
    return train_path, val_path

@flow
def main_hm(date=None):

    categorical = ['PUlocationID', 'DOlocationID']
    
    if date is None:
        # if i want to use current date --> need to download newer files
        date="2021-03-15"
    
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    
    with open(f'models/models_prefect_section/model-{date}.bin', 'wb') as file:
        pickle.dump(lr,file)
    with open(f'models/models_prefect_section/dv-{date}.bin', 'wb') as file:
        pickle.dump(dv,file)
    
    run_model(df_val_processed, categorical, dv, lr)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    flow=main_hm,
    name="homework",
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    tags=["hm"]
)
