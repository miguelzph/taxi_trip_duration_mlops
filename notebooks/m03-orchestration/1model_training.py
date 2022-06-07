import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def read_data(filename):
    df = pd.read_parquet(filename)
    
    categories_columns = ['PULocationID', 'DOLocationID']
    df[categories_columns] = df[categories_columns].astype('str')
    
    df['duration'] = df['lpep_dropoff_datetime'] -  df['lpep_pickup_datetime']
    
    df['duration'] = (df['duration'].dt.seconds /60).astype('float64').round(4) # duration in minutes
    
    # just o get the same result --> but is probably wrong do this in validation data
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    return df


def select_data(data):
    
    data['PU_DO'] = data['PULocationID'] + '_' +data['DOLocationID']
    
    categories_columns = ['PU_DO']
    #categories_columns = ['PULocationID', 'DOLocationID']
    numerical_columns = ['trip_distance']
    
    dict_data = data[categories_columns + numerical_columns].to_dict(orient='records')
    
    return dict_data

def get_train_and_validation_data(train_path='datasets/green_tripdata_2021-01.parquet',
         val_path='datasets/green_tripdata_2021-02.parquet'):
    
    # train
    df_train = read_data(train_path)
    y_train = df_train['duration']
    
    dict_train = select_data(df_train)
    dv = DictVectorizer()
    x_train = dv.fit_transform(dict_train)
    
    # validation
    df_validation = read_data(val_path)
    y_validation = df_validation['duration']
    
    dict_validation = select_data(df_validation)
    x_validation = dv.transform(dict_validation)
    
    return x_train, x_validation, y_train, y_validation, dv


###############################
## MODELLING


# # define model
# lr = LinearRegression()
# # train model
# lr.fit(x_train, y_train)
# # predict on validation
# y_predict = lr.predict(x_validation)

# rmse = mean_squared_error(y_predict, y_validation, squared=False)
# print(f'Mean Squared Errror: {rmse}')

# print(f'Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_predict, y_validation)}')


# with open('models/lin_reg.bin', 'wb') as file:
#     pickle.dump((dv, lr),file) # for now linear regression

# with mlflow.start_run(): 

#     mlflow.set_tag("developer", "miguel")

#     mlflow.log_param("train-data-path", train_path)
#     mlflow.log_param("validation-data-path", val_path)

#     alpha = 0.01

#     mlflow.log_param("alpha", alpha)

#     # define model
#     lr = Lasso(alpha)
#     # train model
#     lr.fit(x_train, y_train)
#     # predict on validation
#     y_predict = lr.predict(x_validation)

#     rmse = mean_squared_error(y_predict, y_validation, squared=False)

#     mlflow.log_metric("rmse", rmse)
#     print(f'Mean Squared Errror: {rmse}')

#     print(f'Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_predict, y_validation)}')
    
#     mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")


def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_validation, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}


    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    
    return best_result


def train_best_model(train, valid, y_validation, dv):
    with mlflow.start_run():
        
        mlflow.set_tag("model", "xgboost")
        
        best_params = {'learning_rate':0.26171580843449055 ,
                'max_depth':17 ,
                'min_child_weight':int(1.8073685780167572) ,
                'objective': 'reg:linear',
                'reg_alpha': '0.20860267861726436',
                'reg_lambda': '0.3579871775866023',
                'seed':42
            }
        
        mlflow.log_params(best_params)

        booster = xgb.train(
                    params=best_params,
                    dtrain=train,
                    num_boost_round=1000,
                    evals=[(valid, 'validation')],
                    early_stopping_rounds=50
                )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_validation, y_pred, squared=False)
        
        mlflow.log_metric("rmse", rmse)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        
        mlflow.xgboost.log_model(booster, artifact_path='models_mlflow')

if __name__ == "__main__":
    x_train, x_validation, y_train, y_validation, dv = get_train_and_validation_data()

    train = xgb.DMatrix(x_train, label=y_train)
    valid = xgb.DMatrix(x_validation, label=y_validation)

    best_result = train_model_search(train, valid, y_validation)
    print(best_result)

    train_best_model(train, valid, y_validation, dv)


