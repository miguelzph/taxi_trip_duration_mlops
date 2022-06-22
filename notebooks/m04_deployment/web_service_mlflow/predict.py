import pickle

from flask import Flask, request, jsonify

import mlflow
from mlflow.tracking import MlflowClient

# MLFLOW_TRACKING_URI = 'http://ec2-44-204-16-207.compute-1.amazonaws.com:5000'

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

RUN_ID = 'ce361da39ca847d7b993f11d7b1dbec6'
EXPERIMENT_ID = '2'

# model 
# logged_model = f'runs:/{RUN_ID}/model'

# if i want to get directly from s3
logged_model = f's3://mlflow-artifacts-mzph/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model'

model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    
    return features


def predict(features):
    preds = model.predict(features)
    
    return float(preds[0])


app = Flask('duration-prediction')

@app.route('/')
def hello_world():
    mensage = {'status': 'ok'}
    return jsonify(mensage)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'RUN_ID': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)