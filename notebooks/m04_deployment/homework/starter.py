import sys
import pickle
import pandas as pd

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def generate_result(input_file, output_file, id_prefix,dv, lr):
    df = read_data(input_file)


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['pred'] = y_pred
    print(df['pred'].mean())

    df['ride_id'] = id_prefix + df.index.astype('str')
    df_result = df[['ride_id', 'pred']].copy()

    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )
    
    return None

def run():

    # data path
    taxi_type = sys.argv[1] # 'fhv'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3

    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    id_prefix = f'{year:04d}/{month:02d}_'
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    generate_result(input_file, output_file, id_prefix,dv, lr)
    
    return None
    
if __name__=='__main__':
    run()