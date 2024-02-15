import os
import requests
import numpy as np
import pandas as pd
import json
from sklearn import datasets
#terminal : pip install scikit-learn 
token = 'dapie37cb7501c9bca4e57b5c9217ada8f03-2'
iris_sklearn= datasets.load_iris()
iris_pandas= pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}
def score_model(dataset):
    url = 'https://adb-3930305076644597.17.azuredatabricks.net/serving-endpoints/token/invocations'
    headers = {'Authorization': f'Bearer {token}',
               'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

print(score_model(iris_pandas))
