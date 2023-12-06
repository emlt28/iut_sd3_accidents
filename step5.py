import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

victime= pd.read_csv("C:/Users/emaillot/Documents/my_virtual_envs/gps_encoding.csv",sep=',',decimal='.')
y = victime['grav']

features = ['catu','sexe','trajet','secu',
            'catv','an_nais','mois',
            'occutc','obs','obsm','choc','manv',
            'lum','agg','int','atm','col','gps',
            'catr','circ','vosp','prof','plan',
            'surf','infra','situ','hrmn','geo']
X_train_data = pd.get_dummies(victime[features].astype(str))

X_train_data.to_csv('C:/Users/emaillot/Documents/my_virtual_envs/one_hot_encoding.csv', index=False)