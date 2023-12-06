import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score


victime= pd.read_csv("C:/Users/emaillot/Documents/my_virtual_envs/gps_encoding.csv",sep=',',decimal='.')
y = victime['grav']

features = ['catu','sexe','trajet','secu',
            'catv','an_nais','mois',
            'occutc','obs','obsm','choc','manv',
            'lum','agg','int','atm','col','gps',
            'catr','circ','vosp','prof','plan',
            'surf','infra','situ','hrmn','geo']
X_train= pd.get_dummies(victime[features].astype(str))

X_train = normalize(X_train.values)

# On divise la base en bases d'entraînements et de test :

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_train,y)

# On construit le modèle :

model_rf = RandomForestClassifier(n_estimators=100, 
                                  max_depth=8
)

# L'entrînement commence :

model_rf.fit(X_train_rf, y_train_rf)

# On a maintenant les prédictions pour la base de test

predictions_test = model_rf.predict(X_test_rf)

# On calcul de même les prédictions pour la base train

predictions_train = model_rf.predict(X_train_rf)

# Les résultats sont calculés de cette manière :

train_acc = accuracy_score(y_train_rf, predictions_train)
print(train_acc)

test_acc = accuracy_score(y_test_rf, predictions_test)
print(test_acc)

# On redécoupe la base en train/test

X_train, X_test, y_train, y_test = train_test_split(X_train,y)


# On crée le modèle :

model_boosting = GradientBoostingClassifier(loss="log_loss",
    learning_rate=0.2,
    max_depth=5,
    max_features="sqrt",
    subsample=0.95,
    n_estimators=200)

# L'entraînement débute :

model_boosting.fit(X_train, y_train)

# On calcul les prédictions
predictions_test_xgb = model_boosting.predict(X_test)
predictions_train_xgb = model_boosting.predict(X_train)

# On affiche les résultats :

train_acc = accuracy_score(y_train, predictions_train_xgb)
print(train_acc)

test_acc = accuracy_score(y_test, predictions_test_xgb)
print(test_acc)
