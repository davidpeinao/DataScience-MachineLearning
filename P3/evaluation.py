# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('clean.csv')
test = pd.read_csv('clean_tst.csv')
train_y = pd.read_csv('water_pump_tra_target.csv')

# first column is useless
train.drop('Unnamed: 0', axis=1, inplace=True)
test.drop('Unnamed: 0', axis=1, inplace=True)
target=train.pop('status_group')

train_y.drop(labels=['id'], axis=1,inplace = True)


train = train.astype(str).apply(LabelEncoder().fit_transform)
test = test.astype(str).apply(LabelEncoder().fit_transform) # categoric features into numeric

X = train.values
X_tst = test.values
y = np.ravel(train_y.values)


#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=75573069)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("cv done")

    return modelo, y_test_all
#------------------------------------------------------------------------
    
#print("------ XGB...")
#clf = XGBClassifier(n_estimators = 200)
#clf = XGBClassifier( max_depth = 7, learning_rate  = .3, n_jobs = 3, colsample_bytree = .4)
print("------ RFC...")
clf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=-1)                      

#print("------ ADA...")
#clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(min_samples_split=6, n_estimators=1000, oob_score=True,random_state=1, n_jobs=-1), n_estimators=500)

clf, y_test_clf = validacion_cruzada(clf,X,y,skf)

clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("Final score: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst = clf.predict(X_tst)

df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission.csv", index=False)








