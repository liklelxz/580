import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

df = pd.read_csv('winequality-white.csv',delimiter=';')
df["quality"]=df['quality'].astype("int32")
df['quality'] = df['quality'].apply(lambda x: 'low' if x<=5 elif x<=7 "medium" else "good" )
X= df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 0)
sc_x =StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

parameters = {"n_estimators":[1500,2000,2500],
              "max_features":("sqrt","log2")}

model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model,param_grid=parameters,cv=10,n_jobs=-1)
grid_search.fit(X_train,Y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
Y_pred =best_grid.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred))

#print('Improvement of {:0.2f}%.'.format(grid_accuracy))
