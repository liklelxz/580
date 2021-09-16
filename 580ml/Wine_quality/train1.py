import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('winequality-white.csv',delimiter=';')
df["quality"]=df['quality'].astype("int32")
df['quality'] = df['quality'].apply(lambda x: 'good' if x>=7 else 'bad')
X= df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 0)
sc_x =StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
print X_train
print X_test
