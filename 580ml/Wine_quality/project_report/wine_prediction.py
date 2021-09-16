
import pandas as pd
import matplotlib as mpl

mpl.use('agg')
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV








wine_dataset = pd.read_csv('winequality-white.csv', delimiter=';')
###print(wine_dataset.shape)
###print(wine_dataset['quality'].value_counts())

wine_dataset["quality_value"] = wine_dataset.quality.apply(lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')
wine_dataset["quality_interval"] = wine_dataset.quality.apply(lambda r: 0 if r <= 5 else 1 if r <= 7 else 2)
###print(wine_dataset.quality_interval.value_counts())



plt.figure(1, figsize=(5,5))
wine_dataset['quality'].value_counts().plot.pie(autopct="%1.1f%%")
plt.savefig('pie1.png')
plt.clf()
wine_dataset['quality_interval'].value_counts().plot.pie(autopct="%1.1f%%")
plt.savefig('pie2.png')
plt.clf()
plt.close()



features = ['fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
            'density', 'pH', 'sulphates', 'alcohol', 'quality', 'quality_interval']
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(wine_dataset[features].corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.savefig('correlation.png')
plt.clf()
plt.close()

X= wine_dataset.iloc[:, :-3].values

Y = wine_dataset.iloc[:, -1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 43)
sc_x =StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


parameters = {"n_estimators":[1000,1500,2000],"max_features":("sqrt","log2")}

model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model,param_grid=parameters,cv=10,n_jobs = -1)
grid_search.fit(X_train,Y_train)
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
Y_pred = best_grid.predict(X_test)
print (metrics.accuracy_score(Y_test,Y_pred))

