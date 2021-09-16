import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics



def wine_filter(inputf, t_size=0.2):
    dtset = pd.read_csv(inputf, delimiter=";", index_col=None)
    X = dtset.iloc[:, :-1].values
    Y = dtset.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size, random_state=3)
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test =  wine_filter('winequality-white.csv',0.2)
clf = RandomForestClassifier(n_estimators=1000,random_state=2000)
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)
print (metrics.accuracy_score(Y_test,Y_pred))