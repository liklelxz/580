from sklearn.preprocessing import StandardScaler
data =[[2,2]]
sc = StandardScaler()
print(sc.fit_transform(data))
print(sc.mean_)
