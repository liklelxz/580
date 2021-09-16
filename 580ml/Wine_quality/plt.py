import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("winequality-white.csv",delimiter=";")
plt.figure(1,figsize=(5,5))
df['quality'].value_counts().plot.pie(autopct="%1.1f&&")
print("success")
plt.show()