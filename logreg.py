import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv("2018data.csv")

X = df['Ryds'].values.reshape(-1,1)
Y = df['Wseason'].values.reshape(-1,1)

lgr = LogisticRegression()
lgr.fit(X, np.ravel(Y.astype(int)))
predictions = lgr.predict(X)

sns.regplot(x='Ryds', y='Wseason', data=df, logistic=True)

plt.xlabel('Turnovers')
plt.ylabel('Winning Season? (1:yes, 0:no)')
plt.show()