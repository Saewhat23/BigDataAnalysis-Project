import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

df = pd.read_csv("2018data.csv")

X = df['GW'].values.reshape(-1,1)
Y = df['Pyds'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, Y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    df['GW'],
    df['Pyds'],
    c='black'
)
plt.plot(
    df['GW'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Games Won")
plt.ylabel("Passing Yards")
plt.show()

X = df['GW']
y = df['Pyds']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())