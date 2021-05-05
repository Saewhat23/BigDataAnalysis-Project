import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# read the csv file
df = pd.read_csv('2019data.csv')

X = np.column_stack((df['Ryds'], df['Pyds'], df['Cmp']))
Y = df['GW']

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())