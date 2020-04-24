import pandas as pd
import numpy as np
import seaborn as sns
used_cars=pd.read_csv("C:/Users/avakk/Downloads/vehicles.csv")  #Reading CSV


ndim = used_cars.ndim 
print(ndim)
used_cars.info()
print(used_cars.isnull().sum())
sns.heatmap(used_cars.isnull())
print(used_cars.isnull().sum() * 100 / len(used_cars))
pd.set_option("display.max.columns", None)
print(used_cars.describe())
print(used_cars.describe(include=np.object))
