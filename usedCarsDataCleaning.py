import pandas as pd
import seaborn as sns
used_cars=pd.read_csv("C:/Users/avakk/Downloads/vehicles.csv")  #Reading CSV
#changes by sukesh
used_cars.shape
used_cars.info()
sns.heatmap(used_cars.isnull())
used_cars.isnull().sum()
used_cars.drop(["region_url"],axis=1,inplace=True) #Does not contribute towards prediction
used_cars.drop(["county"],axis=1,inplace=True) #High percentage of null values
