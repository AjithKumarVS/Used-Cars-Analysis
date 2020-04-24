from sklearn.model_selection import train_test_split
from sklearn import ensemble
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
cleaned=pd.read_csv("C:/Users/avakk/Downloads/primarysecondary.csv")
cleaned.drop(["state","GDP2013","GDP2012","GDP2011"],axis=1,inplace=True)

for index,row in cleaned.iterrows():
    cleaned.at[index,"GDP2018"]=row["GDP2018"].replace(",","")
    cleaned.at[index,"GDP2017"]=row["GDP2017"].replace(",","")
    cleaned.at[index,"GDP2016"]=row["GDP2016"].replace(",","")
    cleaned.at[index,"GDP2015"]=row["GDP2015"].replace(",","")
    cleaned.at[index,"GDP2014"]=row["GDP2014"].replace(",","")
    
tonumeric=["GDP2018","GDP2017","GDP2016","GDP2015","GDP2014"]

for val in tonumeric:
    cleaned[val]=pd.to_numeric(cleaned[val])

y=cleaned["price"]
x=cleaned.drop('price',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

numeric=["odometer","year","GDP2018","GDP2017","GDP2016","GDP2015","GDP2014"]

for val in numeric:
    x_min=x_train[val].min()
    x_max=x_train[val].max()
    x_train[val]=((x_train[val]-x_min)/(x_max-x_min))
    x_test[val]=((x_test[val]-x_min)/(x_max-x_min))
    
ordinal_columns=["cylinders"]

for col in ordinal_columns:
     le = LabelEncoder()
     le.fit_transform(list(x_train[col].astype(str).values))
     x_train[col] = le.transform(list(x_train[col].astype(str).values))

for col in ordinal_columns:
    le.fit(list(x_test[col].astype(str).values))
    x_test[col] = le.transform(list(x_test[col].astype(str).values))
    
xx_train=pd.get_dummies(x_train)
xx_test=pd.get_dummies(x_test)

missing_cols=set(xx_train.columns)-set(xx_test.columns)
for val in missing_cols:
    xx_test[val]=0
    
xx_test=xx_test[xx_train.columns]

params={'n_estimators': 500, 'max_depth': 8,'learning_rate': 0.1}
gb=ensemble.GradientBoostingRegressor(**params)
gb_model=gb.fit(xx_train,y_train)

y_pred=gb_model.predict(xx_test)
mse = mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
rmse
print("Root Mean Squared Error Test={}".format(rmse))

ybar=np.sum(y_test)/len(y_test)
sse = np.sum((y_test-y_pred)**2)  
sst = np.sum((y_test - ybar)**2) 
rsq=1-(sse/sst)
print("R-Squared Test={}".format(rsq))

y_predtr=gb_model.predict(xx_train)

mse = mean_squared_error(y_train,y_predtr)
rmse=sqrt(mse)
rmse
print("Root Mean Squared Error Train={}".format(rmse))

ybar=np.sum(y_train)/len(y_train)
sse = np.sum((y_train-y_predtr)**2)  
sst = np.sum((y_train - ybar)**2) 
rsq=1-(sse/sst)
print("R-Squared={} Train".format(rsq))