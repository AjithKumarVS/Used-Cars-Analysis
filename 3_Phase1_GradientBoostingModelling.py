from sklearn.model_selection import train_test_split
from sklearn import ensemble
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt

cleaned=pd.read_csv("C:/Users/avakk/Downloads/updatedcv.csv")

cleaned.drop(["url","lat","long","id","model","region"],axis=1,inplace=True)

cleaned.drop(cleaned[cleaned["title_status"]=="parts only"].index,inplace=True)
cleaned.drop(cleaned[cleaned["title_status"]=="missing"].index,inplace=True)
cleaned.drop(cleaned[cleaned["title_status"]=="lien"].index,inplace=True)

cleaned.drop(cleaned[cleaned["price"]<2000].index,inplace=True)
cleaned.drop(cleaned[cleaned["price"]>40000].index,inplace=True)

cleaned["price"]=np.log1p(cleaned["price"])

northeast=["ct","me","ma","nh","ri","vt","nj","ny","pa"]
midwest=["il","in","mi","oh","wi","ia","ks","mn","mo","ne","nd","sd"]
south=["de","fl","ga","md","nc","sc","va","wv","dc","ms","al","ky","tn","ar","tx","ok","la"]
west=["az","co","id","mt","nv","nm","ut","wy","wa","or","hi","ca","az","ak"]

for index, row in cleaned.iterrows():
    if(row["state"] in northeast):
        cleaned.at[index,"state"]="northeast"
    if(row["state"] in midwest):
        cleaned.at[index,"state"]="midwest"
    if(row["state"] in south):
        cleaned.at[index,"state"]="south"
    if(row["state"] in west):
        cleaned.at[index,"state"]="west"

yearmed=cleaned["year"].median()
cleaned["year"].fillna(yearmed)
for index, row in cleaned.iterrows():
    if(row["year"]>2020):
        cleaned.at[index,"year"]=yearmed

y=cleaned["price"]
x=cleaned.drop('price',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_odomin=x_train["odometer"].min()
x_odomax=x_train["odometer"].max()
x_train["odometer"]=((x_train["odometer"]-x_odomin)/(x_odomax-x_odomin))

x_yearmin=x_train["year"].min()
x_yearmax=x_train["year"].max()
x_train["year"]=((x_train["year"]-x_yearmin)/(x_yearmax-x_yearmin))

x_test["odometer"]=((x_test["odometer"]-x_odomin)/(x_odomax-x_odomin))
x_test["year"]=((x_test["year"]-x_yearmin)/(x_yearmax-x_yearmin))

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


def train_GB(est,max_depth,lr):
    params={'n_estimators': est, 'max_depth': max_depth,'learning_rate': lr}
    gb=ensemble.GradientBoostingRegressor(**params)
    gb_model=gb.fit(xx_train,y_train)
    y_pred=gb_model.predict(xx_test)
    ybar=np.sum(y_test)/len(y_test)
    sse = np.sum((y_test-y_pred)**2)  
    sst = np.sum((y_test - ybar)**2) 
    rsq=1-(sse/sst)
    print("{}---> NEstimators{}-MaxDepth{}-LearningRate{}".format(rsq,est,max_depth,lr))

params={'n_estimators': 500, 'max_depth': 8,'learning_rate': 0.1}
gb=ensemble.GradientBoostingRegressor(**params)
gb_model=gb.fit(xx_train,y_train)


features=pd.DataFrame(gb_model.feature_importances_,index=xx_train.columns)
features["columns"]=features.index
features.index=range(0,len(features))
features.columns=["value","columns"]

print("Feature Importances")
categorical_columns=['manufacturer','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','state']
for val in categorical_columns:
    p=(features[features["columns"].str.startswith(val)]["value"].sum())/len(features[features["columns"].str.startswith(val)]["value"])
    print("{}>>{}".format(val,p))

for index,row in features.iterrows():
    if(row["columns"]=="odometer"):
        print("Odometer>>{}".format(row["value"]))
    if(row["columns"]=="year"):
        print("Year>>{}".format(row["value"]))

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