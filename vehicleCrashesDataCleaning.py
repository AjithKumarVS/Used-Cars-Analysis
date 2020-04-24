import pandas as pd

crashes=pd.read_csv("C:/Users/avakk/Downloads/Motor_Vehicle_Collisions_-_Crashes.csv")
crashes.info()

crashes.drop(["ZIP CODE","LOCATION","ON STREET NAME","CROSS STREET NAME","OFF STREET NAME","CONTRIBUTING FACTOR VEHICLE 1","CONTRIBUTING FACTOR VEHICLE 2","CONTRIBUTING FACTOR VEHICLE 3","CONTRIBUTING FACTOR VEHICLE 4","CONTRIBUTING FACTOR VEHICLE 5","COLLISION_ID","VEHICLE TYPE CODE 2","VEHICLE TYPE CODE 3","VEHICLE TYPE CODE 4","VEHICLE TYPE CODE 5"],axis=1,inplace=True)

crashes["TOTAL INJURIES/DEATHS"]=crashes["NUMBER OF PERSONS INJURED"]+crashes["NUMBER OF PERSONS KILLED"]+crashes["NUMBER OF PEDESTRIANS INJURED"]+crashes["NUMBER OF PEDESTRIANS KILLED"]+crashes["NUMBER OF CYCLIST INJURED"]+crashes["NUMBER OF CYCLIST KILLED"]+crashes["NUMBER OF MOTORIST INJURED"]+crashes["NUMBER OF MOTORIST KILLED"]

crashes.drop(["NUMBER OF PERSONS INJURED","NUMBER OF PERSONS KILLED","NUMBER OF PEDESTRIANS INJURED","NUMBER OF PEDESTRIANS KILLED","NUMBER OF CYCLIST INJURED","NUMBER OF CYCLIST KILLED","NUMBER OF MOTORIST INJURED","NUMBER OF MOTORIST KILLED"],axis=1,inplace=True)

crashes.columns=["Crash Date","Crash Time","Borough","Latitude","Longitude","Vehicle Type","Total Injuries/Deaths"]

crashes.info()
crashes.isnull().sum()

crashes["Crash Date"]=pd.to_datetime(crashes["Crash Date"])

for index, row in crashes.iterrows():
    crashes.at[index,"Year"]=str(row["Crash Date"]).split("-")[0]
    
crashes["Year"].value_counts()
    
crashes["Year"]=pd.to_numeric(crashes["Year"])

updated=crashes[crashes["Year"]>2014]

