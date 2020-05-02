from flask import Flask,request, render_template
import pandas as pd
import pickle
import numpy as np
from math import exp


app = Flask(__name__)

@app.route('/gdp')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method=="POST":
        year=int(request.form["year"])
        manufacturer=request.form["manufacturer"]
        condition=request.form["condition"]
        cylinders=int(request.form["cylinders"])
        fuel=request.form["fuel"]
        odometer=int(request.form["odometer"])
        title_status=request.form["title_status"]
        transmission=request.form["transmission"]
        drive=request.form["drive"]
        vtype=request.form["vtype"]
        paint_color=request.form["paint_color"]
        state=request.form["state"]
        gdpup=pickle.load(open(r'E:/DPA_Project/Saved_Models/GDPforsecondmodel.pkl','rb'))
        gdptodataframe=gdpup[gdpup["state"]==state]
        for index,row in gdptodataframe.iterrows():
            gdptodataframe.at[index,"GDP2018"]=row["GDP2018"].replace(",","")
            gdptodataframe.at[index,"GDP2017"]=row["GDP2017"].replace(",","")
            gdptodataframe.at[index,"GDP2016"]=row["GDP2016"].replace(",","")
        test_data={"year":[year],"manufacturer":[manufacturer],"condition":[condition],"cylinders":[cylinders],
            "fuel":[fuel],"odometer":[odometer],"title_status":[title_status],"transmission":[transmission],
            "drive":[drive],"type":[vtype],"paint_color":[paint_color],"state_trans":[gdptodataframe.iloc[0]["state_trans"]],
            "GDP2018":[int(gdptodataframe.iloc[0]["GDP2018"])],"GDP2017":[int(gdptodataframe.iloc[0]["GDP2017"])],"GDP2016":[int(gdptodataframe.iloc[0]["GDP2016"])]}
        test=pd.DataFrame(test_data)
        test["year"]=((test["year"]-1900)/(2020-1900))
        test["odometer"]=((test["odometer"]-0)/(10000000-0))
        test["cylinders"]=((test["cylinders"]-0)/(6-0))
        regressor=pickle.load(open('E:/DPA_Project/Saved_Models/RandomFRegPh2.pkl','rb'))
        xx_columns=pickle.load(open('E:/DPA_Project/Saved_Models/xx_columnsph2.pkl','rb'))
        testmodel=pd.get_dummies(test)
        missing_cols=set(xx_columns)-set(testmodel.columns)
        for val in missing_cols:
            testmodel[val]=0
        testmodel=testmodel[xx_columns]
        result=regressor.predict(testmodel)
        result=np.expm1(result)
        return render_template('index.html', prediction_text='The best price is $ {}'.format(result))
    	

if __name__ == "__main__":
    app.run(debug=True)