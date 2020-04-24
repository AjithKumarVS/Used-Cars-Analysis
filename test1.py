from flask import Flask,request, render_template
import pandas as pd
import pickle
import numpy as np
from math import exp


app = Flask(__name__)

@app.route('/p')
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
        test_data={"year":[year],"manufacturer":[manufacturer],"condition":[condition],"cylinders":[cylinders],
            "fuel":[fuel],"odometer":[odometer],"title_status":[title_status],"transmission":[transmission],
            "drive":[drive],"type":[vtype],"paint_color":[paint_color],"state":[state]}
        test=pd.DataFrame(test_data)
        test["year"]=((test["year"]-1900)/(2020-1900))
        test["odometer"]=((test["odometer"]-0)/(10000000-0))
        test["cylinders"]=((test["cylinders"]-0)/(6-0))
        regressor = pickle.load(open('E:/DPA_Project/Saved_Models/RandomFReg.pkl', 'rb'))
        xx_columns=pickle.load(open('E:/DPA_Project/Saved_Models/xx_columns.pkl','rb'))
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