from flask import Flask,request
import pandas as pd
import pickle
from math import exp
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def adder_page():
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
    	regressor=pickle.load(open('/home/ajithkumar/Desktop/TestPrice/RandomFReg.pkl','rb'))
    	xx_columns=pickle.load(open('/home/ajithkumar/Desktop/TestPrice/xx_columns.pkl','rb'))
    	testmodel=pd.get_dummies(test)
    	missing_cols=set(xx_columns)-set(testmodel.columns)
    	for val in missing_cols:
        	testmodel[val]=0
    	testmodel=testmodel[xx_columns]
    	result=regressor.predict(testmodel)
    	result=exp(result)-1
    	return '''
       	<html>
           	<body style="background-image:url('https://wallpaperaccess.com/full/258527.jpg')">
           	<h2 style="color: black;background-color:white; margin-top:100px;padding-top:5px;padding-bottom: 5px; padding-left: 180px;">Price in $ {result}</h2>
           	</body>
       	</html>
   	'''.format(result=result)


    
    
    
    
	return '''
   	<html>
<style>
body{
background: linear-gradient(rgba(255, 255, 255, 0.2),
          	rgba(255, 255, 255, 0.2),
          	rgba(255, 255, 255, 0.2),
          	rgba(255, 255, 255, 0.2)),
          	url('https://wallpaperaccess.com/full/258527.jpg');
}
</style>
<body>
<div>
<h3 style="color: black;background-color:white; padding-top:5px;padding-bottom: 5px; padding-left: 180px;">
Pre-Owned Cars Price Prediction</h3>
<form action="." method="POST">
<div style="margin-top:4px;padding-left: 100px;padding-right: 132px;float:left"><span style="background-color:white">Model Year:</span></div>
<div> <input name="year" style="margin-top:4px;padding-left: 10px; padding-right: 0px;" name="year" type="text" /> </div>
<div style="padding-left: 100px; padding-right: 120px; float: left;"><span style="margin-top:4px;background-color:white">Manufacturer:</span></div>
<div><select name="manufacturer" style="margin-top:4px;padding-left: 40px; padding-right: 8px;">
<option value="ford">Ford</option>
<option value="chevrolet">Chevrolet</option>
<option value="toyota">Toyota</option>
<option value="nissan" selected>Nissan</option>
<option value="ram">Ram</option>
<option value="honda">Honda</option>
<option value="jeep">Jeep</option>
<option value="gmc">GMC</option>
<option value="dodge">Dodge</option>
<option value="bmw">BMW</option>
<option value="hyundai">Hyundai</option>
<option value="mercedes-benz">Mercedes-Benz</option>
<option value="subaru">Subaru</option>
<option value="volkswagen">Volkswagen</option>
<option value="kia">Kia</option>
<option value="chrysler">Chrysler</option>
<option value="cadillac">Cadillac</option>
<option value="lexus">Lexus</option>
<option value="buick">Buick</option>
<option value="mazda">Mazda</option>
<option value="audi">Audi</option>
<option value="acura">Acura</option>
<option value="infiniti">Infiniti</option>
<option value="lincoln">Lincoln</option>
<option value="pontiac">Pontiac</option>
<option value="volvo">Volvo</option>
<option value="mitsubishi">Mitsubishi</option>
<option value="mini">Mini</option>
<option value="rover">Rover</option>
<option value="mercury">Mercury</option>
<option value="saturn">Saturn</option>
<option value="jaguar">Jaguar</option>
<option value="fiat">Fiat</option>
<option value="harley-davidson">Harley-Davidson</option>
<option value="alfa-romeo">Alph-Romeo</option>
<option value="datsun">Datsun</option>
<option value="tesla">Tesla</option>
<option value="land rover">Land Rover</option>
<option value="porche">Porche</option>
<option value="aston-martin">Aston-Martin</option>
<option value="ferrari">Ferrari</option>
<option value="morgan">Morgan</option>
<option value="unknown">Unknown</option>
</select></div>
<div style="padding-left: 100px;padding-right: 143px; float: left;"><span style="margin-top:4px;background-color:white">Condition:</span></div>
<div><select name="condition" style="margin-top:4px;padding-left: 40px; padding-right: 53px;">
<option value="new">New</option>
<option value="excellent" selected>Excellent</option>
<option value="like new">Like New</option>
<option value="good">Good</option>
<option value="salvage">Salvage</option>
<option value="fair">Fair</option>
</select></div>
<div style="padding-left: 100px; padding-right: 145px; float: left;"><span style="margin-top:4px;background-color:white">Cylinders:</span></div>
<div><select  name="cylinders" style="margin-top:4px;padding-left: 40px; padding-right: 37px;">
<option value="0">3 cylinders</option>
<option value="1" selected>4 cylinders</option>
<option value="2">5 cylinders</option>
<option value="3">6 cylinders</option>
<option value="4">8 cylinders</option>
<option value="5">10 cylinders</option>
<option value="6">12 cylinders</option>
</select></div>
<div style="padding-left: 100px; padding-right: 178px; float: left;"><span style="margin-top:4px;background-color:white">Fuel:</span></div>
<div><select name="fuel" style="margin-top:4px;padding-left: 40px; padding-right: 65px;">
<option value="gas">Gas</option>
<option value="diesel" selected>Diesel</option>
<option value="hybrid">Hybrid</option>
<option value="electric">Electric</option>
<option value="other">Other</option>
</select></div>
<div style="padding-left: 100px; padding-right: 143px; float: left;"><span style="background-color:white">Odometer:</span></div>
<div><input style="margin-top:4px;padding-left: 10px; padding-right: 2px;" name="odometer" type="text" /></div>
<div style="padding-left: 100px; padding-right: 134px; float: left;"><span style="margin-top:4px;background-color:white">Title Status:<span></div>
<div>
<select name="title_status" style="margin-top:4px;padding-left: 40px; padding-right: 60px;">
<option value="clean" selected>Clean</option>
<option value="rebuilt">Rebuilt</option>
<option value="salvage">Salvage</option>
</select></div>
<div style="padding-left: 100px; padding-right: 120px; float: left;"><span style="margin-top:4px;background-color:white">Transmission:</span></div>
<div><select  name="transmission" style="margin-top:4px;padding-left: 40px; padding-right: 51px;">
<option value="automatic" selected>Automatic</option>
<option value="manual">Manual</option>
<option value="other">Other</option>
</select></div>
<div style="padding-left: 100px; padding-right: 170px; float: left;"><span style="margin-top:4px;background-color:white">Drive:</span></div>
<div>
<select name="drive" style="margin-top:4px;padding-left: 40px; padding-right: 85px;">
<option value="1wd">1wd</option>
<option value="2wd">2wd</option>
<option value="3wd">3wd</option>
<option value="4wd" selected>4wd</option>
<option value="5wd">5wd</option>
<option value="6wd">6wd</option>
<option value="7wd">7wd</option>
</select></div>

<div style="padding-left: 100px; padding-right: 174px; float: left;"><span style="margin-top:4px;background-color:white">Type:</span></div>
<div>
<select style="margin-top:4px;padding-left: 40px; padding-right: 42px;" name="vtype">
<option value="sedan">Sedan</option>
<option value="SUV">SUV</option>
<option value="truck">Truck</option>
<option value="pickup">Pickup</option>
<option value="van">Van</option>
<option value="coupe">Coupe</option>
<option value="hatchback">Hatchback</option>
<option value="wagon">Wagon</option>
<option value="convertible">Convertible</option>
<option value="bus">Bus</option>
<option value="mini-van">Mini-Van</option>
<option value="offroad">Offroad</option>
<option value="other">Other</option>
</select></div>


<div style="padding-left: 100px; padding-right: 133px; float: left;"><span style="margin-top:4px;background-color:white">Paint Color:</span></div>
<div><select name="paint_color" style="margin-top:4px;padding-left: 40px; padding-right: 63px;">
<option value="white">White</option>
<option value="black" selected>Black</option>
<option value="silver">Silver</option>
<option value="blue">Blue</option>
<option value="red">Red</option>
<option value="grey">Grey</option>
<option value="green">Green</option>
<option value="brown">Brown</option>
<option value="yellow">Yellow</option>
<option value="orange">Orange</option>
<option value="purple">Purple</option>
<option value="custom">Custom</option>
</select></div>
<div style="padding-left: 100px; padding-right: 174px; float: left;"><span style="margin-top:4px;background-color:white">State:</span></div>
<div>
<select name="state" style="margin-top:4px;padding-left: 40px; padding-right: 49px;">
<option value="midwest">MidWest</option>
<option value="south" selected>South</option>
<option value="northeast">NorthEast</option>
<option value="west">West</option>
</select></div><br>
<div style="padding-left: 100px; padding-right: 168px; float: left;"><input style="padding-left: 165px; padding-right: 175px;" type="submit" value="Submit" /></div>
</form>
</div>
</body>
</html>





	'''


if __name__ == '__main__':
	app.run()



