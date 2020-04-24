import pandas as pd
import re
import requests
used_cars=pd.read_csv("C:/Users/avakk/Downloads/vehicles.csv")  #Reading CSV

used_cars.drop(["region_url","image_url"],axis=1,inplace=True) #No contrbution towards prediction
used_cars.drop(["county","vin","lat","long"],axis=1,inplace=True) #Null values

def isNaN(string):
    return string != string

def most_frequent(test_list): 
    counter = 0
    num = test_list[0] 
    
    for i in test_list: 
        curr_frequency = test_list.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 

for index, row in used_cars.iterrows():
    if isNaN(row["drive"]):
        regexp_drive=r"[0-9]WD|[0-9]wd|fwd|FWD|rwd|RWD"
        if isNaN(row["description"]):
            pass
        else:
            drive=re.findall(regexp_drive,row["description"])
            if len(drive)!=0:
                used_cars.at[index,"drive"]=drive[0].lower()   
               
    if isNaN(row["transmission"]):
        regexp_trans=r"[A|a]utomatic|[M|m]anual"
        if isNaN(row["description"]):
            pass
        else:
            trans=re.findall(regexp_trans,row["description"])
            if len(trans)!=0:
                used_cars.at[index,"transmission"]=trans[0].lower() 

    if isNaN(row["year"]):
        regexp_year=r"19[0-9][0-9]|20[0-9][0-9]"
        if isNaN(row["description"]):
            pass
        else:
            years=re.findall(regexp_year,row["description"])
            if len(years)!=0:
                used_cars.at[index,"year"]=most_frequent(years) 

    if isNaN(row["fuel"]):
        regexp_fuel=r"[D|d]iesel|[G|g]as|[H|h]ybrid|[E|e]lectric"
        if isNaN(row["description"]):
            pass
        else:
            fuel=re.findall(regexp_fuel,row["description"])
            if len(fuel)!=0:
                used_cars.at[index,"fuel"]=fuel[0].lower()   
                
    if isNaN(row["type"]):
        regexp_type=r"[S|s]edan|[P|p]ickup|[T|t]ruck|[C|c]oupe|[H|h]atchback|[W|w]agon|[V|v]an|[C|c]onvertible|[M|m]ini-van|[O|o]ffroad|[B|b]us"
        if isNaN(row["description"]):
            pass
        else:
            typecar=re.findall(regexp_type,row["description"])
            if len(typecar)!=0:
                used_cars.at[index,"type"]=typecar[0].lower()  

used_cars["odometer"] = used_cars.groupby('year')['odometer'].apply(lambda x: x.fillna(x.mean()))
used_cars["odometer"] = used_cars["odometer"].fillna(method="ffill")

used_cars[used_cars["condition"]=="new"]["condition"]="like new"

odo_mean_excellent=used_cars[used_cars['condition'] == 'excellent']['odometer'].mean()
odo_mean_good=used_cars[used_cars['condition'] == 'good']['odometer'].mean()
odo_mean_salvage=used_cars[used_cars['condition'] == 'salvage']['odometer'].mean()
odo_mean_fair=used_cars[used_cars['condition'] == 'fair']['odometer'].mean()
odo_mean_like_new=used_cars[used_cars['condition'] == 'like new']['odometer'].mean()

used_cars.loc[used_cars['odometer'] <= odo_mean_like_new, 'condition'] = used_cars.loc[used_cars['odometer'] <= odo_mean_like_new, 'condition'].fillna('like new')

used_cars.loc[used_cars['odometer'] >= odo_mean_fair, 'condition'] = used_cars.loc[used_cars['odometer'] >= odo_mean_fair, 'condition'].fillna('fair')

used_cars.loc[((used_cars['odometer'] > odo_mean_like_new) & 
       (used_cars['odometer'] <= odo_mean_excellent)), 'condition'] = used_cars.loc[((used_cars['odometer'] > odo_mean_like_new) & 
       (used_cars['odometer'] <= odo_mean_excellent)), 'condition'].fillna('excellent')

used_cars.loc[((used_cars['odometer'] > odo_mean_excellent) & 
       (used_cars['odometer'] <= odo_mean_good)), 'condition'] = used_cars.loc[((used_cars['odometer'] > odo_mean_excellent) & 
       (used_cars['odometer'] <= odo_mean_good)), 'condition'].fillna('good')

used_cars.loc[((used_cars['odometer'] > odo_mean_good) & 
       (used_cars['odometer'] <= odo_mean_fair)), 'condition'] = used_cars.loc[((used_cars['odometer'] > odo_mean_good) & 
       (used_cars['odometer'] <= odo_mean_fair)), 'condition'].fillna('salvage')


columns=["cylinders","transmission","fuel","year","drive","title_status","paint_color","type"]
for i in columns:
    used_cars[i]=used_cars[i].fillna(used_cars[i].value_counts().index[0])

used_cars["cylinders"]=used_cars["cylinders"].str.replace('cylinders','')
for index, row in used_cars.iterrows():
    if(row["cylinders"]=="other"):
        used_cars.at[index,"cylinders"]=used_cars["cylinders"].value_counts().index[0]

convert_dict={
        "cylinders":int
        }
used_cars=used_cars.astype(convert_dict)


for index, row in used_cars.iterrows():
    if(row["drive"]=="fwd"):
        used_cars.at[index,"drive"]="2wd"
    if(row["drive"]=="rwd"):
        used_cars.at[index,"drive"]="2wd"

used_cars["drive"]=used_cars["drive"].str.replace('wd','')


convert_dict={
        "drive":int
        }
used_cars=used_cars.astype(convert_dict)

used_cars["manufacturer"]=used_cars["manufacturer"].fillna("unknown")
used_cars["model"]=used_cars["model"].fillna("unknown")


GOOGLE_API_KEY =""

def extract_lat_long_via_address(address):
    lat, lng = None, None
    api_key = GOOGLE_API_KEY
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?address={address}&key={api_key}"
    r = requests.get(endpoint)
    if r.status_code not in range(200, 299):
        return None, None
    try:
        results = r.json()['results'][0]
        lat = results['geometry']['location']['lat']
        lng = results['geometry']['location']['lng']
    except:
        pass
    return lat, lng

used_cars["region"]=used_cars["region"].str.replace('-oshkosh-FDL','')
uni_region=pd.DataFrame(used_cars["region"].value_counts().rename_axis('unique_values').reset_index(name='counts'))

uni_region.drop(["counts"],axis=1,inplace=True)

for index, row in uni_region.iterrows():
    uni_region.at[index,"lat"]=extract_lat_long_via_address(row["unique_values"])[0]
    uni_region.at[index,"long"]=extract_lat_long_via_address(row["unique_values"])[1]

uni_region.columns=["region","lat","long"]

used_cars_up=used_cars.merge(uni_region, on="region", how="left")

used_cars_up.drop(["description","size"],axis=1,inplace=True)

used_cars_up.drop(used_cars_up[used_cars_up["price"]==0].index,inplace=True)
used_cars_up.to_csv(r'C:/Users/avakk/Downloads/updatedcleanedvehicles.csv',index=False)