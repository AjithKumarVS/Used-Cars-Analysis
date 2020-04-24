import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

cleaned=pd.read_csv("C:/Users/avakk/Downloads/updatedcv.csv")

cleaned.drop(["url","lat","long","id","model","region"],axis=1,inplace=True)

cleaned.drop(cleaned[cleaned["title_status"]=="parts only"].index,inplace=True)
cleaned.drop(cleaned[cleaned["title_status"]=="missing"].index,inplace=True)
cleaned.drop(cleaned[cleaned["title_status"]=="lien"].index,inplace=True)

cleaned.drop(cleaned[cleaned["price"]<2000].index,inplace=True)
cleaned.drop(cleaned[cleaned["price"]>40000].index,inplace=True)

cleaned["price"]=np.log1p(cleaned["price"])

ne=["ct","me","ma","nh","ri","vt","nj","ny","pa"]
mw=["il","in","mi","oh","wi","ia","ks","mn","mo","ne","nd","sd"]
s=["de","fl","ga","md","nc","sc","va","wv","dc","ms","al","ky","tn","ar","tx","ok","la"]
w=["az","co","id","mt","nv","nm","ut","wy","wa","or","hi","ca","az","ak"]

for index, row in cleaned.iterrows():
    if(row["state"] in ne):
        cleaned.at[index,"state_trans"]="ne"
    if(row["state"] in mw):
        cleaned.at[index,"state_trans"]="mw"
    if(row["state"] in s):
        cleaned.at[index,"state_trans"]="s"
    if(row["state"] in w):
        cleaned.at[index,"state_trans"]="w"

yearmed=cleaned["year"].median()
cleaned["year"].fillna(yearmed)
for index, row in cleaned.iterrows():
    if(row["year"]>2020):
        cleaned.at[index,"year"]=yearmed
        
        
website_url = requests.get("https://en.wikipedia.org/wiki/List_of_U.S._states_by_GDP_per_capita").text
soup = BeautifulSoup(website_url,"lxml")

my_table = soup.find("table",{"class":"wikitable sortable"})
my_table
names=my_table.findAll('tr')
soup = BeautifulSoup(website_url,"html.parser")
rows = []
for tr in soup.select('tr'):
    rows.append([td.get_text(strip=True) for td in tr.select('th, td')])
    
rows_up=rows[:len(rows)-77]

gdp=pd.DataFrame(rows_up,columns=["Rank","State","2018","2017","2016","2015","2014","2013","2012","2011"])

gdp.drop(["Rank"],axis=1,inplace=True)
gdp.drop(gdp.index[[0,1,23]],inplace=True)
gdp["State"]=gdp["State"].str.lower()

gdp.columns=["US States","GDP2018","GDP2017","GDP2016","GDP2015","GDP2014","GDP2013","GDP2012","GDP2011"]
gdp.index=range(1,len(gdp)+1)

columns=["GDP2018","GDP2017","GDP2016","GDP2015","GDP2014","GDP2013","GDP2012","GDP2011"]

us_state_abbrev = {
    'alabama': 'AL',
    'alaska': 'AK',
    'american samoa': 'AS',
    'arizona': 'AZ',
    'arkansas': 'AR',
    'california': 'CA',
    'colorado': 'CO',
    'connecticut': 'CT',
    'delaware': 'DE',
    'district of columbia': 'DC',
    'florida': 'FL',
    'georgia': 'GA',
    'guam': 'GU',
    'hawaii': 'HI',
    'idaho': 'ID',
    'illinois': 'IL',
    'indiana': 'IN',
    'iowa': 'IA',
    'kansas': 'KS',
    'kentucky': 'KY',
    'louisiana': 'LA',
    'maine': 'ME',
    'maryland': 'MD',
    'massachusetts': 'MA',
    'michigan': 'MI',
    'minnesota': 'MN',
    'mississippi': 'MS',
    'missouri': 'MO',
    'montana': 'MT',
    'nebraska': 'NE',
    'nevada': 'NV',
    'new hampshire': 'NH',
    'new jersey': 'NJ',
    'new mexico': 'NM',
    'new york': 'NY',
    'north carolina': 'NC',
    'north dakota': 'ND',
    'northern mariana islands':'MP',
    'ohio': 'OH',
    'oklahoma': 'OK',
    'oregon': 'OR',
    'pennsylvania': 'PA',
    'puerto rico': 'PR',
    'rhode island': 'RI',
    'south carolina': 'SC',
    'south dakota': 'SD',
    'tennessee': 'TN',
    'texas': 'TX',
    'utah': 'UT',
    'vermont': 'VT',
    'virgin islands': 'VI',
    'virginia': 'VA',
    'washington': 'WA',
    'west virginia': 'WV',
    'wisconsin': 'WI',
    'wyoming': 'WY'
}

for index,row in gdp.iterrows():
    gdp.at[index,"US States"]=us_state_abbrev[row["US States"]]

gdp["US States"]=gdp["US States"].str.lower()
gdp.columns=["state","GDP2018","GDP2017","GDP2016","GDP2015","GDP2014","GDP2013","GDP2012","GDP2011"]


updatedsecondary=pd.merge(cleaned,gdp,on="state")
updatedsecondary.to_csv(r'C:/Users/avakk/Downloads/primarysecondary.csv',index=False)
