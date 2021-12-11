import json
import pprint
import collections
import operator
import pandas as pd
import csv
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sd = open('Dataset/crops_price_updated.csv','w')
print('﻿2003',file=sd)

csv_file = open('Dataset/crops_price_per_quintal.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
aid = {}

for row in csv_reader:
    if row[0] == "Commodities(rs/quin)":
        continue
    aid[row[0]] = row[1]
    if row[1] == "...":
     if row[2] == "...":
        continue
     else:
        a = float(row[2])
    else:
        if row[2] == "...":
           a = float(row[1])
        else:
           a = (float(row[2]) + float(row[1]))/2

    print(a,file =sd)

sd.close()

df1 = pd.read_csv("Dataset/crops_price_per_quintal.csv")
df2 = pd.read_csv("Dataset/crops_price_updated.csv",delim_whitespace=True)
df3 = pd.read_csv("Dataset/area_under_cult.csv")

df1.insert(2,"2003",df2)
df1["Commodities(rs/quin)"].replace({"Paddy (Common) ": "Rice", "Cotton H-4 750 ": "Cotton(lint)", "Jute(TD-5) ":"Jute", "Rapeseed/ mustard ":"Rapeseed &Mustard", "Soyabean (Yellow) ": "Soyabean","Sugarcane (Statutory minimum price) a" : "Sugarcane","Jowar (Hybrid)": "Jowar", "Moong ":"Moong(Green Gram)", "Arhar ": "Arhar/Tur"} , inplace=True)
# result = pd.concat([df1, df2], axis=2) # join by row, not by column
df1.to_csv("Dataset/prices_all_crops",index =False)

df = pd.DataFrame()
# print (df1.columns)
for ind in df1.index:
    crop = df1["Commodities(rs/quin)"][ind]
    for year in df1.columns:
        if year == "Commodities(rs/quin)":
            continue
        crop = str(crop.strip())
        year = str(year)
        a = df3.loc[(df3['Crop'] == crop) & (df3['Crop_Year'] == int(year))]
        try:
            area = round(float(a.iloc[0]['Area']),2)
            production = round(float(a.iloc[0]['Production']),2)
            price = round(float(df1[year][ind]),2)
            revenue = round((price*production)/area,2)
            new_row = {'Crop': crop, 'Crop_Year': year, 'Area in Hectares':area, 'Production in quintals': production, 'Price per quintal':price, "Revenue per hectare": revenue}
            df = df.append(new_row,ignore_index=True)
        except:
            pass

# print(df)
sd = open('Dataset/revenue.csv','w')
print(df.to_csv(sep = ',',index= False),file =sd)
