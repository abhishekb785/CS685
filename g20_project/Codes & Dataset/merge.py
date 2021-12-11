import json
import pprint
import collections
import operator
import pandas as pd
import csv

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter = ',')
crop_prod = crop_prod[['State_Name','District_Name','Crop_Year','Season','Crop','Area','Production']]

sd = open('Dataset/area_under_cult.csv','w')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = crop_prod.fillna(0)
df1 = df.groupby(['Crop','Crop_Year'])['Area','Production'].agg('sum').reset_index()
print(df1.to_csv(sep = ',',index= False),file =sd)

sd.close()
