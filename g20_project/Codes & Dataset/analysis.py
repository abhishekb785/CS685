
# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/My Drive"

import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import seaborn as sns #Need to install
import scipy.stats as stats
import pylab
import warnings
import math
# import keras
from sklearn.linear_model import LinearRegression
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

literacy_df = {}
with open("Dataset/literacy.txt") as f:
    for line in f:
        x = line[:-1].split("\t")
        if len(x)==7:
            literacy_df[(x[1],x[2])] = x[-1]

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter = ',')
crop_prod = crop_prod[['District_Name','State_Name','Crop','Crop_Year','Production','Area']]
crop_prod['Production/Area'] = crop_prod.apply(lambda row: row['Production']/row['Area'] if row['Production']==row['Production'] and row['Area']!=0 and row['Area']==row['Area'] else np.nan,axis=1)
# print(crop_prod[crop_prod['Production'].isnull() | crop_prod['Area'].isnull() | (crop_prod['Area']==0)])

crop_districts = set(crop_prod['District_Name'].tolist())
crop_states = set(crop_prod['State_Name'].tolist())

literacy_districts = set([x[0] for x in literacy_df.keys()])
literacy_states = set([x[1] for x in literacy_df.keys()])

def editDistDP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j    
            elif j == 0:
                dp[i][j] = i    
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1+min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])    
 
    return dp[m][n]

crop_lit_state = {}
crop_lit_district = {}
for st1 in crop_states:
    mn = 3
    mn_state = ""
    for st2 in literacy_states:
        if st1.lower()==st2.lower():
            crop_lit_state[st1] = st2
        else:
            x = editDistDP(st1.lower(),st2.lower(),len(st1),len(st2))
            if x<=mn:
                mn = x 
                crop_lit_state[st1]=st2
crop_lit_state['Madhya Pradesh'] = 'Madhya Pradesh' 

for st1 in crop_districts:
    mn = 3
    mn_state = ""
    for st2 in literacy_districts:
        if st1.lower()==st2.lower():
            crop_lit_district[st1] = st2
            break
        else:
            x = editDistDP(st1.lower(),st2.lower(),len(st1),len(st2))
            if x<=mn:
                mn = x 
                crop_lit_district[st1]=st2

# print(crop_lit_district)
# print(len(crop_lit_district))
# print(set(crop_districts)-set(crop_lit_district.keys()))
# print(set(literacy_districts)-set(crop_lit_district.values()))

'''
BEMETARA': 'Jamtara
KONDAGAON': 'Bongaigaon
KHANDWA': 'Bhandara'
AMETHI': 'Amreli
'AMROHA': 'Amreli'
'SHAMLI': 'Shimla'
'PALGHAR': 'Sagar'
'MUMBAI': 'Samba'
SAMBHAL': 'Samba
HAPUR': 'Rampur'
'KASGANJ': 'Kannauj'
'NORTH GARO HILLS': 'South Garo Hills'
'HOWRAH': 'Haora'
'BALOD': 'Jalor'
'KADAPA': 'Nawada'
'KHOWAI': 'Kota
'GOMATI': 'Mohali'
'LONGDING': 'Longleng'
'HATHRAS': 'Chatra'
'DANG': 'Durg'
'GARIYABAND': 'Faridabad'
'''
hard_map = {'SPSR NELLORE' : 'Sri Potti Sriramulu Nellore', '24 PARAGANAS SOUTH':'South Twenty Four Parganas','24 PARAGANAS NORTH' : 'North Twenty Four Parganas'
,'BENGALURU URBAN' : 'Bangalore','HOOGHLY' : 'Hugli','DANG':'The Dangs'}
multi_hard_map = {'Mumbai' : ['Mumbai City','Mumbai Suburban']}
unmapped = ['GARIYABAND','HATHRAS','LONGDING','GOMATI','KHOWAI','KADAPA','BALOD','HOWRAH','NORTH GARO HILLS','KASGANJ','HAPUR','SAMBHAL','PALGHAR','SHAMLI','AMROHA','AMETHI','KHANDWA','BEMETARA','KONDAGAON']

for i in unmapped:
    del crop_lit_district[i]

for k,v in hard_map.items():
    crop_lit_district[k] = v

for k,v in multi_hard_map.items():
    crop_lit_district[k] = v



# print(crop_prod[crop_prod['Production/Area']>1000])

# print(crop_prod['Production/Area'].corr(crop_prod['Literacy']))
# crop_prod.to_csv('crop_production_lit.csv')
def func(row):
    if row['Literacy'] > 85:
        return 4
    elif row['Literacy']>70:
        return 3
    elif row['Literacy']>50:
        return 2
    elif row['Literacy']>0:
        return 1
crops = set(crop_prod['Crop'].tolist())
crop_corr = {}
for crop in crops:
    # print(crop)
    try:
        crop1_prod = crop_prod[crop_prod['Crop']==crop].reset_index(drop=True)
        crop1_prod = crop1_prod[crop1_prod['Crop_Year']==2011].reset_index(drop=True)
        crop1_prod['Literacy'] = crop1_prod.apply(lambda row : float(literacy_df[(crop_lit_district[row['District_Name']],crop_lit_state[row['State_Name']])]) if row['District_Name'] in crop_lit_district and row['State_Name'] in crop_lit_state and (crop_lit_district[row['District_Name']],crop_lit_state[row['State_Name']]) in literacy_df else np.nan,axis=1)
        crop1_prod['Literacy_range'] = crop1_prod.apply(lambda row : func(row),axis=1)
        crop1_prod = crop1_prod[(~crop1_prod['Production'].isnull()) & (~crop1_prod['Literacy'].isnull())] 
        crop_corr[crop] = crop1_prod['Production/Area'].corr(crop1_prod['Literacy'])

        
        # if crop_corr[crop]>0.2 or crop_corr[crop]<-0.2:
        print(crop)
        sns.boxplot(x="Literacy_range", y="Production/Area", data=crop1_prod,showfliers=False)
        plt.title("crop "+crop+"\ncorrelation is : "+str(crop_corr[crop])) 
        plt.show()
            # plt.plot(crop1_prod['Production/Area'],crop1_prod['Literacy'],'ro')
            
            # plt.show()
    except:
        continue
        # print(crop)
    # plt.plot(crop1_prod['Production/Area'],crop1_prod['Literacy'],'ro')
    # plt.show()

print(crop_corr)

"""**Merging dataset of production,temperature, rainfall**"""

def get_rain_category(x,min1,max1,category_max = 5):
	diff = float(max1-min1)/category_max
	if x==x:
		return int(float(x-min1)/diff+1)
	else:
		return np.NaN

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter = ',')

major_crops = ['Arhar/Tur','Bajra','Barley','Cotton(lint)','Gram','Groundnut','Jowar','Jute','Maize','Moong','Niger seed','Ragi','Rice','Safflower','Sesamum','Soyabean','Sugarcane','Sunflower','Urad','Wheat']

# crop_prod = crop_prod[crop_prod["Crop"]=="Rice"]
# crop_prod["production/Area"] = crop_prod.apply(lambda row : row["Production"]/row["Area"]*100 if row["Area"]!=0 else np.NaN,axis=1)
# crop_prod = crop_prod.drop(columns = ['Production'],axis=0)
# min1 = crop_prod["production/Area"].min()
# max1 = crop_prod["production/Area"].max()
# crop_prod["prod_category"] = crop_prod.apply(lambda row : get_prod_category(row["production/Area"],min1,max1,100),axis=1)
# print(crop_prod)

rainfall = pd.read_csv("Dataset/rainfall/rainfall1.csv")
rainfall = rainfall[['STATES','YEAR','ANNUAL']].reset_index(drop=True)
rainfall_dict = {}
for i in range(rainfall.shape[0]):
	rainfall_dict[(rainfall.loc[i,'STATES'],rainfall.loc[i,'YEAR'])] = rainfall.loc[i,'ANNUAL']

temperature = pd.read_csv("Dataset/Mean_Temp.csv")
temperature = temperature[['YEAR','ANNUAL']].reset_index(drop=True)
print(temperature.columns)
temperature_dict = {}

for i in range(temperature.shape[0]):
	temperature_dict[temperature.loc[i,'YEAR']] = temperature.loc[i,'ANNUAL']


def editDistDP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j    
            elif j == 0:
                dp[i][j] = i    
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1+min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])    
 
    return dp[m][n]
 

crop_states = set(crop_prod['State_Name'].tolist())
rainfall_states = set(rainfall['STATES'].tolist())
# map_rainfall_crop = {}
map_crop_rainfall = {}
map_rainfall_crop = {}
for i in crop_states:
	mn = 3
	mn_state = ""
	for j in rainfall_states:
		if i.lower()==j.lower():
			map_crop_rainfall[i] = j
		else:
			x = editDistDP(i.lower(),j.lower(),len(i),len(j))
			if x<=mn:
				mn = x 
				map_crop_rainfall[i]=j





map_crop_rainfall['Madhya Pradesh'] = 'MADHYA PRADESH' 
map_crop_rainfall['Jammu and Kashmir '] = 'JAMMU & KASHMIR'
# map_rainfall_crop['LAKSHADWEEP'] = 
print(map_crop_rainfall)
print(len(list(map_crop_rainfall.keys())))
print(set(rainfall_states)-set(list(map_crop_rainfall.values())))
print(set(crop_states)-set(list(map_crop_rainfall.keys())))
print(crop_states)




merged_csv = crop_prod
merged_csv = merged_csv[(~merged_csv['State_Name'].isin(set(crop_states)-set(list(map_crop_rainfall.keys())))) & (merged_csv['Crop_Year']>=2000)].reset_index(drop=True)
# rainfall = rainfall[~rainfall['STATES'].isin(set(rainfall_states)-set(list(map_rainfall_crop.keys())))]
merged_csv['rainfall'] = merged_csv.apply(lambda row: rainfall_dict[(map_crop_rainfall[row['State_Name']],row['Crop_Year'])],axis=1)
merged_csv['temperature'] = merged_csv.apply(lambda row: temperature_dict[row['Crop_Year']],axis=1)

# merged_csv = pd.concat([crop_prod,rainfall.rename(columns={'STATES':'State_Name','ANNUAL':'rainfall'})],axis=1,join='')
# merged_csv = pd.concat([merged_csv,temperature.rename(columns = {'YEAR' : 'Crop_Year','ANNUAL':'temperature'})], axis=1, join='')
# merged_csv = merged_csv.loc[:, ~merged_csv.columns.duplicated()]
# print(merged_csv.shape)



merged_csv = merged_csv[~merged_csv['Production'].isna()].reset_index(drop=True)





# enc = OneHotEncoder(handle_unknown='ignore')
# enc_df = pd.DataFrame(enc.fit_transform(X[['Season']]).toarray())
# X = X.join(enc_df)


# # print(dum_df.shape)
# # y = y.fillna(0)
# # X = X.join(dum_df)
# # print(dum_df.shape[1])
# print(crop_prod.shape)
# print(merged_csv.shape)
# print(list(X_train.columns))
# # print(dum_df.shape)
# # print(y.head())
# print(merged_csv.columns)



# for i in range(len(y_pred)):
# 	print(y_pred[i],y_test[i])


# crop_prod = crop_prod[(crop_prod['Crop']=='Rice')].reset_index(drop=True)
# districts = set(crop_prod['District_Name'].tolist())
# for district in districts:
#     df = crop_prod[(crop_prod['District_Name']==district)]
#     if df.shape[0]!=0:
#         plt.plot(df['Crop_Year'],df['Production'],'ro')
#         plt.show()
merged_csv.shape

rainfall = pd.read_csv("Dataset/rainfall/district wise rainfall normal.csv")
rainfall_dict = {}
cols = list(rainfall.columns)
cols.remove('STATE_UT_NAME')
cols.remove('DISTRICT')
l= [cols.remove(x) for x in ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']]
print(cols)
for i in range(rainfall.shape[0]):
  st = rainfall.loc[i,"DISTRICT"]
  rainfall_dict[st] = rainfall.loc[i,"ANNUAL"]

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter=',')
print(rainfall[['DISTRICT','ANNUAL']].sort_values(by='ANNUAL'))
print(rainfall_dict)

district1 = set(crop_prod["District_Name"].tolist())
district2 = set(rainfall['DISTRICT'].tolist())
map_rainfall_crop = {}
map_district_rainfall = {}
for i in district1:
  mn = 3
  mn_state = ""
  for j in district2:
    if i.lower()==j.lower():
      map_district_rainfall[i] = j
      break
    else:
      x = editDistDP(i.lower(),j.lower(),len(i),len(j))
      if x<=mn:
        mn = x 
        map_district_rainfall[i]=j



d = {'BID':'BEED',
'GANGANAGAR':'SRI GANGANAGA',
'HAORA':'HOWRAH',
'HUGLI' : 'HOOGHLY',
'KHERI' : 'KHERI LAKHIMP',
 'TIRUCHCHIRAPPALLI':'TIRUCHIRAPPAL',
 'TIRUNELVELI-KATTABO':'TIRUNELVELI',
 'SINGHBHUM':'WEST SINGHBHUM',
 'KACHCHH':'KUTCH',
 'VADODARA':'BARODA',
 'THE-DANGS':'DANGS',
 'KENDUJHAR':'KEONDJHARGARH',
 'PHULABANI':'KANDHAMAL/PHU',
 'KOCH-BIHAR':'COOCH BEHAR',
 'BARDDHAMAN':'BURDWAN'}



for dist in d.keys():
  map_district_rainfall[dist]=d[dist]

for k,v in map_district_rainfall.items():
	map_rainfall_crop[v] = k

multi_d = {'CHAMPARAN':['EAST CHAMPARAN','WEST CHAMPARAN'],
'KANPUR' : ['KANPUR NAGAR','KANPUR DEHAT'],
'MEDINIPUR':['EAST MIDNAPOR','WEST MIDNAPOR']
,'BANGLORE' : ['BANGALORE RUR','BANGALORE URB']
,'MUMBAI' : ['MUMBAI CITY','MUMBAI SUB']
}
dels = set(['KASGANJ','NUAPADA','RAMBAN','AMETHI','SHAMLI','SAMBHAL','NAVSARI','DHEMAJI','BOUDH','DOHAD','KHOWAI','MUMBAI','PATAN','SUKMA','KADAPA','SONIPAT','KONDAGAON','ANJAW','GANJAM','GORAKHPUR','BALOD','CHIRANG','AMROHA','HAPUR','ANAND','SHRAVASTI','TAPI','GARIYABAND','SURAJPUR','ALIRAJPUR','TAWANG','KANKER','ANUPPUR','BALESHWAR','NARMADA','PALWAL','NAMSAI','BEMTARA','PALGHAR','HATHRAS','RAMANAGARA'])
for i in dels:
  if i in map_district_rainfall:
    del map_district_rainfall[i]

residuals = set(crop_prod['District_Name'].tolist())-set(map_district_rainfall.keys())
print(len(map_district_rainfall))
print(len(residuals))

print(map_district_rainfall)

# def months(season):
# 	switcher = {
# 		"Kharif" : ['JUL', 'AUG', 'SEP', 'OCT'],
# 		"Rabi" : ['NOV', 'DEC', 'JAN', 'FEB','MAR'],
# 		"Summer" : ['APR', 'MAY', 'JUN'],
#     "WholeYear" : ['JAN', 'FEB','MAR','APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT']
# 		}
# 	return switcher.get(season, "Invalid season")
# def get_rainfall(row):
#   season = ''.join(row['Season'].split())
#   # print(season)
#   mnths = months(season)
#   # print(mnths)
#   dist = row['District_Name']
#   rain = 0
#   for mnth in mnths:
#     # print(mnth)
#     rain+=rainfall_dict[(map_district_rainfall[dist],mnth)]
#   return rain
# print(set(df['Season'].tolist()))
cat_max=20


df = merged_csv

df = df[df['District_Name'].isin(map_district_rainfall.keys())].reset_index(drop=True)
df["rainfall"]=df.apply(lambda row : rainfall_dict[map_district_rainfall[row["District_Name"]]],axis=1)
# df.loc[df.Season == 'Winter     ', 'Season'] = 'Rabi       '
# df.loc[df.Season == 'Autumn     ','Season'] = 'Kharif     ' 
# df['rainfall1'] = df.apply(lambda row: get_rainfall(row),axis=1)
min_rain = df["rainfall"].min(axis=0)
max_rain =  df["rainfall"].max(axis=0)
diff = (max_rain-min_rain)/cat_max
cats = [(min_rain+diff*i,min_rain+diff*(i+1)) for i in range(cat_max)]
print(cats)
df['rain_cat'] = df.apply(lambda row:get_rain_category(row["rainfall"],min_rain,max_rain,cat_max),axis=1)
df['Production/Area'] = df.apply(lambda row:row['Production']/row['Area'],axis=1)
df = df.groupby(['rain_cat','Crop'])['Production/Area'].agg('mean').reset_index()
crops = set(df["Crop"].tolist())
optim_rain = {}
mx_prod = 0
opt_cat = ""
for crop in major_crops:
  df1 = df[df["Crop"]==crop].reset_index(drop=True)
  plt.plot(df1["rain_cat"],df1["Production/Area"])
  plt.xticks(range(1,cat_max+1))
  plt.title("Average yield vs rainfall range for "+str(crop))
  plt.ylabel("Average yield(Tonnes)")
  plt.xlabel("Rainfall category")
  plt.show()
  
  for i in range(df1.shape[0]):
    if df1.loc[i,"Production/Area"]>mx_prod:
      mx_prod=df1.loc[i,"Production/Area"]
      opt_cat = cats[df1.loc[i,"rain_cat"]-1]
  
  optim_rain[crop] = str(int(opt_cat[0]))+"-"+str(int(opt_cat[1]))

# df1 = merged_csv.groupby(['rain_cat','Crop'])['Production'].agg('median').reset_index()
# df1 = df1[df1["Crop"]=="Jute"]
# plt.plot(df["rain_cat"],df["Production"])
# plt.xticks(range(1,cat_max+1))
# plt.show()
# sns.boxplot(x="rain_cat", y="Production", data=df1,showfliers=False)
# merged_csv
print(optim_rain)
# print(crops)
l = [[a,b] for (a,b) in list(optim_rain.items())]
df2 = pd.DataFrame(l,columns=["Crop","Annual Rainfall Range"])
# df2.to_csv("Dataset/rainfall_range.csv")

income = pd.read_csv('Dataset/income.csv',delimiter=',')
income.loc[income["Crop"]=="Moong(Green Gram)","Crop"] = "Moong"
income.loc[income["Crop"]=="Rapeseed &Mustard","Crop"] = "Rapeseed"
# revenue['Revenue per hectare'] = revenue.apply(lambda row: row['Revenue per hectare']*10, axis=1)
plts = []
leg = []
price_slope = {}
crops = set(income['Crop'].tolist())

print(sorted(list(crops)))
plt.figure(figsize=(15,10))
for crop in crops:
  df = income[income['Crop']==crop].reset_index(drop=True)
  # print(df.loc[df['Crop_Year']==2002,'Price per quintal'])
#   st = df.loc[df['Year']==2008,:].reset_index().loc[0,'Income per quintal']
#   fin = df.loc[df['Year']==2013,:].reset_index().loc[0,'Income per quintal']
#   # print(st)
#   price_slope[crop] =  (fin - st)/10
  x, = plt.plot(df['Year'],df['New Income/Hect'])
  plts.append(x)
  leg.append(crop)
# plt
plt.xticks(range(2008, 2013))
plt.xlabel('Year')
plt.ylabel('Income/hectare(Rs)')
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor=(1.2, 0))
plt.savefig('crop_income.png',bbox_inches='tight')
plt.show()
plt.clf()
plts = []
leg = []
plt.figure(figsize=(15,10))
for crop in crops:
  df = income[income['Crop']==crop].reset_index(drop=True)
  # print(df.loc[df['Crop_Year']==2002,'Price per quintal'])
#   st = df.loc[df['Year']==2008,:].reset_index().loc[0,'Income per quintal']
#   fin = df.loc[df['Year']==2013,:].reset_index().loc[0,'Income per quintal']
#   # print(st)
#   price_slope[crop] =  (fin - st)/10
  x, = plt.plot(df['Year'],df['New Cost PerHectare'])
  plts.append(x)
  leg.append(crop)
# plt
plt.xticks(range(2008, 2013))
plt.xlabel('Year')
plt.ylabel('Cost/hectare(Rs)')
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor=(1.2, 0))
plt.savefig('crop_cost.png',bbox_inches='tight')

# plt.clf()
# revenue['yield'] = revenue.apply(lambda row: row['Production in quintals']*10/row['Area in Hectares'] if row['Crop']!='Sugarcane' else row['Production in quintals']*10/(10*row['Area in Hectares']),axis=1)
df = income.groupby(['Crop'])['New Income/Hect'].agg('mean').reset_index()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
crop = df['Crop'].tolist()
yld = df['New Income/Hect'].tolist()
bars =ax.bar(crop,yld)
# bars[-2].set_color('bl')
plt.xticks(rotation='vertical')
plt.ylabel('Income/Hectare of Crops')
plt.savefig('crop_income_bar.png')
plt.show()

df = income.groupby(['Crop'])['New Cost PerHectare'].agg('mean').reset_index()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
crop = df['Crop'].tolist()
yld = df['New Cost PerHectare'].tolist()
bars =ax.bar(crop,yld)
# bars[-2].set_color('bl')
plt.xticks(rotation='vertical')
plt.ylabel('Cost/Hectare of Crops')
# plt.savefig('crop_income_bar.png')
plt.savefig('crop_income_bar.png', bbox_inches='tight')

plt.show()

df = merged_csv.fillna(0)
df = df[['Area','rainfall','State_Name','Crop_Year']]
df1 = df.groupby(['State_Name','Crop_Year','rainfall'])['Area'].agg('sum').reset_index()
# df1.reset_index(level=1, inplace=True)
# df1 = df1.groupby(['State_Name'])['Area','rainfall'].agg({'Area':'mean','rainfall':'mean'})
# df1.reset_index(level=0,inplace=True)
states = set(df1['State_Name'].tolist())
for state in states:
  small_df = df1[df1['State_Name']==state].reset_index(drop=True)
  
  corr = small_df['Area'].corr(small_df['rainfall'])
  if corr>=0.65: #or corr<=-0.2:
    plt.plot(small_df['rainfall'],small_df['Area'],'ro')
    plt.title("State : "+state+"\ncorr : "+str(corr))
    plt.xlabel("Rainfall(cm)")
    plt.ylabel("Cultivation Area(Hectare)")
    plt.rcParams.update({'font.size': 15})
    plt.show()
  if state=="West Bengal":
    plt.plot(small_df['rainfall'],small_df['Area'],'ro')
    plt.title("State : "+state+"\ncorr : "+str(corr))
    plt.xlabel("Rainfall(cm)")
    plt.ylabel("Cultivation Area(Hectare)")
    plt.rcParams.update({'font.size': 15})
    plt.show()


# crop_prod_sum.columns = crop_prod_sum.columns.droplevel(0)

crop_prod = pd.read_csv('Dataset/apy.csv',delimiter=',')
# crop_prod = crop_prod.fillna(0)
crop_prod = crop_prod.groupby(['Crop_Year','Crop'])['Production'].agg('sum').reset_index()
crop_prod = crop_prod[crop_prod['Crop_Year'].isin(range(2002,2013))].reset_index(drop=True)
plts = []
leg = []
plt.figure(figsize=(15,10))
for crop in major_crops:
  
  df = crop_prod[crop_prod['Crop']==crop].reset_index(drop=True)
  if crop=="Bajra":
    print(df)
  if crop in ['Sugarcane','Wheat','Rice']:
    x, =plt.plot(df['Crop_Year'],df['Production']/10,'--')
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['Production'])
    plts.append(x)
    leg.append(crop)

 

plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.12,0))

plt.show()

crops = set(crop_prod['Crop'].tolist()) - set(['Coconut '])
# crops = {'Gram', 'Ragi', 'Groundnut', 'Rice', 'Sunflower', 'Jowar', 'Safflower', 'Rapeseed &Mustard', 'Bajra', 'Niger seed', 'Sugarcane', 'Arhar/Tur', 'Maize', 'Moong(Green Gram)', 'Urad', 'Sesamum', 'Soyabean', 'Wheat', 'Cotton(lint)', 'Jute', 'Barley'}
slope_d = {}
W_slope = {}
crops_with_less_yearwise_data = []
for crop in crops:
  df3 = crop_prod[crop_prod['Crop']==crop]
  if df3.shape[0]<8:
    crops_with_less_yearwise_data.append(crop)
  df1 = crop_prod[(crop_prod['Crop_Year']<=2004) & (crop_prod['Crop']==crop)].reset_index(drop=True)
  df1 = df1[~df1.isna()]
  start = df1['Production'].mean(axis=0)
  df2 = crop_prod[(crop_prod['Crop_Year']>=2010) & (crop_prod['Crop']==crop)].reset_index(drop=True)
  df2 = df2[~df2.isna()]
  end = df2['Production'].mean(axis=0)
  if start!=0:
    if start==start and end==end:
      slope_d[crop] = (math.pow(end/start,1/9)-1)*100#(end-start)/start
  if start==start and end==end:
    W_slope[crop] = (end-start)/9

prod_slope = list(slope_d.items())
prod_slope = sorted(prod_slope,key = lambda x : x[1])
W_slope_l = list(W_slope.items())
W_slope_l = sorted(W_slope_l,key = lambda x : x[1])
W_slope

fig = plt.figure(figsize=(15, 6))
ax = fig.add_axes([0,0,1,1])
crops_l = list(slope_d.keys())

slopes = [slope_d[crop] for crop in crops_l]

bars =ax.bar(crops_l,slopes)
plt.xticks(rotation='vertical')
plt.ylabel('Production growth(in %)')
plt.show()


print(crops_with_less_yearwise_data)

"""**Train Test Split**"""

## train and test production only on major crops
merged_csv_ = merged_csv[merged_csv['Crop'].isin(major_crops)].reset_index(drop=True)
## -----------------

## train and test production on non major crops
# merged_csv_ = merged_csv[~merged_csv['Crop'].isin(major_crops)].reset_index(drop=True)
## ----------------

## Crops with less yearwise production data are excluded

crops_with_less_yearwise_data = ['Bitter Gourd', 'Pome Fruit', 'Other Fresh Fruits', 'Jute & mesta', 'Peas  (vegetable)', 'Ber', 'Other Vegetables', 'Pineapple', 'Rubber', 'Cashewnut Processed', 'Plums', 'Cowpea(Lobia)', 'Tea', 'Drum Stick', 'Pome Granet', 'Water Melon', 'Lemon', 'Brinjal', 'Tomato', 'Blackgram', 'Yam', 'Ash Gourd', 'Bhindi', 'Korra', 'Turnip', 'Other Citrus Fruit', 'Cabbage', 'Citrus Fruit', 'Apple', 'Paddy', 'Litchi', 'Lab-Lab', 'Redish', 'Grapes', 'Varagu', 'Carrot', 'Ginger', 'Samai', 'Pulses total', 'Cucumber', 'Coffee', 'Beet Root', 'Jack Fruit', 'Pump Kin', 'Bottle Gourd', 'Cauliflower', 'Orange', 'Ribed Guard', 'other fibres', 'Pear', 'Mango', 'other misc. pulses', 'Other Dry Fruit', 'Papaya', 'Peach', 'Sapota', 'Cashewnut Raw', 'Snak Guard', 'Arcanut (Processed)', 'Atcanut (Raw)', 'Beans & Mutter(Vegetable)']
# merged_csv_ = merged_csv[~merged_csv['Crop'].isin(crops_with_less_yearwise_data)].reset_index(drop=True)
## -----------

merged_csv_ = pd.get_dummies(merged_csv_,columns = ["Season","Crop"],prefix = ['Season_is','Crop_is'])
merged_csv_ = merged_csv_.drop(columns = ['State_Name','District_Name'],axis=0)
merged_csv_ = merged_csv_.fillna(0).reset_index(drop=True)
# merged_csv_X = merged_csv_.drop(["Production","Crop_Year"],axis=1)
# merged_csv_Y = merged_csv_[["Production","Crop_Year"]]
# mm_scaler = preprocessing.MinMaxScaler()
# l = mm_scaler.fit_transform(merged_csv_X)
# merged_csv_ = pd.DataFrame(l, index=merged_csv_X.index, columns=merged_csv_X.columns)
# merged_csv_["Production"] = merged_csv_Y["Production"]
# merged_csv_["Crop_Year"] = merged_csv_Y["Crop_Year"]

merged_csv_train = merged_csv_[merged_csv_['Crop_Year']<=2008].reset_index(drop=True)
merged_csv_test = merged_csv_[merged_csv_['Crop_Year']>2008].reset_index(drop=True)
X_train = merged_csv_train.drop(columns=['Production'],axis=1)
y_train = merged_csv_train['Production']
X_test = merged_csv_test.drop(columns=['Production'],axis=1)
y_test = merged_csv_test['Production']
# y = merged_csv["production/Area"]
# X = merged_csv.drop(columns = ["production/Area"],axis=1)
# X["Season"] =X["Season"].astype('category')
# X["Season_Cat"] = X["Season"].cat.codes

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
print(X_train.shape,X_test.shape)
print(X_train.columns)

"""**Linear Regression**"""

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test , y_pred)
print('Random forest validation MAE = ', MAE)
print(r2_score(y_test,y_pred))
plt.plot(y_test,y_pred,'ro')
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.show()

"""**Decision Tree**"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
MAE = mean_absolute_error(y_test , y_pred)
print('Random forest validation MAE = ', MAE)
print(r2_score(y_test,y_pred))
plt.plot(y_test,y_pred,'ro')
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.show()
print(regressor.get_params)

"""**Random Forest**"""

# random forest 
model = RandomForestRegressor()
model.fit(X_train,y_train)

# Get the mean absolute error on the validation data
y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test , y_pred)
print('Random forest validation MAE = ', MAE)
print(r2_score(y_test,y_pred))
plt.plot(y_test,y_pred,'ro')
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.show()
print(model.get_params)

"""**XGBoost**"""

## XGBOOST
XGBModel = XGBRegressor()
XGBModel.fit(X_train,y_train , verbose=False)

# Get the mean absolute error on the validation data :
y_pred = XGBModel.predict(X_test)
MAE = mean_absolute_error(y_test , y_pred)
print('XGBoost validation MAE = ',MAE)

print(r2_score(y_test,y_pred))
plt.plot(y_test,y_pred,'ro')
plt.show()
print(XGBModel.get_params)

"""**ANN**"""

# # ANN
# NN_model = Sequential()

# # The Input Layer :
# NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# # The Hidden Layers :
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# # Compile the network :
# NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# NN_model.summary()

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]

# NN_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split = 0.2, callbacks=callbacks_list)
# # wights_file = 'Weights-081--85248.93294.hdf5' # choose the best checkpoint 
# # NN_model.load_weights(wights_file) # load it

# y_pred = NN_model.predict(X_test)
# # print(y_pred)
# print(r2_score(y_test,y_pred))

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter = ',')
crop_prod["production/Area"] = crop_prod.apply(lambda row : row["Production"]/row["Area"]*100 if row["Area"]!=0 else np.NaN,axis=1)

crop_prod=crop_prod.fillna(0)

# print(crop_prod.columns)


crops = set([])
districts = set([])
for i in range(crop_prod.shape[0]):
	crops.add(crop_prod.loc[i,"Crop"])
	districts.add(crop_prod.loc[i,"District_Name"])
# crop_prod =  crop_prod[crop_prod["District_Name"]=="UNA"]
print(districts)
print(crops)
# # crop_prod = crop_prod[crop_prod["Crop"]=="Jute"]
# # crop_prod = crop_prod.reset_index(drop=True)
# # print(crop_prod.head(10))
# crop_prod_sum = crop_prod.groupby(['Crop','Crop_Year'])['Production'].agg('sum').to_frame('total').reset_index()
# # crop_prod_sum.reset_index(level=0, inplace=True)
# # crop_prod_sum.columns = crop_prod_sum.columns.droplevel(0)

# crop_prod_sum["rate"] = crop_prod_sum.total.pct_change().mul(100).round(2)
# crop_prod_stats = crop_prod_sum.groupby(['Crop'])['total'].agg({'mean':'mean','std':'std','first':'first','last':'last'})
# # crop_prod_sum = crop_prod_sum[(crop_prod_sum.total!=0)]
# crop_prod_stats['overall_rate'] = crop_prod_stats.apply(lambda row : (row['last']-row['first'])/row['first']*100 if row['first']!=0 else np.NaN, axis = 1) 
# crop_prod_stats = crop_prod_stats.sort_values(by=['overall_rate'])
# print(crop_prod_stats)
# # print(crop_prod_sum["Crop_Year"])
# # plt.plot(crop_prod_sum["Crop_Year"],crop_prod_sum["rate"])
# # plt.show(1)


rainfall = pd.read_csv("Dataset/rainfall/rainfall1.csv")
# print(rainfall.head())
# print(rainfall.columns)
# print(result.head(10))
# print(result.columns)
crop_prod = crop_prod.groupby(['State_Name','Crop_Year','Crop'])['production/Area'].agg('sum').to_frame('total').reset_index()
result=pd.concat([crop_prod, rainfall.rename(columns={'STATES':'State_Name'})], axis=1, join='inner')
result = result[result["Crop"]=="Rice"]
print(result)
print(result.columns)
print(result["total"].corr(result["ANNUAL"]))
# plt.plot(result['ANNUAL'],result['total'],'ro')
# plt.show(1)


model = LinearRegression()
model.fit(pd.DataFrame(result["ANNUAL"]),pd.DataFrame(result["total"]))
print(model.coef_)

y_actual = pd.DataFrame(result["total"])
# print(y_actual)
y_pred = model.predict(pd.DataFrame(result["ANNUAL"]))
print(y_pred,y_actual)
ssr = np.sum((y_pred - y_actual)**2)
print(math.sqrt(ssr))


plt.plot(result["ANNUAL"],result["total"],'ro')
plt.plot(result["ANNUAL"],model.predict(pd.DataFrame(result["ANNUAL"])))

plt.legend(loc='best')
plt.show()

# find the optimal rainfall range, temperature range

crop_prod = pd.read_csv('Dataset/apy.csv',delimiter = ',')
start_overall = 75.26 
end_overall = 82.14
start_men = 53.67
end_men = 65.46
start_women = 64.83
end_women = 74.04
diff_overall = (end_overall-start_overall)/10
diff_men = (end_men-start_men)/10
diff_women = (end_women-start_women)/10
pop_2001 = 1028737436
pop_2011 = 1210193422
rate = math.pow(pop_2011/pop_2001,0.1)-1
print(rate)
crop_prod['Literacy_overall'] = crop_prod.apply(lambda row : start_overall+diff_men*(row['Crop_Year']-2001),axis=1) 
crop_prod['Literacy_men'] = crop_prod.apply(lambda row : start_men+diff_men*(row['Crop_Year']-2001),axis=1)
crop_prod['Literacy_women'] = crop_prod.apply(lambda row : start_women+diff_women*(row['Crop_Year']-2001),axis=1)   
crop_prod['population'] = crop_prod.apply(lambda row :pop_2001*math.pow((1+rate),(row['Crop_Year']-2001)),axis=1)

"""**Top5 crops whose production is most correlated to literacy rate**"""

## Realtionship between production and literacy rate
# for crop in major_crops:
df = crop_prod.groupby(['Crop_Year','Crop'])['Production','Literacy_overall','Literacy_men','Literacy_women'].agg({'Production':'sum','Literacy_overall':'min','Literacy_men':'min','Literacy_women':'min'}).reset_index()
print(crop_prod)
lito_prod_corr = {}
litm_prod_corr = {}
litw_prod_corr = {}
for crop in major_crops:
  df1 = df[df['Crop']==crop].reset_index(drop=True)
  lito_prod_corr[crop] = df1['Production'].corr(df1['Literacy_overall'])
  litm_prod_corr[crop] = df1['Production'].corr(df1['Literacy_men'])
  litw_prod_corr[crop] = df1['Production'].corr(df1['Literacy_women'])
  if litw_prod_corr[crop]>0.1 or litm_prod_corr[crop] >0.1 or lito_prod_corr[crop]>0.1:
    plt.plot(df1['Literacy_overall'],df1['Production'],'bo')
    plt.title("Production of "+crop+" vs Literacy rate\n overall corr : "+str(lito_prod_corr[crop]))
    plt.show()  

l = list(lito_prod_corr.items())
l = sorted(l,key=lambda x:x[1],reverse=True)
top5_correlated_crops = [x[0] for x in l[:5]]
top5_correlated_crops

"""**Top 5 crops whose production is most correlated with population**"""

df = crop_prod.groupby(['Crop_Year','Crop','population'])['Production'].agg('sum').reset_index()
print(crop_prod)
pop_prod_corr = {}
for crop in major_crops:
  df1 = df[df['Crop']==crop].reset_index(drop=True)
  pop_prod_corr[crop] = df1['Production'].corr(df1['population'])
  if pop_prod_corr[crop]>0.1:
    plt.plot(df1['population'],df1['Production'],'bo')
    plt.title("Production of "+crop+" vs Population\n overall corr : "+str(pop_prod_corr[crop]))
    plt.show()  

l = list(pop_prod_corr.items())
l = sorted(l,key=lambda x:x[1],reverse=True)
top5_correlated_crops = [x[0] for x in l[:5]]
top5_correlated_crops

"""**Top 5 major crops whose production is correlated with rainfall**"""

df = crop_prod.groupby(['Crop_Year','Crop'])['Production'].agg('sum').reset_index()
df = df[df['Crop_Year'].isin(range(2002,2013))].reset_index(drop=True)
rainfall = pd.read_csv('Dataset/rainfall/rainfall1.csv',delimiter=',')
rainfall = rainfall.groupby(['YEAR'])['ANNUAL'].agg('sum').reset_index()
df['rainfall'] = df.apply(lambda row : rainfall[rainfall['YEAR']==row['Crop_Year']].reset_index(drop=True).iloc[0,1],axis=1)
# print(crop_prod)
print(df)
rainfall_prod_corr = {}
for crop in major_crops:
  df1 = df[df['Crop']==crop].reset_index(drop=True)
  rainfall_prod_corr[crop] = df1['Production'].corr(df1['rainfall'])
  if rainfall_prod_corr[crop]>0.1:
    plt.plot(df1['rainfall'],df1['Production'],'bo')
    plt.title("Production of "+crop+" vs Rainfall\n overall corr : "+str(rainfall_prod_corr[crop]))
    plt.show()  

l = list(rainfall_prod_corr.items())
l = sorted(l,key=lambda x:x[1],reverse=True)
top5_correlated_crops = [x[0] for x in l[:5]]
top5_correlated_crops

"""**Top 5 crops with highest average cultivation area and production over years**"""

df = crop_prod.groupby(['Crop_Year','Crop'])['Production','Area'].agg({'Production':'sum','Area':'sum'}).reset_index()
df = crop_prod.groupby(['Crop'])['Production','Area'].agg({'Production':'mean','Area':'mean'}).reset_index()

## Top 5 crops with highest average production over 15 years
df = df.sort_values(by=['Production'],ascending = False)
print(df.head(5))
##
df = df.sort_values(by=['Area'],ascending = False)
print(df.head(5))

"""**Top 5 crops with highest increase in production over years**"""

df = crop_prod.groupby(['Crop_Year','Crop'])['Production'].agg('sum').reset_index()
df= df[df['Crop_Year'].isin(range(2002,2013))].reset_index(drop=True)
crop_prod_growth = {}
for crop in major_crops:
  df1 = df[df['Crop']==crop].reset_index(drop=True)
  df2 = df1[(df1['Crop_Year']<=2004) & (df1['Crop']==crop)].reset_index(drop=True)
  df2 = df2[~df2.isna()]
  start = df2['Production'].mean(axis=0)
  df3 = df1[(df1['Crop_Year']>=2010) & (df1['Crop']==crop)].reset_index(drop=True)
  df3 = df3[~df3.isna()]
  end = df3['Production'].mean(axis=0)
  a = (end-start)/start
  b = (end-start)/8
  crop_prod_growth[crop] = (a,b)

l = list(crop_prod_growth.items())
l = sorted(l,key=lambda x:x[1][0],reverse=True)
l = [x[0] for x in l[:5]]
print("Top 5 crops with highest increase in production relative to initial production b/w 2002-2004(percent-wise)",l)

l = list(crop_prod_growth.items())
l = sorted(l,key=lambda x:x[1][1],reverse=True)
l = [x[0] for x in l[:5]]
print("Top 5 crops with highest slope of production b/w 2002-2012",l)

"""**Top 5 crops with highest increase in cultivation area over years**"""

df = crop_prod.groupby(['Crop_Year','Crop'])['Area'].agg('sum').reset_index()
df= df[df['Crop_Year'].isin(range(2002,2013))].reset_index(drop=True)
crop_area_growth = {}
for crop in major_crops:
  df1 = df[df['Crop']==crop].reset_index(drop=True)
  df2 = df1[(df1['Crop_Year']<=2004) & (df1['Crop']==crop)].reset_index(drop=True)
  df2 = df2[~df2.isna()]
  start = df2['Area'].mean(axis=0)
  df3 = df1[(df1['Crop_Year']>=2010) & (df1['Crop']==crop)].reset_index(drop=True)
  df3 = df3[~df3.isna()]
  end = df3['Area'].mean(axis=0)
  a = (end-start)/start
  b = (end-start)/8
  crop_area_growth[crop] = (a,b)

l = list(crop_area_growth.items())
l = sorted(l,key=lambda x:x[1][0],reverse=True)
l = [x[0] for x in l[:5]]
print("Top 5 crops with highest increase in cultivation area relative to initial Area b/w 2002-2004(percent-wise)",l)

l = list(crop_prod_growth.items())
l = sorted(l,key=lambda x:x[1][1],reverse=True)
l = [x[0] for x in l[:5]]
print("Top 5 crops with highest slope of cultivation area b/w 2002-2012",l)

"""**Top 5 district and state for each crop where production/Area is highest**"""

crop_prod = pd.read_csv('Dataset/apy.csv',delimiter=',')
crop_prod = crop_prod[~crop_prod['Production'].isna()].reset_index(drop=True)
crop_prod['Production/Area'] = crop_prod.apply(lambda row: row['Production']/row['Area'] if row['Area']!=0 else np.nan, axis=1)
crop_prod = crop_prod.groupby(['Crop','District_Name','State_Name'])['Production/Area'].agg('mean').reset_index()
crop_state = {}
crop_district = {}
for crop in crops:
  df = crop_prod[crop_prod['Crop']==crop].reset_index(drop=True)
  df = df.sort_values(by=['Production/Area'],ascending=False)
  # print(df)
  for i in range(min(df.shape[0],20)):
    if i==0:
      crop_district[crop]=[df.loc[i,'District_Name']]
    else:
      crop_district[crop].append(df.loc[i,'District_Name'])


for crop in crops:
  df = crop_prod[crop_prod['Crop']==crop].reset_index(drop=True)
  df = df.groupby(['Crop','State_Name'])['Production/Area'].agg('mean').reset_index()
  df = df.sort_values(by=['Production/Area'],ascending=False)
  for i in range(min(df.shape[0],5)):
    if i==0:
      crop_state[crop]=[df.loc[i,'State_Name']]
    else:
      crop_state[crop].append(df.loc[i,'State_Name'])

print(crop_district)
print(crop_state)

"""**Crops with similarity in condition at which they are grown clustered together**"""



crop_prod = pd.read_csv('Dataset/apy.csv',delimiter=',')
crop_prod = crop_prod[~crop_prod['Production'].isna()].reset_index(drop=True)
crop_prod['Production/Area'] = crop_prod.apply(lambda row: row['Production']/row['Area'] if row['Area']!=0 else np.nan, axis=1)
crop_prod = crop_prod[['Crop','Production/Area','Season','State_Name']]
states = set(crop_prod['State_Name'].tolist())
crop_prod = pd.get_dummies(crop_prod,columns = ["Season","State_Name"],prefix = ['Season_is','State_is'])
# crop_prod = crop_prod[~(crop_prod['Season_is_Whole Year ']==1)].reset_index(drop=True)
# print(crop_prod.columns)
crop_prod = crop_prod.groupby(['Crop'])[ 'Season_is_Autumn     ','Season_is_Kharif     ', 'Season_is_Rabi       ','Season_is_Summer     ','Season_is_Winter     ','Season_is_Whole Year ','State_is_Andaman and Nicobar Islands',
       'State_is_Andhra Pradesh', 'State_is_Arunachal Pradesh',
       'State_is_Assam', 'State_is_Bihar', 'State_is_Chandigarh',
       'State_is_Chhattisgarh', 'State_is_Dadra and Nagar Haveli',
       'State_is_Goa', 'State_is_Gujarat', 'State_is_Haryana',
       'State_is_Himachal Pradesh', 'State_is_Jammu and Kashmir ',
       'State_is_Jharkhand', 'State_is_Karnataka', 'State_is_Kerala',
       'State_is_Madhya Pradesh', 'State_is_Maharashtra', 'State_is_Manipur',
       'State_is_Meghalaya', 'State_is_Mizoram', 'State_is_Nagaland',
       'State_is_Odisha', 'State_is_Puducherry', 'State_is_Punjab',
       'State_is_Rajasthan', 'State_is_Sikkim', 'State_is_Tamil Nadu',
       'State_is_Telangana ', 'State_is_Tripura', 'State_is_Uttar Pradesh',
       'State_is_Uttarakhand', 'State_is_West Bengal'].agg('mean').reset_index()

# crops = crop_prod['Crop'].tolist()
# for crop in crops:
#   for i in states:
#     s= 'State_is_'+i
#     # print(crop_prod[crop_prod['Crop']==crop][s])
#     crop_prod.loc[crop_prod['Crop']==crop,s] = 0 
#     # print(crop_prod.loc[crop_prod['Crop']==crop,s])
#   for i in crop_state[crop]:
#     s = 'State_is_'+i
#     crop_prod.loc[crop_prod['Crop']==crop,s] = 1

df = crop_prod.drop(columns = ['Crop'],axis=0)
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=10000)
tsne_results = tsne.fit_transform(df)
tsne_results = pd.DataFrame(tsne_results,columns= ['x','y'])
model = KMeans(n_clusters=6).fit(df)
y_pred = model.labels_
cluster_crops = {}
crops = crop_prod['Crop'].tolist()
for i in range(len(y_pred)):
  if y_pred[i] in cluster_crops:
    cluster_crops[y_pred[i]].append(crops[i])
  else:
    cluster_crops[y_pred[i]] = [crops[i]]
# for i in range():
#   print(cluster_crops[i])
print(cluster_crops)
centroids  = model.cluster_centers_ 
print(centroids)

plt.scatter(tsne_results['x'],tsne_results['y'],c=y_pred.astype('float'))
plt.show()

for i in range(4):
  print(len(cluster_crops[i]))
  print(cluster_crops[i])

"""**Cluster similar district based on ph,soil type and rainfall and crops based on soil type and season in which it is grown**"""

rainfall = pd.read_csv("Dataset/rainfall/district wise rainfall normal.csv")
rainfall_dict = {}
cols = list(rainfall.columns)
cols.remove('STATE_UT_NAME')
cols.remove('DISTRICT')
l= [cols.remove(x) for x in ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']]
print(cols)
for i in range(rainfall.shape[0]):
  st = rainfall.loc[i,"DISTRICT"]
  rainfall_dict[st] = rainfall.loc[i,"ANNUAL"]

crop_prod = pd.read_csv("Dataset/apy.csv",delimiter=',')
district1 = set(crop_prod["District_Name"].tolist())
district2 = set(rainfall['DISTRICT'].tolist())
map_rainfall_crop = {}
map_district_rainfall = {}
for i in district1:
  mn = 3
  mn_state = ""
  for j in district2:
    if i.lower()==j.lower():
      map_district_rainfall[i] = j
      break
    else:
      x = editDistDP(i.lower(),j.lower(),len(i),len(j))
      if x<=mn:
        mn = x 
        map_district_rainfall[i]=j



d = {'BID':'BEED',
'GANGANAGAR':'SRI GANGANAGA',
'HAORA':'HOWRAH',
'HUGLI' : 'HOOGHLY',
'KHERI' : 'KHERI LAKHIMP',
 'TIRUCHCHIRAPPALLI':'TIRUCHIRAPPAL',
 'TIRUNELVELI-KATTABO':'TIRUNELVELI',
 'SINGHBHUM':'WEST SINGHBHUM',
 'KACHCHH':'KUTCH',
 'VADODARA':'BARODA',
 'THE-DANGS':'DANGS',
 'KENDUJHAR':'KEONDJHARGARH',
 'PHULABANI':'KANDHAMAL/PHU',
 'KOCH-BIHAR':'COOCH BEHAR',
 'BARDDHAMAN':'BURDWAN'}



for dist in d.keys():
  map_district_rainfall[dist]=d[dist]

for k,v in map_district_rainfall.items():
	map_rainfall_crop[v] = k

multi_d = {'CHAMPARAN':['EAST CHAMPARAN','WEST CHAMPARAN'],
'KANPUR' : ['KANPUR NAGAR','KANPUR DEHAT'],
'MEDINIPUR':['EAST MIDNAPOR','WEST MIDNAPOR']
,'BANGLORE' : ['BANGALORE RUR','BANGALORE URB']
,'MUMBAI' : ['MUMBAI CITY','MUMBAI SUB']
}
dels = set(['KASGANJ','NUAPADA','RAMBAN','AMETHI','SHAMLI','SAMBHAL','NAVSARI','DHEMAJI','BOUDH','DOHAD','KHOWAI','MUMBAI','PATAN','SUKMA','KADAPA','SONIPAT','KONDAGAON','ANJAW','GANJAM','GORAKHPUR','BALOD','CHIRANG','AMROHA','HAPUR','ANAND','SHRAVASTI','TAPI','GARIYABAND','SURAJPUR','ALIRAJPUR','TAWANG','KANKER','ANUPPUR','BALESHWAR','NARMADA','PALWAL','NAMSAI','BEMTARA','PALGHAR','HATHRAS','RAMANAGARA'])
for i in dels:
  if i in map_district_rainfall:
    del map_district_rainfall[i]

residuals = set(crop_prod['District_Name'].tolist())-set(map_district_rainfall.keys())

district_df = pd.read_csv("Dataset/district_soildata.csv",delimiter=',')
district_ph ={}
district_soil = {}
for i in range(district_df.shape[0]):
  district_ph[district_df.loc[i,"DISTNAME"]] = district_df.loc[i,"SoilPh"]
  district_soil[district_df.loc[i,"DISTNAME"]] = district_df.loc[i,"SoilType"]
district_df = pd.get_dummies(district_df,columns = ["SoilType"],prefix = ['SoilType_is'])
print(district_df.shape)
rainfall = pd.read_csv("Dataset/rainfall/district wise rainfall normal.csv",delimiter=',')

rainfall = rainfall[['DISTRICT','ANNUAL']]
# print(rainfall.shape)
dist_rainfall = {}
for i in range(rainfall.shape[0]):
  dist_rainfall[rainfall.loc[i,"DISTRICT"]] = rainfall.loc[i,"ANNUAL"]

# district_df['rainfall'] = district_df.apply(lambda row:dist_rainfall[row['DISTNAME']],axis=1)
# district_df

district1 = district_df['DISTNAME'].tolist()
district2 = rainfall['DISTRICT'].tolist()

map_district_rainfall = {}
for i in district1:
  mn = 3
  mn_state = ""
  for j in district2:
    if i.lower()==j.lower():
      map_district_rainfall[i] = j
      break
    else:
      x = editDistDP(i.lower(),j.lower(),len(i),len(j))
      if x<=mn:
        mn = x 
        map_district_rainfall[i]=j


d = {'BID':'BEED',
'GANGANAGAR':'SRI GANGANAGA',
'HAORA':'HOWRAH',
'HUGLI' : 'HOOGHLY',
'KHERI' : 'KHERI LAKHIMP',
 'TIRUCHCHIRAPPALLI':'TIRUCHIRAPPAL',
 'TIRUNELVELI-KATTABO':'TIRUNELVELI',
 'SINGHBHUM':'WEST SINGHBHUM',
 'KACHCHH':'KUTCH',
 'VADODARA':'BARODA',
 'THE-DANGS':'DANGS',
 'KENDUJHAR':'KEONDJHARGARH',
 'PHULABANI':'KANDHAMAL/PHU',
 'KOCH-BIHAR':'COOCH BEHAR',
 'BARDDHAMAN':'BURDWAN'}

for dist in d.keys():
  map_district_rainfall[dist]=d[dist]

multi_d = {'CHAMPARAN':['EAST CHAMPARAN','WEST CHAMPARAN'],
'KANPUR' : ['KANPUR NAGAR','KANPUR DEHAT'],
'MEDINIPUR':['EAST MIDNAPOR','WEST MIDNAPOR']
,'BANGLORE' : ['BANGALORE RUR','BANGALORE URB']
,'MUMBAI' : ['MUMBAI CITY','MUMBAI SUB']
}

print(len(map_district_rainfall))

# map_district_rainfall['BANGLORE'] = 
rem = [dist for dist in district2 if dist not in map_district_rainfall.values()]
# rem

#  'WEST_DINAJPUR',
#  '24_PARGANAS',
# 'WEST_NIMAR',
#  'EAST_NIMAR',
#  'CHENGALPATTU',
#  'NORTH_ARCOT_AMBEDKAR',
#  'SOUTH_ARCOT',

# for k,v in district_ph.items():
#   if v==0:
#     print(k)
# print(district_ph)

district_df = district_df[(district_df['DISTNAME'].isin(map_district_rainfall.keys()))].reset_index(drop=True)

df = merged_csv
map_district_crop = {}
map_crop_district = {}

for k,v in map_district_rainfall.items():
  if v in map_rainfall_crop:
    map_district_crop[k] = map_rainfall_crop[v]
    map_crop_district[map_rainfall_crop[v]] = k
l = district_df["DISTNAME"].tolist()
l1 = list(map_crop_district.keys())
for crop in l1:
  if map_crop_district[crop] not in l:
    del map_crop_district[crop] 

del map_crop_district['ETAH']
df = df[(df['District_Name'].isin(map_crop_district.keys()))].reset_index(drop=True)
df['Ph'] = df.apply(lambda row: district_ph[map_crop_district[row['District_Name']]] ,axis = 1)
df['SoilType'] = df.apply(lambda row: district_soil[map_crop_district[row['District_Name']]],axis = 1)
# map_crop_district["VALSAD"]

df['Production/Area'] = df.apply(lambda row:row['Production']/row['Area'],axis=1)
df_ = df.groupby(['SoilType','Crop'])['Production/Area'].agg('mean').reset_index()
crops = set(df_["Crop"].tolist())
optim_soil = {}

opt_cat = ""
labels = ["","","Laterite","Red and Yellow","Shallow Black","Medium Black","Deep Black","Mixed Red and Black","Coastal Alluvial","Deltaic Alluvium","Calcerous","Gray Brown","Desert","Tarai","Black (Karail)","Saline and Alkaline","Alluvial River","Skeletal","Saline and Deltaic","Red","Red and Gravely"]
df_['SoilType'] = df_.apply(lambda row : labels[row["SoilType"]],axis=1)
for crop in major_crops:
  if "Moong"==crop:
    continue
    
  mx_prod = 0
  df1 = df_[df_["Crop"]==crop].reset_index(drop=True)
  if df1.shape[0]!=0:
    print(df1)
    # y = [df1.loc[df1["SoilType"]==x,:].reset_index().loc[0,"Production/Area"] for x in labels[2:]]
    plt.plot(df1["SoilType"],df1["Production/Area"])
    plt.xticks(rotation='vertical')
    plt.title("Average yield vs Ph for "+str(crop))
    plt.ylabel("Average yield")
    plt.xlabel("Ph")
    plt.show()
  soil_yield = []
  for i in range(df1.shape[0]):
    soil_yield.append((df1.loc[i,"SoilType"],df1.loc[i,"Production/Area"]))
  
  sorted_soil_type = sorted(soil_yield,key = lambda x: x[1],reverse=True)
  opt_soils = [x[0] for x in sorted_soil_type[:3] if x[0]!='']
  s="\&".join(opt_soils)
  optim_soil[crop] = s
  # for i in range(df1.shape[0]):
  #   if df1.loc[i,"Production/Area"]>mx_prod:
  #     mx_prod=df1.loc[i,"Production/Area"]
  #     opt_cat = df1.loc[i,"SoilType"]
  
  # optim_soil[crop] = opt_cat


df_ = df.groupby(['Ph','Crop'])['Production/Area'].agg('mean').reset_index()
crops = set(df_["Crop"].tolist())
optim_ph = {}
ph_dict = { 4 :"4-5",5:"5-6",6:"6-7",7:"7-8",8:"8-9"}
opt_cat = ""
labels = ["","","Laterite","Red and Yellow","Shallow Black","Medium Black","Deep Black","Mixed Red and Black","Coastal Alluvial","Deltaic Alluvium","Calcerous","Gray Brown","Desert","Tarai","Black (Karail)","Saline and Alkaline","Alluvial River","Skeletal","Saline and Deltaic","Red","Red and Gravely"]

for crop in major_crops:
  mx_prod = 0
  df1 = df_[df_["Crop"]==crop].reset_index(drop=True)
  if df1.shape[0]!=0:
    print(df1)
    df1["ph_r"] = df1.apply(lambda row: row["Ph"]+0.5,axis=1)
    # y = [df1.loc[df1["SoilType"]==x,:].reset_index().loc[0,"Production/Area"] for x in labels[2:]]
    plt.plot(df1["ph_r"],df1["Production/Area"])
    plt.xticks(rotation='vertical')
    plt.title("Average yield vs pH for "+str(crop))
    plt.ylabel("Average yield")
    plt.xlabel("pH")
    plt.show()
  
  for i in range(df1.shape[0]):
    
    if df1.loc[i,"Production/Area"]>mx_prod:
      mx_prod=df1.loc[i,"Production/Area"]
      opt_cat = df1.loc[i,"Ph"]+0.5
      print(df1.loc[i,"Production/Area"])
      print(df1.loc[i,"Ph"])
    print(i,opt_cat)
  
  optim_ph[crop] = opt_cat

# df1 = merged_csv.groupby(['rain_cat','Crop'])['Production'].agg('median').reset_index()
# df1 = df1[df1["Crop"]=="Jute"]
# plt.plot(df["rain_cat"],df["Production"])
# plt.xticks(range(1,cat_max+1))
# plt.show()
# sns.boxplot(x="rain_cat", y="Production", data=df1,showfliers=False)
# merged_csv
print(optim_ph)
# print(crops)
l = [[a,optim_soil[a],optim_ph[a]] for a in list(optim_soil.keys())]
df2 = pd.DataFrame(l,columns=["Crop","Soil Type","PH"])
df2.to_csv("Dataset/ph.csv")

df3 = merged_csv 

map_district_crop = {}
map_crop_district = {}

for k,v in map_district_rainfall.items():
  if v in map_rainfall_crop:
    map_district_crop[k] = map_rainfall_crop[v]
    map_crop_district[map_rainfall_crop[v]] = k
l = district_df["DISTNAME"].tolist()
l1 = list(map_crop_district.keys())
for crop in l1:
  if map_crop_district[crop] not in l:
    del map_crop_district[crop] 

del map_crop_district['ETAH']
df3 = df3[(df3['District_Name'].isin(map_crop_district.keys()))].reset_index(drop=True)

df3 = df3[(df3['District_Name'].isin(map_crop_district.keys()))].reset_index(drop=True)
df3['Production/Area'] = df3.apply(lambda row: row['Production']/row['Area'] if row['Area']!=0 else np.nan, axis=1)
df3['SoilType'] = df3.apply(lambda row: district_soil[map_crop_district[row['District_Name']]],axis = 1)
# df3['SoilType'] = df3.apply(lambda row : labels[row["SoilType"]],axis=1)
labels = ["","","Laterite","Red and Yellow","Shallow Black","Medium Black","Deep Black","Mixed Red and Black","Coastal Alluvial","Deltaic Alluvium","Calcerous","Gray Brown","Desert","Tarai","Black (Karail)","Saline and Alkaline","Alluvial River","Skeletal","Saline and Deltaic","Red","Red and Gravely"]
df3['SoilType'] = df3.apply(lambda row : labels[row["SoilType"]],axis=1)

df3 = df3[['Crop','Production/Area','Season','SoilType','rainfall']]
# states = set(crop_prod['State_Name'].tolist())
df3 = pd.get_dummies(df3,columns = ["Season","SoilType"],prefix = ['Season_is','SoilType_is'])
# crop_prod = crop_prod[~(crop_prod['Season_is_Whole Year ']==1)].reset_index(drop=True)
print(df3.columns)

df3 = df3.groupby(['Crop'])[ 'Season_is_Autumn     ',
       'Season_is_Kharif     ', 'Season_is_Rabi       ',
       'Season_is_Summer     ', 'Season_is_Whole Year ',
       'Season_is_Winter     ', 'SoilType_is_', 'SoilType_is_Alluvial River',
       'SoilType_is_Black (Karail)', 'SoilType_is_Calcerous',
       'SoilType_is_Coastal Alluvial', 'SoilType_is_Deep Black',
       'SoilType_is_Deltaic Alluvium', 'SoilType_is_Desert',
       'SoilType_is_Gray Brown', 'SoilType_is_Medium Black',
       'SoilType_is_Mixed Red and Black', 'SoilType_is_Red',
       'SoilType_is_Red and Gravely', 'SoilType_is_Red and Yellow',
       'SoilType_is_Saline and Alkaline', 'SoilType_is_Saline and Deltaic',
       'SoilType_is_Shallow Black', 'SoilType_is_Skeletal',
       'SoilType_is_Tarai','rainfall'].agg('mean').reset_index()

print(df3.shape)
df3
# # crops = crop_prod['Crop'].tolist()
# # for crop in crops:
# #   for i in states:
# #     s= 'State_is_'+i
# #     # print(crop_prod[crop_prod['Crop']==crop][s])
# #     crop_prod.loc[crop_prod['Crop']==crop,s] = 0 
# #     # print(crop_prod.loc[crop_prod['Crop']==crop,s])
# #   for i in crop_state[crop]:
# #     s = 'State_is_'+i
# #     crop_prod.loc[crop_prod['Crop']==crop,s] = 1

df_ = df3.drop(columns = ['Crop'],axis=0)
mm_scaler = preprocessing.MinMaxScaler()
l = mm_scaler.fit_transform(df_)
df_ = pd.DataFrame(l, index=df_.index, columns=df_.columns)
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=10000)
tsne_results = tsne.fit_transform(df_)
tsne_results = pd.DataFrame(tsne_results,columns= ['x','y'])
model = KMeans(n_clusters=8).fit(df_)
y_pred = model.labels_
cluster_crops = {}
crops = df3['Crop'].tolist()
for i in range(len(y_pred)):
  if y_pred[i] in cluster_crops:
    cluster_crops[y_pred[i]].append(crops[i])
  else:
    cluster_crops[y_pred[i]] = [crops[i]]
# for i in range():
#   print(cluster_crops[i])
print(cluster_crops)
centroids  = model.cluster_centers_ 
print(centroids)

plt.scatter(tsne_results['x'],tsne_results['y'],c=y_pred.astype('float'))
plt.show()

for i in range(8):
  print(cluster_crops[i])
# df3[df3["Crop"].isin(cluster_crops[1])]

df3[df3["Crop"].isin(cluster_crops[2])]

sil = []
kmax = 10
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k,max_iter=10000).fit(df_)
  labels = kmeans.labels_
  sil.append(silhouette_score(df_, labels, metric = 'euclidean'))
plt.plot(range(2,kmax+1),sil)
plt.xlabel("No. of clusters")
plt.ylabel("Silhouette score")
plt.show()

df3 = merged_csv 

map_district_crop = {}
map_crop_district = {}

for k,v in map_district_rainfall.items():
  if v in map_rainfall_crop:
    map_district_crop[k] = map_rainfall_crop[v]
    map_crop_district[map_rainfall_crop[v]] = k
l = district_df["DISTNAME"].tolist()
l1 = list(map_crop_district.keys())
for crop in l1:
  if map_crop_district[crop] not in l:
    del map_crop_district[crop] 

del map_crop_district['ETAH']
df3 = df3[(df3['District_Name'].isin(map_crop_district.keys()))].reset_index(drop=True)

df3 = df3[(df3['District_Name'].isin(map_crop_district.keys()))].reset_index(drop=True)
df3['Production/Area'] = df3.apply(lambda row: row['Production']/row['Area'] if row['Area']!=0 else np.nan, axis=1)
df3['SoilType'] = df3.apply(lambda row: district_soil[map_crop_district[row['District_Name']]],axis = 1)
df3['Ph'] = df3.apply(lambda row: district_ph[map_crop_district[row['District_Name']]] ,axis = 1)
# df3['SoilType'] = df3.apply(lambda row : labels[row["SoilType"]],axis=1)
labels = ["","","Laterite","Red and Yellow","Shallow Black","Medium Black","Deep Black","Mixed Red and Black","Coastal Alluvial","Deltaic Alluvium","Calcerous","Gray Brown","Desert","Tarai","Black (Karail)","Saline and Alkaline","Alluvial River","Skeletal","Saline and Deltaic","Red","Red and Gravely"]
df3['SoilType'] = df3.apply(lambda row : labels[row["SoilType"]],axis=1)
min_rain = df3['rainfall'].min(axis=0)
max_rain = df3['rainfall'].max(axis=0)
diff = (max_rain-min_rain)/10
cats = [(min_rain+i*diff,min_rain+(i+1)*diff) for i in range(10)]
print(cats)
df3['rain_cat'] = df3.apply(lambda row:get_rain_category(row["rainfall"],min_rain,max_rain,10),axis=1)
df3 = df3[['State_Name','Production/Area','SoilType','Ph','rain_cat']]
df3 = pd.get_dummies(df3,columns = ["rain_cat","SoilType"],prefix = ['rain_cat_is','SoilType_is'])
print(df3.columns)
df3 = df3.groupby(['State_Name'])[ 'Production/Area', 'Ph', 'rain_cat_is_1', 'rain_cat_is_2',
       'rain_cat_is_3', 'rain_cat_is_4', 'rain_cat_is_5', 'rain_cat_is_6',
       'rain_cat_is_7', 'rain_cat_is_8', 'rain_cat_is_9', 'rain_cat_is_10',
       'rain_cat_is_11',
       'SoilType_is_', 'SoilType_is_Alluvial River',
       'SoilType_is_Black (Karail)', 'SoilType_is_Calcerous',
       'SoilType_is_Coastal Alluvial', 'SoilType_is_Deep Black',
       'SoilType_is_Deltaic Alluvium', 'SoilType_is_Desert',
       'SoilType_is_Gray Brown', 'SoilType_is_Medium Black',
       'SoilType_is_Mixed Red and Black', 'SoilType_is_Red',
       'SoilType_is_Red and Gravely', 'SoilType_is_Red and Yellow',
       'SoilType_is_Saline and Alkaline', 'SoilType_is_Saline and Deltaic',
       'SoilType_is_Shallow Black', 'SoilType_is_Skeletal',
       'SoilType_is_Tarai'].agg('mean').reset_index()


df_ = df3.drop(columns = ['State_Name'],axis=0)
mm_scaler = preprocessing.MinMaxScaler()
l = mm_scaler.fit_transform(df_)
df_ = pd.DataFrame(l, index=df_.index, columns=df_.columns)
tsne = TSNE(n_components=2, verbose=1, perplexity=8, n_iter=10000)
tsne_results = tsne.fit_transform(df_)
tsne_results = pd.DataFrame(tsne_results,columns= ['x','y'])
model = KMeans(n_clusters=10,max_iter=100000000).fit(df_)
print(model.get_params)
y_pred = model.labels_
cluster_crops = {}
crops = df3['State_Name'].tolist()
for i in range(len(y_pred)):
  if y_pred[i] in cluster_crops:
    cluster_crops[y_pred[i]].append(crops[i])
  else:
    cluster_crops[y_pred[i]] = [crops[i]]
# for i in range():
#   print(cluster_crops[i])
print(cluster_crops)
centroids  = model.cluster_centers_ 
print(centroids)

plt.scatter(tsne_results['x'],tsne_results['y'],c=y_pred.astype('float'))
plt.show()

for i in range(10):
  print(cluster_crops[i])

"""cluster 0 - rainfall - 1,2, soil majorly - alluvial, ph - 6-8
cluster 1 - rainfall - 3,4 , soil majorly - red, ph - 4-6
cluster 2 - rainfall - 2,3, soil majorly - deep black,ph - 5-7
cluster 3 - rainfall - 7,8,9,10, soil majorly - saline and alkaline
cluster 4 - rainfall - 2,3,4
cluster 5 - rainfall - 4,5 , soil type - calcerous, tarai, ph - 5-6
cluster 6 - rainfall - 4,5,6,7, soil type - alluvial, deep black, ,ph - 5-8 
cluster 7 - rainfall - 3,4, soil type majorly - red and yellow, ph - 4-7
cluster 8 - rainfall - 2,3, soil type majorly alluvial, ph - 5.5-7.5
cluster 9 - rainfall - 7,8,9 , soil type majorly red, ph - 4-5 
"""

df3[df3["State_Name"].isin(cluster_crops[9])]

# sil = []
# kmax = 15
# for k in range(2, kmax+1):
#   kmeans = KMeans(n_clusters = k,max_iter=1000000000).fit(df_)
#   labels = kmeans.labels_
#   sil.append(silhouette_score(df_, labels, metric = 'euclidean'))

plt.plot(range(2,kmax+1),sil)
plt.xlabel("No. of clusters")
plt.ylabel("Silhouette Score")
plt.show()

df3[df3['State_Name'].isin(cluster_crops[7])]

district_df['rainfall'] = district_df.apply(lambda row:dist_rainfall[map_district_rainfall[row['DISTNAME']]],axis=1)
district_df = district_df.drop(['Unnamed: 0'],axis=1)
district_df

df = district_df.drop(['DISTNAME'],axis=1)
mm_scaler = preprocessing.MinMaxScaler()
l = mm_scaler.fit_transform(df)
df = pd.DataFrame(l, index=df.index, columns=df.columns)
# mm_scaler.transform(X_test)
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=10000)
tsne_results = tsne.fit_transform(df)
tsne_results = pd.DataFrame(tsne_results,columns= ['x','y'])
model = KMeans(n_clusters=13).fit(df)
y_pred = model.labels_
cluster_crops = {}
crops = crop_prod['Crop'].tolist()
for i in range(len(y_pred)):
  if y_pred[i] in cluster_crops:
    cluster_crops[y_pred[i]].append(district_df.loc[i,"DISTNAME"])
  else:
    cluster_crops[y_pred[i]] = [district_df.loc[i,"DISTNAME"]]
# for i in range():
#   print(cluster_crops[i])
print(cluster_crops)
centroids  = model.cluster_centers_ 
print(centroids)

plt.scatter(tsne_results['x'],tsne_results['y'],c=y_pred.astype('float'))
plt.show()

for i in range(13):
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  phs = range(3,9)
  df = district_df[(district_df['DISTNAME'].isin(cluster_crops[i]))].reset_index(drop=True)
  dists = [df[df['SoilPh']==x].shape[0] for x in range(3,9)]

  bars =ax.bar(phs,dists)
  plt.xticks(rotation='vertical')
  plt.ylabel('Ph of crop(Rs/Hectare)')
  plt.show()

"""**Production share of each crop category in India**"""

residuals = set([])
def get_cat(crop):
  if crop in ['Rice','Maize','Wheat','Barley','Varagu','Other Cereals & Millets','Ragi','Small millets','Bajra','Jowar','Paddy']:
    return 'Cereal'
  if crop in ['Moong','Moong(Green Gram)','Urad','Arhar/Tur','Peas & beans','Masoor',
              'Other Kharif pulses','other misc. pulses','Ricebean (nagadal)',
              'Rajmash Kholar','Lentil','Samai','Blackgram','Korra','Cowpea(Lobia)',
              'Other  Rabi pulses','Other Kharif pulses','Peas & beans (Pulses)','Pulses total']:
    return 'Pulses'
  if crop in ['Peach','Apple','Litchi','Pear','Plums','Ber','Sapota','Lemon','Pome Granet',
               'Other Citrus Fruit','Water Melon','Jack Fruit','Grapes','Pineapple','Orange',
               'Pome Fruit','Citrus Fruit','Other Fresh Fruits','Mango','Papaya','Coconut','Banana','Other Dry Fruit']:
    return 'Fruits'
  if crop in ['Bean','Lab-Lab','Moth','Guar seed','Tapioca','Soyabean','Horse-gram','Gram']:
    return 'Beans'
  if crop in ['Turnip','Peas','Beet Root','Carrot','Yam','Ribed Guard','Ash Gourd ','Pump Kin','Redish','Snak Guard','Bottle Gourd',
              'Bitter Gourd','Cucumber','Drum Stick','Cauliflower','Beans & Mutter(Vegetable)','Cabbage',
              'Bhindi','Tomato','Brinjal','Khesari','Sweet potato','Potato','Onion','Other Vegetables','Peas  (vegetable)']:
    return 'Vegetables'
  if crop in ['other fibres','Kapas','Jute & mesta','Jute','Mesta','Cotton(lint)']:
    return 'Fibres'
  if crop in ['Arcanut (Processed)','Atcanut (Raw)','Cashewnut Processed','Cashewnut Raw','Cashewnut','Arecanut','Groundnut']:
    return 'Nuts'
  if crop in ['Tea','Coffee']:
    return 'Tea-Coffee'
  if crop in ['Castor seed','Niger seed','other oilseeds']:
    return "Seeds"
  if crop=='Sugarcane':
    return 'Sugarcane'
  if crop =='Coconut':
    return 'Coconut'
  if "Total" in crop:
    return "Totals"
  else:
    # print(crop)
    residuals.add(crop)
    return "Others"
  

crop_prod = pd.read_csv('Dataset/apy.csv',delimiter=',')
crop_prod['crop_cat'] = crop_prod.apply(lambda row: get_cat(row['Crop']),axis=1)
df = crop_prod.groupby(['crop_cat','Crop_Year'])['Production'].agg('sum').reset_index()
df = df.groupby(['crop_cat'])['Production'].agg('mean').reset_index()
df = df[~(df['crop_cat']=='Others')].reset_index()
type_cnt = df['Production']
type_labels = df['crop_cat']
print(residuals)
plt.rcParams['font.size'] = 14.0
plt.pie(type_cnt, labels=type_labels,radius = 3, autopct='%1.1f%%', shadow=True)
# plt.savefig('Dataset/Prod_share.jpg')

residuals = set([])
crop_prod = pd.read_csv('Dataset/apy.csv',delimiter=',')
crop_prod['crop_cat'] = crop_prod.apply(lambda row: get_cat(row['Crop']),axis=1)
df = crop_prod.groupby(['crop_cat','Crop_Year'])['Area'].agg('sum').reset_index()
df = df.groupby(['crop_cat'])['Area'].agg('mean').reset_index()
df = df[~(df['crop_cat']=='Others')].reset_index()
type_cnt = df['Area']
type_labels = df['crop_cat']
print(residuals)
plt.rcParams['font.size'] = 14.0
plt.pie(type_cnt, labels=type_labels,radius = 3, autopct='%1.1f%%', shadow=True)
# plt.savefig('Dataset/Prod_share.jpg')

"""**Crop Price over years**"""

revenue = pd.read_csv('Dataset/revenue.csv',delimiter=',')
revenue['Revenue per hectare'] = revenue.apply(lambda row: row['Revenue per hectare']*10, axis=1)
plts = []
leg = []
price_slope = {}
crops = set(revenue['Crop'].tolist())
print(sorted(list(crops)))
plt.figure(figsize=(15,10))
for crop in crops:
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  # print(df.loc[df['Crop_Year']==2002,'Price per quintal'])
  st = df.loc[df['Crop_Year']==2002,:].reset_index().loc[0,'Price per quintal']
  fin = df.loc[df['Crop_Year']==2012,:].reset_index().loc[0,'Price per quintal']
  # print(st)
  price_slope[crop] =  (fin - st)/10
  x, = plt.plot(df['Crop_Year'],df['Price per quintal'])
  plts.append(x)
  leg.append(crop)
# plt
plt.xticks(range(2002, 2013))
plt.xlabel('Year')
plt.ylabel('Price/quintal(Rs)')
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor=(1.2, 0))
plt.savefig('Dataset/crop_prices.png')
plt.show()

print(len(crops))
print(price_slope)


for crop in crops:
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  df['price change%'] = df['Price per quintal'].pct_change().mul(100).round(2)
  # print(df.loc[df['Crop_Year']==2002,'Price per quintal'])
  st = df.loc[df['Crop_Year']==2002,:].reset_index().loc[0,'Price per quintal']
  fin = df.loc[df['Crop_Year']==2012,:].reset_index().loc[0,'Price per quintal']
  # print(st)
  price_slope[crop] =  (fin - st)/10
  x, = plt.plot(df['Crop_Year'],df['price change%'])
  plts.append(x)
  leg.append(crop)
plt.xticks(range(2002, 2013))
plt.xlabel('Year')
plt.ylabel('Price change/quintal(Rs)')
plt.savefig('Dataset/crop_price_chng.png')
plt.show()



revenue = pd.read_csv('Dataset/revenue.csv',delimiter=',')
plts = []
leg = []
crops = set(revenue['Crop'].tolist())
plt.figure(figsize=(15,10))
for crop in crops:
  
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  # print(df)
  if crop in ['Sugarcane','Wheat','Rice']:
    x, =plt.plot(df['Crop_Year'],df['Production in quintals'])
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['Production in quintals'])
    plts.append(x)
    leg.append(crop)

plt.ylabel('Production of crop')
plt.xlabel('Year')
plt.xticks(range(2002,2013))
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.12,0))

plt.show()

revenue = pd.read_csv('Dataset/revenue.csv',delimiter=',')
plts = []
leg = []
crops = set(revenue['Crop'].tolist())
plt.figure(figsize=(15,10))
for crop in crops:
  
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  # print(df)
  if crop in ['Sugarcane','Wheat','Rice']:
    x, =plt.plot(df['Crop_Year'],df['Area in Hectares'])
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['Area in Hectares'])
    plts.append(x)
    leg.append(crop)

plt.ylabel('Cultivation Area of crop(Hectares)')
plt.xlabel('Year')
plt.xticks(range(2002,2013))
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.2,0))

plt.show()

revenue

revenue = pd.read_csv('Dataset/revenue.csv',delimiter=',')
revenue['Revenue per hectare'] = revenue.apply(lambda row: row['Revenue per hectare']*10, axis=1)

revenue['yield'] = revenue.apply(lambda row: row['Production in quintals']*10/row['Area in Hectares'] if row['Crop']!='Sugarcane' else row['Production in quintals']*10/(10*row['Area in Hectares']),axis=1)
df = revenue.groupby(['Crop'])['yield'].agg('mean').reset_index()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
crop = df['Crop'].tolist()
yld = df['yield'].tolist()
bars =ax.bar(crop,yld)
bars[-4].set_color('black')
plt.xticks(rotation='vertical')
plt.ylabel('Yield of crop(Quintal/Hectare)')
plt.show()
revenue.columns

plts = []
leg = []
crops = set(revenue['Crop'].tolist())
print(crops)
plt.figure(figsize=(15,10))
for crop in crops:
  
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  # print(df)
  if crop in ['Sugarcane','Jute']:
    x, =plt.plot(df['Crop_Year'],df['yield']/3,'--')
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['yield'])
    plts.append(x)
    leg.append(crop)

plt.ylabel('Yield of crop(Quintals/Hectare)')
plt.xlabel('Year')
plt.xticks(range(2002,2013))
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.25,0))

plt.show()

plts = []
leg = []
crops = set(revenue['Crop'].tolist())
plt.figure(figsize=(15,10))
for crop in crops:
  
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  df['yield change%'] = df['yield'].pct_change().mul(100).round(2)
  # print(df)
  if crop in ['Sugarcane','Wheat','Rice']:
    x, =plt.plot(df['Crop_Year'],df['yield change%'])
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['yield change%'])
    plts.append(x)
    leg.append(crop)

plt.ylabel('% Yield growth of crop')
plt.xlabel('Year')
plt.xticks(range(2002,2013))
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.2,0))

plt.show()

df = revenue.groupby(['Crop'])['Revenue per hectare'].agg('mean').reset_index()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
crop = df['Crop'].tolist()
yld = df['Revenue per hectare'].tolist()
bars =ax.bar(crop,yld)
plt.xticks(rotation='vertical')
plt.ylabel('Revenue generated per hectare of crop(Rs/Hectare)')
plt.show()
revenue.columns

plts = []
leg = []
crops = set(revenue['Crop'].tolist())
plt.figure(figsize=(15,10))
for crop in crops:
  
  df = revenue[revenue['Crop']==crop].reset_index(drop=True)
  # print(df)
  if crop in ['Cotton(lint)','Jute','Sugarcane']:
    x, =plt.plot(df['Crop_Year'],df['Revenue per hectare']/5,'--')
    plts.append(x)
    leg.append(crop)
  else:
    x, =plt.plot(df['Crop_Year'],df['Revenue per hectare'])
    plts.append(x)
    leg.append(crop)

plt.ylabel('Revenue per hectare of crop(Rs/Hectare)')
plt.xlabel('Year')
plt.xticks(range(2002,2013))
plt.legend(plts,leg,loc='lower right',borderaxespad=0.,bbox_to_anchor = (1.25,0))

plt.show()

slope_rev = {}
crops = set(revenue['Crop'].tolist())
for crop in crops:
  df = revenue.loc[revenue['Crop']==crop,:].reset_index()
  st = df.loc[df['Crop_Year']==2002,:].reset_index().loc[0,"Revenue per hectare"]
  fn = df.loc[df['Crop_Year']==2012,:].reset_index().loc[0,"Revenue per hectare"]
  slope_rev[crop] = (math.pow(fn/st,0.1)-1)*100#((fn-st)/st)*100
  print(fn-st)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
crops_l = list(crops)
slopes = [slope_rev[crop] for crop in crops_l]
bars =ax.bar(crops_l,slopes)
plt.xticks(rotation='vertical')
plt.ylabel('Revenue growth slope per hectare of crop(Rs/Hectare)')
plt.show()