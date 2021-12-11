import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
xls = pd.ExcelFile(r"Dataset/india87.xls") #use r before absolute file path 
df = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

# var1 = df['ColumnName']

cols=df.columns
# for i in range(0,len(cols)):
#     print(i,cols[i])

soils=[]
for i in range(0,20):
    soils+=["DMS"+str("{0:0=2d}".format(i+1))]
# DMPH4
ph=[]
for i in range(4,9):
    ph+=["DMPH"+str(i)]
# finaldata=[]
df["SoilType"]=0
df["SoilPh"]=0

for index, row in df.iterrows(): 
    # print (row)
    for ind in range(2,len(soils)):
        if str(row[soils[ind]])[0]=="1":
            print(row[soils[ind]],df.loc[index,"DISTNAME"])
            df.loc[index,"SoilType"]=ind+1
            # break
    for ind in range(0,len(ph)):
        if str(row[ph[ind]])[0]=="1":
            df.loc[index,"SoilPh"]=ind+4
    # break

newdf=df[["DISTNAME","SoilType","SoilPh"]]
print(newdf)
newdf.to_csv("Dataset/district_soildata.csv", sep=',')

# x=df["SoilType"]
# y=df["SoilPh"]
# xy=np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
# # plt.scatter(x,y)
# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# # plt.show()
# plt.xlabel("Soiltype")
# plt.ylabel("SoilPh")
# plt.show()
# print(soils)
# print(ph)
# for col in cols:
#     if col.find("DMS")!=-1:
#         print(col)
# print(cols) #1 is the row number...