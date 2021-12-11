import json
import pprint
import datetime
import pandas as pd
import statistics
from collections import defaultdict
import csv
csv_file = open('cases-week.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = row[2]
    else:
        df[row[0]] = {row[1]:row[2]}

csv_file = open('neighbor-week.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df2 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df2.keys():
        df2[row[0]][row[1]] = [row[2],row[3]]
    else:
        df2[row[0]] = {row[1]:[row[2],row[3]]}

csv_file = open('state-week.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df3 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df3.keys():
        df3[row[0]][row[1]] = [row[2],row[3]]
    else:
        df3[row[0]] = {row[1]:[row[2],row[3]]}

# print(df2)
# exit()
di_z1 = defaultdict(dict)
di_z2 = defaultdict(dict)
cw = open('zscore-week.csv', 'w')
for d in range(101,828):
    for i in range(1,25):
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            f = df2[d][i]
        except KeyError:
            continue
        try:
            ff = df3[d][i]
        except KeyError:
            continue
        try:
            u = float(f[0])
        except:
            u = 0
        try:
            sig = float(f[1])
        except:
            sig = 0
        try:
            x = int(t)
        except:
            x = 0
        try:
            z = (x - u )/sig
        except:
            z = 0

        di_z1[d][i] = z
        try:
            u2 = float(ff[0])
        except:
            u2 = 0
        try:
            sig2 = float(ff[1])
        except:
            sig2 = 0
        try:
            z2 = (x - u2 )/sig2
        except:
            z2 = 0
        di_z2[d][i] = z2

print('districtid,timeid,neighborhoodzscore,statezscore',file=cw)
for key in sorted(di_z1):
    for l in di_z1[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_z1[key][l],end = ",",file=cw)
        print("%.2f" % di_z2[key][l],file=cw)

cw.close()
csv_file = open('cases-month.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = row[2]
    else:
        df[row[0]] = {row[1]:row[2]}

csv_file = open('neighbor-month.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df2 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df2.keys():
        df2[row[0]][row[1]] = [row[2],row[3]]
    else:
        df2[row[0]] = {row[1]:[row[2],row[3]]}

csv_file = open('state-month.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df3 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df3.keys():
        df3[row[0]][row[1]] = [row[2],row[3]]
    else:
        df3[row[0]] = {row[1]:[row[2],row[3]]}

di_z1 = defaultdict(dict)
di_z2 = defaultdict(dict)
cw = open('zscore-month.csv', 'w')
f = open("districts.txt", "r")
for d in range(101,828):
    for i in range(1,8):
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            f = df2[d][i]
        except KeyError:
            continue
        try:
            ff = df3[d][i]
        except KeyError:
            continue
        try:
            u = float(f[0])
        except:
            u = 0
        try:
            sig = float(f[1])
        except:
            sig = 0
        try:
            x = int(t)
        except:
            x = 0
        try:
            z = (x - u )/sig
        except:
            z = 0

        di_z1[d][i] = z
        try:
            u2 = float(ff[0])
        except:
            u2 = 0
        try:
            sig2 = float(ff[1])
        except:
            sig2 = 0
        try:
            z2 = (x - u2 )/sig2
        except:
            z2 = 0
        di_z2[d][i] = z2

print('districtid,timeid,neighborhoodzscore,statezscore',file=cw)
for key in sorted(di_z1):
    for l in di_z1[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_z1[key][l],end = ",",file=cw)
        print("%.2f" % di_z2[key][l],file=cw)

cw.close()

csv_file = open('cases-overall.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = row[2]
    else:
        df[row[0]] = {row[1]:row[2]}

csv_file = open('neighbor-overall.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df2 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df2.keys():
        df2[row[0]][row[1]] = [row[2],row[3]]
    else:
        df2[row[0]] = {row[1]:[row[2],row[3]]}

csv_file = open('state-overall.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df3 = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df3.keys():
        df3[row[0]][row[1]] = [row[2],row[3]]
    else:
        df3[row[0]] = {row[1]:[row[2],row[3]]}

di_z1 = defaultdict(dict)
di_z2 = defaultdict(dict)
cw = open('zscore-overall.csv', 'w')
for d in range(101,828):
        i = 1
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            f = df2[d][i]
        except KeyError:
            continue
        try:
            ff = df3[d][i]
        except KeyError:
            continue
        try:
            u = float(f[0])
        except:
            u = 0
        try:
            sig = float(f[1])
        except:
            sig = 0
        try:
            x = int(t)
        except:
            x = 0
        try:
            z = (x - u )/sig
        except:
            z = 0

        di_z1[d][i] = z
        try:
            u2 = float(ff[0])
        except:
            u2 = 0
        try:
            sig2 = float(ff[1])
        except:
            sig2 = 0
        try:
            z2 = (x - u2 )/sig2
        except:
            z2 = 0
        di_z2[d][i] = z2

print('districtid,timeid,neighborhoodzscore,statezscore',file=cw)
for key in sorted(di_z1):
    for l in di_z1[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_z1[key][l],end = ",",file=cw)
        print("%.2f" % di_z2[key][l],file=cw)

cw.close()
