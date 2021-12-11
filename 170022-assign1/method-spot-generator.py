import json
import pprint
import datetime
import pandas as pd
import statistics
from collections import defaultdict
import csv
csv_file = open('zscore-week.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = [row[2],row[3]]
    else:
        df[row[0]] = {row[1]:[row[2],row[3]]}
# print(df)
# exit()
di_n = defaultdict(dict)
di_s = defaultdict(dict)
cw = open('method-spot-week.csv', 'w')
for d in range(101,828):
    for i in range(1,25):
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            u = float(t[0])
        except:
            u = 0
        if u > 1:
            di_n[i][d] = 'hot'
        elif u < -1:
            di_n[i][d] = 'cold'
        try:
            v = float(t[1])
        except:
            v = 0
        if v > 1:
            di_s[i][d] = 'hot'
        elif v < -1:
            di_s[i][d] = 'cold'

print('timeid,method,spot,districtid',file=cw)
for key in sorted(di_n):
    for l in di_n[key]:
        print(key, end = ",", file = cw)
        print('neighborhood', end = ",",file=cw)
        print(di_n[key][l],end = ",",file=cw)
        print(l,file=cw)
for key in sorted(di_s):
    for l in di_s[key]:
        print(key, end = ",", file = cw)
        print('state', end = ",",file=cw)
        print(di_s[key][l],end = ",",file=cw)
        print(l,file=cw)
cw.close()
# exit()
csv_file = open('zscore-month.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = [row[2],row[3]]
    else:
        df[row[0]] = {row[1]:[row[2],row[3]]}

di_n = defaultdict(dict)
di_s = defaultdict(dict)
cw = open('method-spot-month.csv', 'w')
for d in range(101,828):
    for i in range(1,8):
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            u = float(t[0])
        except:
            u = 0
        if u > 1:
            di_n[i][d] = 'hot'
        elif u < -1:
            di_n[i][d] = 'cold'
        try:
            v = float(t[1])
        except:
            v = 0
        if v > 1:
            di_s[i][d] = 'hot'
        elif v < -1:
            di_s[i][d] = 'cold'
print('timeid,method,spot,districtid',file=cw)
for key in sorted(di_n):
    for l in di_n[key]:
        print(key, end = ",", file = cw)
        print('neighborhood', end = ",",file=cw)
        print(di_n[key][l],end = ",",file=cw)
        print(l,file=cw)
for key in sorted(di_s):
    for l in di_s[key]:
        print(key, end = ",", file = cw)
        print('state', end = ",",file=cw)
        print(di_s[key][l],end = ",",file=cw)
        print(l,file=cw)
cw.close()
csv_file = open('zscore-overall.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}
for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = [row[2],row[3]]
    else:
        df[row[0]] = {row[1]:[row[2],row[3]]}
di_n = defaultdict(dict)
di_s = defaultdict(dict)
cw = open('method-spot-overall.csv', 'w')
for d in range(101,828):
        i = 1
        d = str(d)
        i = str(i)
        try:
            t = df[d][i]
        except KeyError:
            continue
        try:
            u = float(t[0])
        except:
            u = 0
        if u > 1:
            di_n[i][d] = 'hot'
        elif u < -1:
            di_n[i][d] = 'cold'
        try:
            v = float(t[1])
        except:
            v = 0
        if v > 1:
            di_s[i][d] = 'hot'
        elif v < -1:
            di_s[i][d] = 'cold'

print('timeid,method,spot,districtid',file=cw)
for key in sorted(di_n):
    for l in di_n[key]:
        print(key, end = ",", file = cw)
        print('neighborhood', end = ",",file=cw)
        print(di_n[key][l],end = ",",file=cw)
        print(l,file=cw)
for key in sorted(di_s):
    for l in di_s[key]:
        print(key, end = ",", file = cw)
        print('state', end = ",",file=cw)
        print(di_s[key][l],end = ",",file=cw)
        print(l,file=cw)
cw.close()
