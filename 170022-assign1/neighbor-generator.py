import json
import pprint
import datetime
import pandas as pd
import statistics
from collections import defaultdict
import csv
di_avg = defaultdict(dict)
di_std = defaultdict(dict)

csv_file = open('cases-week.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
dff = {}

for row in csv_reader:
    if row[0] == "districtid":
        continue
    if row[0] in dff.keys():
        dff[row[0]][row[1]] = row[2]
    else:
        dff[row[0]] = {row[1]:row[2]}

f = open('neighbor-district-ids.txt') #use only district ids here
neighbours = eval(f.read())
# print(neighbours)
cw = open('neighbor-week.csv', 'w')

for d in range(101,828):
    for i in range(1,25):
        list = []
        d = str(d)
        try:
            for k in neighbours[d]:
                try:
                    y = int(dff[str(k)][str(i)])
                except KeyError:
                    y = 0
                list.append(y)
                # print(y)
            try:
                di_avg[d][i] = statistics.mean(list)
            except:
                di_avg[d][i] = 0
            try:
                di_std[d][i] = statistics.stdev(list)
            except:
                di_std[d][i] = 0
        except:
            pass

print('districtid,timeid,neighbormean,neighborstdev',file=cw)
for key in di_avg:
    for l in di_avg[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg[key][l],end = ",",file=cw)
        print("%.2f" % di_std[key][l],file=cw)

cw.close()

di_avg = defaultdict(dict)
di_std = defaultdict(dict)
csv_file2 = open('cases-month.csv')
csv_reader2 = csv.reader(csv_file2,delimiter = ',')
dff2 = {}

for row in csv_reader2:
    if row[0] == "districtid":
        continue
    if row[0] in dff2.keys():
        dff2[row[0]][row[1]] = row[2]
    else:
        dff2[row[0]] = {row[1]:row[2]}
# print(dff2)
cw = open('neighbor-month.csv', 'w')
for d in range(101,828):
    for i in range(1,8):
        list = []
        d = str(d)
        try:
            for k in neighbours[d]:
                try:
                    y = int(dff2[str(k)][str(i)])
                except KeyError:
                    y = 0
                list.append(y)

            try:
                di_avg[d][i] = statistics.mean(list)
            except:
                di_avg[d][i] = 0
            try:
                di_std[d][i] = statistics.stdev(list)
            except:
                di_std[d][i] = 0
        except:
            pass
# print(di_avg)
print('districtid,timeid,neighbormean,neighborstdev',file=cw)
for key in di_avg:
    for l in di_avg[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg[key][l],end = ",",file=cw)
        print("%.2f" % di_std[key][l],file=cw)
cw.close()

di_avg = defaultdict(dict)
di_std = defaultdict(dict)
csv_file3 = open('cases-overall.csv')
csv_reader3 = csv.reader(csv_file3,delimiter = ',')
dff3 = {}

for row in csv_reader3:
    if row[0] == "districtid":
        continue
    if row[0] in dff3.keys():
        dff3[row[0]][row[1]] = row[2]
    else:
        dff3[row[0]] = {row[1]:row[2]}

cw = open('neighbor-overall.csv', 'w')
for d in range(101,828):
        list = []
        i = 1
        d = str(d)
        try:
            for k in neighbours[d]:
                try:
                    y = int(dff3[str(k)][str(i)])
                except KeyError:
                    y = 0
                list.append(y)
            try:
                di_avg[d]['1'] = statistics.mean(list)
            except:
                di_avg[d]['1'] = 0
            try:
                di_std[d]['1'] = statistics.stdev(list)
            except:
                di_std[d]['1'] = 0
        except:
            pass

print('districtid,timeid,neighbormean,neighborstdev',file=cw)

for key in di_avg:
    for l in di_avg[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg[key][l],end = ",",file=cw)
        print("%.2f" % di_std[key][l],file=cw)
cw.close()
