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

cw = open('state-week.csv', 'w')
f = open('states-districts.txt')
states = eval(f.read())

for s in states:
    for i in range(1,25):
        x = states[s]
        for k in states[s]:
            list = []
            for j in x:
                if j != k:
                    try:
                        y = int(dff[str(j)][str(i)])
                    except KeyError:
                        y = 0
                    list.append(y)
            # print(y)
            try:
                di_avg[k][i] = statistics.mean(list)
            except:
                di_avg[k][i] = 0
            try:
                di_std[k][i] = statistics.stdev(list)
            except:
                di_std[k][i] = 0

print('districtid,timeid,statemean,statestdev',file=cw)
for key in sorted(di_avg):
    for l in di_avg[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg[key][l],end = ",",file=cw)
        print("%.2f" % di_std[key][l],file=cw)

cw.close()
dff2 = {}
di_avg2 = defaultdict(dict)
di_std2 = defaultdict(dict)
csv_file2 = open('cases-month.csv')
csv_reader2 = csv.reader(csv_file2,delimiter = ',')
for row in csv_reader2:
    if row[0] == "districtid":
        continue
    if row[0] in dff2.keys():
        dff2[row[0]][row[1]] = row[2]
    else:
        dff2[row[0]] = {row[1]:row[2]}
cw = open('state-month.csv', 'w')
for s in states:
    for i in range(1,8):
        x = states[s]
        for k in states[s]:
            list = []
            for j in x:
                if j != k:
                    try:
                        y = int(dff2[str(j)][str(i)])
                    except KeyError:
                        y = 0
                    list.append(y)
            try:
                di_avg2[k][i] = statistics.mean(list)
            except:
                di_avg2[k][i] = 0
            try:
                di_std2[k][i] = statistics.stdev(list)
            except:
                di_std2[k][i] = 0
print('districtid,timeid,statemean,statestdev',file=cw)
for key in sorted(di_avg2):
    for l in di_avg2[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg2[key][l],end = ",",file=cw)
        print("%.2f" % di_std2[key][l],file=cw)

cw.close()

dff3 = {}
csv_file3 = open('cases-overall.csv')
csv_reader3 = csv.reader(csv_file3,delimiter = ',')
di_avg3 = defaultdict(dict)
di_std3 = defaultdict(dict)
for row in csv_reader3:
    if row[0] == "districtid":
        continue
    if row[0] in dff3.keys():
        dff3[row[0]][row[1]] = row[2]
    else:
        dff3[row[0]] = {row[1]:row[2]}
cw = open('state-overall.csv', 'w')
for s in states:
        i = 1
        x = states[s]
        for k in states[s]:
            list = []
            for j in x:
                if j != k:
                    try:
                        y = int(dff3[str(j)][str(i)])
                    except KeyError:
                        y = 0
                    list.append(y)
            try:
                di_avg3[k][i] = statistics.mean(list)
            except:
                di_avg3[k][i] = 0
            try:
                di_std3[k][i] = statistics.stdev(list)
            except:
                di_std3[k][i] = 0
print('districtid,timeid,statemean,statestdev',file=cw)
for key in sorted(di_avg3):
    for l in di_avg3[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print("%.2f" % di_avg3[key][l],end = ",",file=cw)
        print("%.2f" % di_std3[key][l],file=cw)

cw.close()
