import json
import pprint
from datetime import datetime
from collections import defaultdict
import pandas as pd
import csv

d = defaultdict(dict)
m = defaultdict(dict)
overall = {}
ff = datetime.strptime('10/03/2020','%d/%m/%Y')
start_date = int(ff.strftime("%U"))
cw = open('cases-week.csv', 'w')
cm = open('cases-month.csv','w')
co = open('cases-overall.csv','w')
csv_file = open('district-id.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
df = {}

for row in csv_reader:
    if row[0] == "district":
        continue
    if row[0] in df.keys():
        df[row[0]][row[1]] = int(row[2])
    else:
        df[row[0]] = {row[1]:int(row[2])}

# print(df)
# exit()

for r in range(1,15):
      sample = 'raw_data' + str(r) + '.json'
      f = open(sample)
      data = json.load(f)
      data = data['raw_data']
      for key in data:
        if key['currentstatus'] == 'Hospitalized':
            ll = datetime.strptime(key['dateannounced'],'%d/%m/%Y')
            w = int(ll.strftime("%U"))
            w = w - start_date
            if w > 0:
                num = int(key['numcases'])
                dist = key['detecteddistrict']
                state = key['detectedstate']
                try:
                    xx = df[dist][state]
                except:
                    continue
                try:
                    d[xx][w] = num + dp[xx][w]
                except:
                    try:
                        d[xx][w] = num
                    except:
                        continue
                try:
                    m[xx][ll.month-2] += num
                except:
                    try:
                        m[xx][ll.month-2] = num
                    except:
                        continue
                try:
                    overall[xx] += num
                except:
                    try:
                        overall[xx] = num
                    except:
                        continue

print('districtid,timeid,cases', file = cw)
print('districtid,timeid,cases', file = cm)
print('districtid,timeid,cases', file = co)
d1 = defaultdict(dict)
m1 = defaultdict(dict)
overall1 = {}
for key in d:
    for l in d[key]:
        if d[key][l]>=0:
            d1[key][l] = d[key][l]

for key in m:
    for l in m[key]:
        if m[key][l]>=0:
            m1[key][l] = m[key][l]

for key in overall:
        if overall[key]>=0:
            overall1[key] = overall[key]

for key in sorted(d1):
    for l in d1[key]:
        print(key, end = ",", file = cw)
        print(l, end = ",",file=cw)
        print(d1[key][l],file=cw)

for key in sorted(m1):
    for l in m1[key]:
        print(key, end = ",", file = cm)
        print(l, end = ",",file=cm)
        print(m1[key][l],file=cm)

for key in sorted(overall1):
        print(key, end = ",", file = co)
        print("1", end = ",",file=co)
        print(overall1[key],file=co)
