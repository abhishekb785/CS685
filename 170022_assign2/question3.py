import json
import pprint
import collections
import operator
import pandas as pd
import csv

csv_file = open('article-ids.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
aid = {}

for row in csv_reader:
    if row[0] == "﻿Article_Name":
        continue
    aid[row[0]] = row[1]

csv_file = open('category-ids.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
cid = {}

for row in csv_reader:
    if row[0] == "﻿Category_Name":
        continue
    cid[row[0]] = row[1]

sample = open('temp','r')
lines = sample.readlines()

dic = {}

for line in lines:
    a = line[:-1]
    k = a.split("\t")
    if aid[k[0]] in dic.keys():
        dic[aid[k[0]]].append(cid[k[1]])
    else:
        t = cid[k[1]]
        dic[aid[k[0]]] = [t]

for key in dic:
    dic[key].sort()

sd = open('article-categories.csv','w')
print('﻿Article ID,Category_ID',file=sd)
#
# for key in sorted(dic):
#     print(key, end=",", file=sd)
#     print(*dic[key],file=sd)


# print(dic)
for i in range(1,4605):
    key = "A" + "{:04n}".format(i)
    if key in dic.keys():
        print(key, end=",",file=sd)
        print(*dic[key],file=sd)
    else:
        print(key + ",C0001",file =sd)

    # for i in dic[key]:
        # print(i,end=" ", file=sd)
    # print("",file =sd)
