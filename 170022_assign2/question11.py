import json
import pprint
import collections
import operator
import pandas as pd
import csv
import math
from collections import defaultdict
import networkx as nx
import statistics

dic2D = {}
dic3D = {}
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
cid2 = {}

for row in csv_reader:
    if row[0] == "﻿Category_Name":
        continue
    cid[row[0]] = row[1]
    cid2[row[1]] = row[0]

csv_file = open('article-categories.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
acid = {}
for row in csv_reader:
    if row[0] == "﻿Article ID":
        continue
    lis = row[1].split(" ")
    acid[row[0]] = lis

sample = open('temp10','r')
lines = sample.readlines()
i = 1
shortest = {}
for line in lines:
    for j in range(0,len(line)-1):
        if i in shortest.keys():
            shortest[i][j+1] = line[j]
        else:
            shortest[i] = {j+1:line[j]}
    i = i+1

# print(shortest[11])
# exit()
sample = open('temp4','r')
lines = sample.readlines()

for line in lines:
    a = line[:-1]
    lis = a.split(";")
    f = aid[lis[0]]
    l = aid[lis[-1]]
    source = []
    destination = []
    n = a.count('<')
    f1 = int(aid[lis[0]][1:])
    l1 = int(aid[lis[-1]][1:])
    h1 = len(lis)-1
    h2 = h1 - 2*n
    # print(f1,l1)
    if shortest[f1][l1] != '_' and shortest[f1][l1]!= '0':
        s1 =int(shortest[f1][l1])

    if f in acid:
        for cats in acid[f]:
            a = cid2[cats]
            n = a.count('.') +1
            for i in range(1,n):
                k = ".".join(a.split(".", i)[:i])
                cat = cid[k]
                if cat not in source:
                    source.append(cat)
            if cats not in source:
                source.append(cats)
    if l in acid:
        for cats in acid[l]:
            a = cid2[cats]
            n = a.count('.') +1
            for i in range(1,n):
                k = ".".join(a.split(".", i)[:i])
                cat = cid[k]
                if cat not in destination:
                    destination.append(cat)
            if cats not in destination:
                destination.append(cats)

    val = round(h2/s1,2)

    for s in source:
        for d in destination:
            if s in dic2D.keys():
                if d in dic2D[s].keys():
                    dic2D[s][d].append(val)
                else:
                    dic2D[s][d] = [val]
            else:
                dic2D[s] = {d:[val]}

sd = open('category-ratios.csv','w')
print('﻿From_Category,To_Category,Ratio_of_human_to_shortest',file=sd)

for i in sorted(dic2D):
    for j in sorted(dic2D[i]):
        a = statistics.mean(dic2D[i][j])
        print(i,j,round(a,2),sep=",",file =sd)
