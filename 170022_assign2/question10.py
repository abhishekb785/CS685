#input zero wala in the output
import json
import pprint
import collections
import operator
import pandas as pd
import csv
import math
from collections import defaultdict
import networkx as nx

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

sample = open('temp4','r')
lines = sample.readlines()

for line in lines:
    a = line[:-1]
    lis = a.split(";")
    f = aid[lis[0]]
    l = aid[lis[-1]]
    source = []
    destination = []

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

    for s in source:
        for d in destination:
            if s in dic2D.keys():
                if d in dic2D[s].keys():
                    dic2D[s][d] = dic2D[s][d] + 1
                else:
                    dic2D[s][d] = 1
            else:
                dic2D[s] = {d:1}

sample = open('temp6','r')
lines = sample.readlines()
for line in lines:
    a = line[:-1]
    lx = a.split(" ")
    lis = lx[0].split(";")
    f = aid[lis[0]]
    try:
        l = aid[lx[1]]
    except:
        continue
    # print(f,l)
    source = []
    destination = []

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

    for s in source:
        for d in destination:
            if s in dic3D.keys():
                if d in dic3D[s].keys():
                    dic3D[s][d] = dic3D[s][d] + 1
                else:
                    dic3D[s][d] = 1
            else:
                dic3D[s] = {d:1}

# print(dic3D)
sd = open('category-pairs.csv','w')
print('﻿From_Category,To_Category,Percentage_of_finished_paths,Percentage_of_unfinished_paths',file=sd)


# for i in sorted(dic2D):
#     for j in sorted(dic2D[i]):
#         try:
#             a = round((dic2D[i][j]*100)/(dic2D[i][j] + dic3D[i][j]),2)
#         except:
#             a = 100
#         print(i,j,a,round(100-a,2),sep=",",file =sd)


for i in range(1,147):
    key = "C" + "{:04n}".format(i)
    for j in range(1,147):
        key2 = "C" + "{:04n}".format(j)
        try:
            a = dic2D[key][key2]
        except:
            a = 0
        try:
            b = dic3D[key][key2]
        except:
            b = 0
        if a!=0 or b!=0:
            c = round((a*100/(a+b)),2)
            print(key,key2,c,round(100-c,2),sep=",",file=sd)
