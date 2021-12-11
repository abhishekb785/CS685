import json
import pprint
import collections
import operator
import pandas as pd
import csv
import math
from collections import defaultdict
back = {}
back = defaultdict(lambda:0,back)
no_back = {}
no_back = defaultdict(lambda:0,no_back)
csv_file = open('article-ids.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')
aid = {}

for row in csv_reader:
    if row[0] == "﻿Article_Name":
        continue
    aid[row[0]] = row[1]

sample = open('temp8','r')
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

sd = open('finished-paths-back.csv','w')
print('﻿Human_Path_Length,Shortest_Path_Length,Ratio',file=sd)

sd2 = open('finished-paths-no-back.csv','w')
print('﻿Human_Path_Length,Shortest_Path_Length,Ratio',file=sd2)

# print(shortest[11][1])
sample = open('temp4','r')
lines = sample.readlines()
p = 0
for line in lines:
    a = line[:-1]
    n = a.count('<')
    lis = a.split(";")
    f = int(aid[lis[0]][1:])
    l = int(aid[lis[-1]][1:])
    h1 = len(lis)-1
    h2 = h1 - 2*n
    if shortest[f][l] != '_' and shortest[f][l]!= '0':
        s1 =int(shortest[f][l])
        r1 = round(h1/s1,2)
        r2 = round(h2/s1,2)
        diff1 = h1-s1
        diff2 = h2 -s1
        if diff1 > 10:
            diff1 = 11
        if diff2 > 10:
            diff2 = 11
        back[diff1] = back[diff1] + 1
        no_back[diff2] = no_back[diff2] + 1
        print(h1,end=",",file = sd)
        print(s1,end=",",file =sd)
        print(r1, file =sd)
        print(h2,end=",",file = sd2)
        print(s1,end=",",file =sd2)
        print(r2, file =sd2)
        p = p+1

sd3 = open('percentage-paths-back.csv','w')
print('Equal_Length,Larger_by_1,Larger_by_2,Larger_by_3,Larger_by_4,Larger_by_5,Larger_by_6,Larger_by_7,Larger_by_8,Larger_by_9,Larger_by_10,Larger_by_more_than_10',file=sd3)

sd4 = open('percentage-paths-no-back.csv','w')
print('﻿Equal_Length,Larger_by_1,Larger_by_2,Larger_by_3,Larger_by_4,Larger_by_5,Larger_by_6,Larger_by_7,Larger_by_8,Larger_by_9,Larger_by_10,Larger_by_more_than_10',file=sd4)

for key in sorted(back):
    r = round(back[key]*100/p,2)
    if key!= 11:
        print(r,end =",",file=sd3)
    else:
        print(r,file =sd3)

for key in sorted(no_back):
    r = round(no_back[key]*100/p,2)
    if key!= 11:
        print(r,end =",",file=sd4)
    else:
        print(r,file =sd4)
