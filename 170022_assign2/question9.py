#1 wala issue
import json
import pprint
import collections
import operator
import pandas as pd
import csv
import math
from collections import defaultdict
import networkx as nx

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

G = nx.DiGraph()
csv_file = open('edges.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')

for row in csv_reader:
    if row[0] == "﻿From_ArticleID":
        continue
    x = (row[0],row[1])
    G.add_edge(*x)

sample = open('temp4','r')
lines = sample.readlines()

catnp = {}
catnt = {}
catsp = {}
catst = {}
catnp = defaultdict(lambda:0,catnp)
catnt = defaultdict(lambda:0,catnt)
catsp = defaultdict(lambda:0,catsp)
catst = defaultdict(lambda:0,catst)

for line in lines:
    done = []
    done2 = []
    a = line[:-1]
    lis = a.split(";")
    stack = []
    for key in lis:
        if key == '<' :
            stack.pop()
        else:
            stack.append(aid[key])
    f = stack[0]
    l = stack[-1]
    if f != l:
        try:
            short = nx.shortest_path(G,f,l)
        except:
            continue

        for key in stack:
            if key in acid:
                for cats in acid[key]:
                    a = cid2[cats]
                    n = a.count('.') +1
                    for i in range(1,n):
                        k = ".".join(a.split(".", i)[:i])
                        cat = cid[k]
                        if cat not in done:
                            catnp[cat] = catnp[cat] + 1
                            done.append(cat)
                        catnt[cat] = catnt[cat] + 1
                    if cats not in done:
                        catnp[cats] = catnp[cats] + 1
                        done.append(cats)
                    catnt[cats] = catnt[cats] + 1

        for key in short:
            if key in acid:
                for cats in acid[key]:
                    a = cid2[cats]
                    n = a.count('.') +1
                    for i in range(1,n):
                        k = ".".join(a.split(".", i)[:i])
                        cat = cid[k]
                        if cat not in done2:
                            catsp[cat] = catsp[cat] + 1
                            done2.append(cat)
                        catst[cat] = catst[cat] + 1
                    if cats not in done2:
                        catsp[cats] = catsp[cats] + 1
                        done2.append(cats)
                    catst[cats] = catst[cats] + 1

sd = open('category-subtree-paths.csv','w')
print('﻿Category_ID,Number_of_human_paths_traversed,Number_of_human_times_traversed,Number_of_shortest_paths_traversed,Number_of_shortest_times_traversed',file=sd)


for i in range(1,147):
    key = "C" + "{:04n}".format(i)
    if key in catsp.keys():
        print(key,catnp[key],catnt[key],catsp[key],catst[key],sep=",",file =sd)
    else:
        print(key,0,0,0,0,sep=",",file =sd)
