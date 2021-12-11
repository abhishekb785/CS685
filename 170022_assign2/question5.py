import json
import pprint
import collections
import operator
import pandas as pd
import csv
import networkx as nx

G = nx.DiGraph()
csv_file = open('edges.csv')
csv_reader = csv.reader(csv_file,delimiter = ',')

for row in csv_reader:
    if row[0] == "﻿From_ArticleID":
        continue
    x = (row[0],row[1])
    # print(x)
    G.add_edge(*x)
# print(G)
# exit()
sd = open('graph-components.csv','w')
print('﻿Nodes,Edges,Diameter',file=sd)

for c in nx.strongly_connected_components(G):
    temp = G.subgraph(c)
    # print(len(temp),temp.number_of_edges(),nx.diameter(temp))
    print(len(temp),temp.number_of_edges(),nx.diameter(temp),sep=",",file=sd)


