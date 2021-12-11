import json
import pprint
import collections
import operator
import pandas as pd
import csv

sample = open('temp2','r')
lines = sample.readlines()

sd = open('category-ids.csv','w')
print('﻿Category_Name,Category_ID',file=sd)
graph = {}

for line in lines:
    a = line[:-1]
    n = a.count('.') +1
    # print(n)
    for i in range(1,n):
        k = ".".join(a.split(".", i)[:i])
        val = ".".join(a.split(".", i+1)[:i+1])
        if k in graph.keys():
            if val not in graph[k]:
                graph[k].append(val)
        else:
            graph[k] = [val]


visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue

for key in graph:
    graph[key].sort()

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)
  i = 1
  while queue:
    s = queue.pop(0)
    print(s + ",C", end="", file=sd)
    print("{:04n}".format(i), file =sd)
    i = i+1

    if s in graph.keys():
        for neighbour in graph[s]:
          if neighbour not in visited:
            visited.append(neighbour)
            queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'subject')
