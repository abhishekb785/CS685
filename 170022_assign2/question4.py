import json
import pprint
import collections
import operator
import pandas as pd
import csv

sample = open('temp8','r')
lines = sample.readlines()
i = 1

sd = open('edges.csv','w')
print('From_ArticleID,To_ArticleID',file=sd)

for line in lines:
    lis = [pos for pos, char in enumerate(line) if char == '1']
    for j in lis:
        print("A", end="", file=sd)
        print("{:04n}".format(i),end="", file =sd)
        print(",A", end="", file=sd)
        print("{:04n}".format(j+1), file =sd)
    i = i+1
    # print(lis)
    # exit()
