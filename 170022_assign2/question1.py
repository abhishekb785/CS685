import json
import pprint
import collections
import operator
import pandas as pd
import csv

sd = open('article-ids.csv','w')
print('﻿Article_Name,Article_ID',file=sd)

sample = open('temp2','r')
lines = sample.readlines()
i = 1
for line in lines:
    print(line[:-1] + ",A", end="", file=sd)
    print("{:04n}".format(i), file =sd)
    i = i+1
