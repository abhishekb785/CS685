import json
import pprint
import collections
import operator
import pandas as pd
import csv
s = open('states.txt', 'w') #name of all states
d = open('districts.txt','w') # name of all districts
sd = open('states-districts.csv','w') # mapping of district id
ss = open('states-districts.txt','w') # mapping of state with district id json
trash = ["", "Other State", "Foreign Evacuees", "Italians", "Airport Quarantine", "Evacuees", "BSF Camp"]
for w in range(1,14):
    sample = 'raw_data' + str(w) + '.json'
    with open(sample) as f:
      data = json.load(f)
      data = data['raw_data']
      states = set()
      store = {}
      for key in data:
          states.add(key['detectedstate'])
          if key['detecteddistrict'] and key['detecteddistrict'] not in trash:
              try:
                  store[key['detectedstate']].add(key['detecteddistrict'])
              except KeyError:
                  store[key['detectedstate']] = {key['detecteddistrict']}

for i in states:
    print(i,file = s)
    #to create state.txt for all the states

for key in store:
    for i in store[key]:
        print(i,file=d)
# to create a map of all the districts


for key in store:
    for i in store[key]:
        print(key, end = ",", file = sd)
        print(i,file=sd)
sd.close()
# print(df)
sample = open('states-districts.csv','r')
df = csv.reader(sample,delimiter=',')
sortedlist = sorted(df, key=operator.itemgetter(1))
sd = open('district-id.csv','w')
t = 101
store = {}
print('district,state,districtid',file=sd) #to create district ids
for i in sortedlist:
    try:
        store[i[0]].add(t)
    except KeyError:
        store[i[0]] = {t}
    print(i[1], end = "," , file = sd)
    print(i[0], end="," , file=sd)
    print(t,file=sd)
    t+=1
# print(store)
print(store,file=ss)
