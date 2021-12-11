import json
import pprint
import datetime
from collections import defaultdict
import fileinput
import collections
import operator
import csv
f = open('mapping.txt')
dic = eval(f.read())
store = {}
file2 = open('neighbor-district-temp.json', 'r')
c = open('neighbor-district-ids.txt','w')
data = json.load(file2)
stores = {}
for key in data:
      try:
          x = key.split('/')[0]
          x = x.strip()
          l = dic[x]
      except:
          try:
              x = key
              x = x.strip()
              l = dic[x]
          except:
              pass
      for j in data[key]:
            try:
                f = j.split('/')[0]
                f = f.strip()
                ll = dic[f]
            except:
                try:
                    f = j
                    f = f.strip()
                    ll = dic[f]
                except:
                    pass
            try:
              store[l].append(ll)
              k = l.split('(')[1]
              k = k[0:3]
              kk = ll.split('(')[1]
              kk = kk[0:3]
              kk = int(kk)
              stores[k].add(kk)
            except :
                try:
                    store[l] = [ll]
                    k = l.split('(')[1]
                    k = k[0:3]
                    kk = ll.split('(')[1]
                    kk = kk[0:3]
                    kk = int(kk)
                    stores[k] = {kk}
                except:
                    pass

print(stores,file=c)

app_json = json.dumps(store,sort_keys=True)
pprint.pprint(app_json)
# pprint.pprint(store)
