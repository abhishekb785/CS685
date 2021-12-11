# df.sort_values('some_column', ascending=False)[:10]
import json
import pprint
import datetime
import pandas as pd
import statistics
from collections import defaultdict
df = pd.read_csv('zscore-week.csv')
cw = open('top-week.csv', 'w')
print('timeid,method,spot,districtid1,districtid2,districtid3,disxtrictid4,districtid5' ,file = cw)

for i in range(1,25):
    # i = str(i)
    t = df[(df['timeid'] == i)]
    k = t.sort_values('neighborhoodzscore',ascending=False)[:6]
    j = t.sort_values('statezscore',ascending=False)[:6]
    print(i, end = ",", file = cw)
    print('neighborhood', end = ",",file=cw)
    print('hot', end = '',file=cw)
    for ind in k.index:
        print(',',end = '',file=cw)
        print(k['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    print(i, end = ",", file = cw)
    print('state', end = ",",file=cw)
    print('hot', end = '',file=cw)
    for ind in j.index:
        print(',',end = '',file=cw)
        print(j['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    k = t.sort_values('neighborhoodzscore',ascending=True)[:5]
    j = t.sort_values('statezscore',ascending=True)[:5]
    print(i, end = ",", file = cw)
    print('neighborhood', end = ",",file=cw)
    print('cold', end = '',file=cw)
    for ind in k.index:
        print(',',end = '',file=cw)
        print(k['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    print(i, end = ",", file = cw)
    print('state', end = ",",file=cw)
    print('cold', end = '',file=cw)
    for ind in j.index:
        print(',',end = '',file=cw)
        print(j['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)
cw.close()
# exit()

df = pd.read_csv('zscore-month.csv')
cw = open('top-month.csv', 'w')
print('timeid,method,spot,districtid1,districtid2,districtid3,disxtrictid4,districtid5' ,file = cw)
for i in range(1,8):
    # i = str(i)
    t = df[(df['timeid'] == i)]
    k = t.sort_values('neighborhoodzscore',ascending=False)[:5]
    j = t.sort_values('statezscore',ascending=False)[:5]
    print(i, end = ",", file = cw)
    print('neighborhood', end = ",",file=cw)
    print('hot', end = '',file=cw)
    for ind in k.index:
        print(',',end = '',file=cw)
        print(k['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    print(i, end = ",", file = cw)
    print('state', end = ",",file=cw)
    print('hot', end = '',file=cw)
    for ind in j.index:
        print(',',end = '',file=cw)
        print(j['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    k = t.sort_values('neighborhoodzscore',ascending=True)[:5]
    j = t.sort_values('statezscore',ascending=True)[:5]
    print(i, end = ",", file = cw)
    print('neighborhood', end = ",",file=cw)
    print('cold', end = '',file=cw)
    for ind in k.index:
        print(',',end = '',file=cw)
        print(k['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)

    print(i, end = ",", file = cw)
    print('state', end = ",",file=cw)
    print('cold', end = '',file=cw)
    for ind in j.index:
        print(',',end = '',file=cw)
        print(j['districtid'][ind],end = "",file=cw)
    print('\r',file=cw)
cw.close()
# exit()

df = pd.read_csv('zscore-overall.csv')
cw = open('top-overall.csv', 'w')
print('timeid,method,spot,districtid1,districtid2,districtid3,disxtrictid4,districtid5' ,file = cw)
i=1
t = df[(df['timeid'] ==1)]
k = t.sort_values('neighborhoodzscore',ascending=False)[:5]
j = t.sort_values('statezscore',ascending=False)[:5]
print(i, end = ",", file = cw)
print('neighborhood', end = ",",file=cw)
print('hot', end = '',file=cw)
for ind in k.index:
    print(',',end = '',file=cw)
    print(k['districtid'][ind],end = "",file=cw)
print('\r',file=cw)

print(i, end = ",", file = cw)
print('state', end = ",",file=cw)
print('hot', end = '',file=cw)
for ind in j.index:
    print(',',end = '',file=cw)
    print(j['districtid'][ind],end = "",file=cw)
print('\r',file=cw)

k = t.sort_values('neighborhoodzscore',ascending=True)[:5]
j = t.sort_values('statezscore',ascending=True)[:5]
print(i, end = ",", file = cw)
print('neighborhood', end = ",",file=cw)
print('cold', end = '',file=cw)
for ind in k.index:
    print(',',end = '',file=cw)
    print(k['districtid'][ind],end = "",file=cw)
print('\r',file=cw)

print(i, end = ",", file = cw)
print('state', end = ",",file=cw)
print('cold', end = '',file=cw)
for ind in j.index:
    print(',',end = '',file=cw)
    print(j['districtid'][ind],end = "",file=cw)
print('\r',file=cw)
cw.close()
