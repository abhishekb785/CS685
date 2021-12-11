import json
import pprint
import datetime
import pandas as pd
import statistics
from collections import defaultdict

districts_portal = []
f = open("districts.txt", "r")
dics = {}
for i in f.readlines():
    for value in i.strip().split('\n'):
        t = value.lower()
        districts_portal.append(t)
        dics[t] = value
# print(districts_portal)
district_portal = set(districts_portal)
# print(len(set(districts_portal)))
# exit()
districts_neighbor = []
f = open("dis.txt", "r")
for i in f.readlines():
    for value in i.strip().split('\n'):
        districts_neighbor.append(value)
# print(districts_neighbor)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
dic = {}
same = intersection(districts_portal, districts_neighbor) # 535

def Diff(list1, list2):
    return (list(list(set(list1)-set(list2))))

lis1 = Diff(districts_portal,same) #172
lis2 = Diff(districts_neighbor,same) #183

rep = ['aurangabad','balrampur','bilaspur','pratapgarh','hamirpur']

same = Diff(same,rep)
# print(same)
# exit()
for i in same:
    dic[i] = dics[i]

s1 = []
s2 = []
for i in lis2:
    for j in lis1:
        if(j.find(i) != -1 ):
            dic[i] = dics[j]
            s1.append(j)
            s2.append(i)
            break
        elif(i.find(j)!=-1):
            dic[i] = dics[j]
            s1.append(j)
            s2.append(i)
            break

# print(dic)
lis1 = Diff(lis1,s1)
lis2 = Diff(lis2,s2)

# print(len(lis2)) #630 till here

def editDistDP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j    # Min. operations = j
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]

# dic = {}
for i in lis1:
    for j in lis2:
        k = editDistDP(i,j,len(i),len(j))
        if k<=3 and len(i) > 5 and len(j) > 5 :
            dic[j] = dics[i]

# ('aurangabad', 'BR') :"  ",
# ('aurangabad', 'MH') :"  ",
# ('balrampur', 'CT') :"  ",
# ('balrampur', 'UP') :"  ",
# ('bilaspur', 'CT') :"  ",
# ('bilaspur', 'HP') :"  ",
# ('hamirpur', 'HP') :"  ",
# ('hamirpur', 'UP') :"  ",
# ('pratapgarh', 'RJ') :"  ",
# ('pratapgarh', 'UP') :"  ",
dic['jyotiba phule nagar/Q1891677'] = 'Amroha'
dic['aurangabad/Q43086'] = 'Aurangabad(133)'
dic['aurangabad/Q592942'] = 'Aurangabad(134)'
dic['faizabad/Q1814132'] = 'Ayodhya'
dic['baleshwar/Q2022279'] = 'Balasore'
dic['balrampur/Q16056268'] = 'Balrampur(152)'
dic['balrampur/Q1948380'] = 'Balrampur(151)'
dic['bid/Q814037'] = 'Beed'
dic['belgaum/Q815464'] = 'Belagavi'
dic['sant ravidas nagar/Q127533'] = 'Bhadohi'
dic['bilaspur/Q100157'] = 'Bilaspur(197)'
dic['bilaspur/Q1478939'] = 'Bilaspur(196)'
dic['baudh/Q2363639'] = 'Boudh'
dic['kochbihar/Q2728658'] = 'Cooch Behar'
dic['sonapur/Q1473957'] = 'Subarnapur'
dic['pashchim champaran/Q100124'] = 'West Champaran'
dic['sri potti sriramulu nellore/Q15383'] = 'S.P.S. Nellore'
dic['sahibzada ajit singh nagar/Q2037672'] = 'S.A.S. Nagar'
dic['pratapgarh/Q1585433'] = 'Pratapgarh(615)'
dic['pratapgarh/Q1473962'] = 'Pratapgarh(616)'
dic['palghat/Q1535742'] = 'Palakkad'
dic['pakaur/Q2295930'] = 'Pakur'
dic['hugli/Q548518'] = 'Hooghly'
dic['hamirpur/Q2086180'] = 'Hamirpur(342)'
dic['hamirpur/Q2019757'] = 'Hamirpur(343)'
dic['purbi singhbhum/Q2452921'] = 'East Singhbhum'
dic['purba champaran/Q49159'] = 'East Champaran'
dic['bijapur/Q1727570'] = 'Vijayapura'
dic['pashchimi singhbhum/Q1950527'] = 'West Singhbhum'
# print(len(dic))
# pprint.pprint(dic)
# exit()
rep2= ['Aurangabad(133)','Aurangabad(134)','Balrampur(152)','Balrampur(151)','Bilaspur(197)','Bilaspur(196)','Pratapgarh(615)','Pratapgarh(616)','Hamirpur(342)','Hamirpur(343)']
df = pd.read_csv('district-id.csv')
for i in dic:
    if dic[i] not in rep2:
        yy = df[(df['district'] == dic[i])]
        try:
            xx = int(yy['districtid'])
        except:
            pass
        dic[i] = dic[i] + "(" + str(xx) + ")"
pprint.pprint(dic)
