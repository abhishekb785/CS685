import json
sample = open('edge-graph.csv', 'w')
with open('neighbor-districts-modified.json') as f:
  data = json.load(f,strict=False)
  data_items = data.items()
  for key,value in sorted(data_items):
        for i in value:
            # if i < key:
                print(key,i,sep=',',file = sample)
sample.close()
