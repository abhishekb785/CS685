import pandas as pd
crop_cost = pd.read_csv('Dataset/cult_cost.csv',delimiter = ',')
df = crop_cost.groupby(['Crop','Year'], as_index=False).agg({'CostperQuint':  'min','CostperHect':'min'})
# .agg(['min']),on=['Crop','Year'])
print(df)
rev = pd.read_csv('Dataset/revenue.csv',delimiter = ',')
rev = rev.rename(columns={"Crop_Year": "Year"})

print(rev)
# exit()
# comb = df.set_index('Crop',"Year").join(rev.set_index('Crop',"Year"))
comb =pd.merge(df,rev,on=['Crop',"Year"])
comb["Income per quintal"] = comb["Price per quintal"]-comb["CostperQuint"]
comb["Income per hectare"] = comb["Revenue per hectare"]*10 - comb["CostperHect"]
comb['New Income/Hect'] = comb.apply(lambda row: row["Production in quintals"]*row["Income per quintal"]/row["Area in Hectares"],axis=1)
comb['New Cost PerHectare'] = comb.apply(lambda row: row["Production in quintals"]*row["CostperQuint"]/row["Area in Hectares"]/10,axis=1)
print (comb)
comb.to_csv('Dataset/income.csv', index=False)