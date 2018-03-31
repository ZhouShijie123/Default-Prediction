import pandas as pd
import numpy as np

data = pd.read_csv('NonBank.csv',encoding='cp1252')
rup = pd.read_csv('bankruptcy.csv',encoding='cp1252')
common = []
for i in data.columns:
    if i in rup.columns:
        common.append(i)
all_data = data[common]
ruptcy = rup[common]
data2 = all_data.append(ruptcy)
data2.index = range(len(data2))

for i in data2.columns:
    temp = []
    for j in range(len(data2)):
        if data2.loc[j,i] == '-':
            temp.append(np.nan)
        else:
            temp.append(data2[i][j])
    data2[i] = temp
    
    
for i in data2.columns[6:-1]:
    temp = []
    for j in range(len(data2)):
        if type(data2[i][j]) == type('-'):
            try:
                te = data2[i][j].replace('(','')
                te = te.replace(')','')
                te = te.replace(',','')
                temp.append(float(te))
            except:
                temp.append(np.nan)
                
        else:
            temp.append(data2[i][j])
    data2[i] = temp
    
for i in data2.columns[6:-1]:
    data2[i] = data2[i].astype(float)
data3 = data2.drop(data2.columns[0:1].tolist() + data2.columns[2:5].tolist(),axis = 1)
count = []
for i in data3.columns:
    cnt = 0
    for j in range(len(data3)):
        if type(data3[i][j]) == np.float64 and not data3[i][j] >= 0 and not data3[i][j] <= 0:
            cnt += 1
    if (cnt/len(data3) < 0.77 and cnt/len(data3) > 0.2):
        count.append(i)
data4 = data3.dropna(subset=count)

count2 = []
data4.index = range(len(data4))
drop_index = []
for i in data4.columns:
    cnt = 0
    for j in range(len(data4)):
        if type(data4[i][j]) == np.float64 and not data4[i][j] >= 0 and not data4[i][j] <= 0:
            cnt += 1
    count2.append(cnt/len(data4))
    if (cnt/len(data4)) > 0.5:
        drop_index.append(i)
data5 = data4.drop(drop_index,axis=1)

for i in data5.columns[2:-1]:
    temp = []
    for j in range(len(data5)):
        if not data3[i][j] >= 0 and not data3[i][j] <= 0:
            temp.append(np.mean(data5[i]))
        else:
            temp.append(data3[i][j])
    data5[i] = temp
data5.to_csv("data123.csv")
for i in data5.columns[2:-1]:
    data5[i] = (data5[i] - np.mean(data5[i]))/np.std(data5[i])

for i in data5.columns[0:2]:
    index = list(set(data5[i]))
    temp = []
    for j in range(len(data5)):
        temp.append(index.index(data5[i][j]))
    data5[i] = temp
data5.to_csv("data.csv")
        
        
        
        
        
        