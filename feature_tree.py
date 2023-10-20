'''
Authors: Ashwani Kashyap, Anshul Pardhi
https://github.com/anshul1004/DecisionTree
'''

from tree_func import *
import pandas as pd
import numpy as np

# default data set
df = pd.read_csv("C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv")
df.drop(columns=['policy_id'], inplace=True)
df=df.sample(frac=1, random_state=101)
header = list(df.columns)

row=df.shape[0] #58592
train_index=int(row*0.8)

#train_data=df.iloc[0:int(train_index*0.02)].values.tolist()
train_data=df.iloc[0:int(train_index*0.02)]
val_data=df.iloc[train_index:train_index+int(row*0.1*0.02)]

choose=[]
for feature in range(len(header)-1):
    train=train_data.iloc[:, [feature, -1]].values.tolist()
    val=val_data.iloc[:, [feature, -1]].values.tolist()

    t = build_tree(train, header)
    maxAccuracy = computeAccuracy(val, t)
    choose.append(maxAccuracy)

sort=np.argsort(choose)[::-1]

choose_frature=[]
for i in range(len(sort)):
    choose_frature.append(header[sort[i]])
print(choose_frature)
