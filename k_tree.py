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
train_index=int(row*0.9)

#train_data=df.iloc[0:int(train_index*0.02)].values.tolist()
train_data=df.iloc[0:int(train_index*0.02)]
test_data=df.iloc[train_index:train_index+int(row*0.1*0.02)].values.tolist()


K=10
tol=int(train_index*0.02)
result=[]
for i in range(K):
    start = int(tol * (i / K))
    end = int(tol * ((i + 1) / K))

    val=train_data.iloc[start:end].values.tolist()
    train=train_data.drop(train_data.index[start:end]).values.tolist()
    
    t = build_tree(train, header)
    maxAccuracy = computeAccuracy(val, t)
    print("Val Tree accuracy: " + str(maxAccuracy*100))

    res = predict(test_data, t)
    result.append(res)

ans=np.array(result)
ans=np.reshape(result,(K,-1))

count_zeros = np.sum(ans == 0, axis=0)
count_ones = np.sum(ans== 1, axis=0)
result=[]
for col in range(ans.shape[1]):
    if count_zeros[col] > count_ones[col]:
        result.append(0)
    elif count_zeros[col] < count_ones[col]:
        result.append(1)
    else:
       result.append(0)

acc=0
label=np.array(test_data)[:,-1].astype(int)
for i in range(len(result)):
    if result[i]==label[i]:
        acc+=1

print(f"Test Accuracy: {acc/len(result)}")