'''
Authors: Ashwani Kashyap, Anshul Pardhi
https://github.com/anshul1004/DecisionTree
'''

from tree_func import *
import pandas as pd
import random


# default data set
df = pd.read_csv("C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv")
df.drop(columns=['policy_id'], inplace=True)
df=df.sample(frac=1, random_state=101)
header = list(df.columns)
row=df.shape[0] #58592
train_index=int(row*0.8)
val_index=int(row*0.9)

train_data=df.iloc[0:int(train_index*0.02)].values.tolist()
val_data=df.iloc[train_index:val_index].values.tolist()
test_data=df.iloc[val_index:].values.tolist()




# building the tree
t = build_tree(train_data, header)

# get leaf and inner nodes
print("\nLeaf nodes ****************")
leaves = getLeafNodes(t)

# for leaf in leaves:
#     print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

# print("\nNon-leaf nodes ****************")
innerNodes = getInnerNodes(t)

# for inner in innerNodes:
#     print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# print tree
maxAccuracy = computeAccuracy(test_data, t)

logger1 = open('tree.txt', 'a')
logger1.write('Test Tree before pruning with accuracy %f\n'%(maxAccuracy*100))
logger1.close()
print("\n  Test Tree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
maxAccuracy = computeAccuracy(val_data, t)
logger1 = open('tree.txt', 'a')
logger1.write('Val Tree before pruning with accuracy %f\n'%(maxAccuracy*100))
logger1.close()

print("\n  Val Tree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")

#print_tree(t)

# TODO: You have to decide on a pruning strategy
# Pruning strategy
nodeIdToPrune = -1
count=0

random.seed(101)
random.shuffle(innerNodes)
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(val_data, t)
        print("Pruned node_id: " + str(node.id) + " to achieve accuracy: " + str(currentAccuracy*100) + "%")
        # print("Pruned Tree")
        # print_tree(t)
        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            nodeIdToPrune = node.id
        t = build_tree(train_data, header)
        if maxAccuracy == 1:
            break
    count=count+1
    if count >=20:
        break
if nodeIdToPrune != -1:
    t = build_tree(train_data, header)
    prune_tree(t, [nodeIdToPrune])
    logger1 = open('tree.txt', 'a')
    logger1.write('Val Tree after pruning with accuracy %f\n'%(maxAccuracy*100))
    logger1.close()
    print("\n  Val Tree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
else:
    t = build_tree(train_data, header)
    print("\nPruning strategy did'nt increased accuracy")

print("\n********************************************************************")
print("*********** Final Tree with accuracy: " + str(maxAccuracy*100) + "%  ************")
print("********************************************************************\n")
#print_tree(t)

#########################################################



currentAccuracy = computeAccuracy(test_data, t)
print("\n  Test Tree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
logger1 = open('tree.txt', 'a')
logger1.write('Test Tree after pruning with accuracy %f\n'%(maxAccuracy*100))
logger1.close()
