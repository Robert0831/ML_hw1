import numpy as np
from k_data import dataset
class Perceptron:
    def __init__(self, input_size=42, learning_rate=0.1):
        self.weights = np.random.random(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        return 1 if activation >=0 else 0

    def train(self,data,label):
        total_error = 0.0
        for index, (row,labels) in enumerate(zip(data,label)):
            prediction = self.predict(row)
            error = labels - prediction

            self.weights += self.learning_rate * error *2* row
            self.bias += self.learning_rate * error*2
            total_error += error ** 2




    def test(self,data,label):
        acc=0.0
        for index, (row,labels) in enumerate(zip(data,label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        print(f"Val Accuracy: {acc/data.shape[0]}")

    def test_com(self,weight,data,label):
        acc=0.0
        self.weights=weight
        for index, (row,labels) in enumerate(zip(data,label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        print(f"Test Accuracy: {acc/data.shape[0]}")

file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
data=dataset(file_path)
traindata=data.train_data
trainlabel=data.train_label
testdata=data.test_data
testlabel=data.test_label

tol=traindata.shape[0]
epoch=3
K=10
WEIGHT=[]
for i in range(K):
    start = int(tol * (i / K))
    end = int(tol * ((i + 1) / K))

    vald=traindata[start:end,:]
    vall=trainlabel[start:end]
    trand=np.delete(traindata, range(start,end), axis=0)
    tranl=np.delete(traindata, range(start,end))

    model=Perceptron()
    for ep in range(epoch): 
        model.train(trand,tranl)
    model.test(vald,vall)
    WEIGHT.append(model.weights)

WEIGHT=np.array(WEIGHT)
WEIGHT=np.sum(WEIGHT,axis=0)/K
model.test_com(weight=WEIGHT,data=testdata,label=testlabel)

