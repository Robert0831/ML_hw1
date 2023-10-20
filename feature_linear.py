import numpy as np
from data import dataset

class Perceptron:
    def __init__(self, input_size=1, learning_rate=0.1):
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

    def test_val(self,data,label):
        acc=0.0
        for index, (row,labels) in enumerate(zip(data,label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        #print(f"Val Accuracy: {acc/int(data.shape[0])}")
        return acc/int(data.shape[0])

file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
data=dataset(file_path)
train_data=data.train_data
train_label=data.train_label
val_data=data.val_data
val_label=data.val_label

choose=[]
col=train_data.shape[1]
epoch=5
for feature in range(col):
    model=Perceptron()
    for i in range(epoch):
        model.train(train_data[feature],train_label)
    temp=model.test_val(val_data[feature],val_label)
    choose.append(temp)

sort=np.argsort(choose)[::-1]

choose_frature=[]
for i in range(col):
    choose_frature.append(data.header[sort[i]])
print(choose_frature)
