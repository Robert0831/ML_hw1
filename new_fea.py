import numpy as np
from data import dataset

class Perceptron:
    def __init__(self, input_size=43, learning_rate=0.1):
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
        print(f"Test Accuracy: {acc/int(data.shape[0])}")



file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
data=dataset(file_path)
train_data=data.train_data
train_label=data.train_label
new1=train_data[:,0]
new2=train_data[:,1]
new=new1*new2
new=np.reshape(new,(-1,1))
train_data=np.concatenate((train_data,new),axis=1)

test_data=data.test_data
new1=test_data[:,0]
new2=test_data[:,1]
new=new1*new2
new=np.reshape(new,(-1,1))
test_data=np.concatenate((test_data,new),axis=1)
test_label=data.test_label

model=Perceptron()
epoch=100
for i in range(epoch):
    model.train(train_data,train_label)

model.test_val(test_data,test_label)