import numpy as np
from data import dataset
class Perceptron:
    def __init__(self, input_size=42, learning_rate=0.1,file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"):
        self.weights = np.random.random(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.data=dataset(file_path)

    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        return 1 if activation >=0 else 0

    def train(self):
        total_error = 0.0
        for index, (row,labels) in enumerate(zip(self.data.train_data,self.data.train_label)):
            prediction = self.predict(row)
            error = labels - prediction

            self.weights += self.learning_rate * error *2* row
            self.bias += self.learning_rate * error*2
            total_error += error ** 2


        mse = total_error / int(self.data.row*0.8)
        #print(f"Epoch {epoch+1}, Mean Square Error: {mse}")
        logger1 = open('linear_train.txt', 'a')
        logger1.write('%d %f\n'%(epoch+1,round(mse,4)))
        logger1.close()

################### val error
        total_error = 0.0
        for index, (row,labels) in enumerate(zip(self.data.val_data,self.data.val_label)):
            prediction = self.predict(row)
            error = labels - prediction
            total_error += error ** 2
            mse = total_error / int(self.data.row*0.1)
        logger1 = open('linear_val.txt', 'a')
        logger1.write('%d %f\n'%(epoch+1,round(mse,4)))
        logger1.close()

    def test_train(self):
        acc=0.0
        for index, (row,labels) in enumerate(zip(self.data.train_data,self.data.train_label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        print(f"Train Accuracy: {acc/ int(self.data.row*0.8)}")
    def test_val(self):
        acc=0.0
        for index, (row,labels) in enumerate(zip(self.data.val_data,self.data.val_label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        print(f"Val Accuracy: {acc/int(self.data.row*0.1)}")
    def test_test(self):
        acc=0.0
        for index, (row,labels) in enumerate(zip(self.data.test_data,self.data.test_label)):
            prediction = self.predict(row)
            if prediction==labels:
                acc+=1
        print(f"Test Accuracy: {acc/int(self.data.row*0.1)}")
file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
#a=dataset(file_path)
epochs=20
model=Perceptron(file_path)
for epoch in range(epochs):
    model.train()
model.test_train()
model.test_val()
model.test_test()


