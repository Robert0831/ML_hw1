import numpy as np
from collections import Counter
from data import dataset

class KNNeu:
    def __init__(self, k=21):
        self.k = k
        self.combine=[]
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def predict_val(self,val,vall,train,trainlab):
        pred = [self._predict(x,train,trainlab) for x in val]
        acc=0
        for i in range(len(pred)):
            if pred[i]==vall[i]:
                acc+=1
        print(f"EUC Val Accuracy: {acc/len(pred)}")

    def predict_test(self,test,train,trainlab):
        pred = [self._predict(x,train,trainlab) for x in test]
        self.combine.append(pred)
        
    def _predict(self, x,train,trainlab):
        distances = [self.euclidean_distance(x, train) for train in train]

        # 前k小
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [trainlab[i] for i in k_indices]
        
        #算誰多
        common = Counter(k_nearest_labels)
        if common[1] >common[0]:
            return 1
        elif common[1] < common[0]:
            return 0



file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
data=dataset(file_path)
traindata=data.train_data
trainlabel=data.train_label
testdata=data.test_data
testlabel=data.test_label
tol=traindata.shape[0]
traindata=traindata[:int(tol*0.02),:]
trainlabel=trainlabel[:int(tol*0.02)]
tol=traindata.shape[0]

testdata=testdata[:int(testdata.shape[0]*0.02),:]
testlabel=testlabel[:int(testlabel.shape[0]*0.02)]


K=10
model=KNNeu()
for i in range(K):
    start = int(tol * (i / K))
    end = int(tol * ((i + 1) / K))

    vald=traindata[start:end,:]
    vall=trainlabel[start:end]
    trand=np.delete(traindata, range(start,end), axis=0)
    tranl=np.delete(traindata, range(start,end))


    model.predict_val(vald,vall,trand,tranl)
    model.predict_test(testdata,trand,tranl)
ans=np.array(model.combine)
ans=np.reshape(ans,(K,-1))
count_zeros = np.sum(ans == 0, axis=0)
count_ones = np.sum(ans == 1, axis=0)
result=[]
for col in range(ans.shape[1]):
    if count_zeros[col] > count_ones[col]:
        result.append(0)
    elif count_zeros[col] < count_ones[col]:
        result.append(1)
    else:
       result.append(0)
acc=0
for i in range(len(result)):
    if result[i]==testlabel[i]:
        acc+=1

print(f"Test Accuracy: {acc/len(result)}")