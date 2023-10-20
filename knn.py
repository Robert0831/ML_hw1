import numpy as np
from collections import Counter
from data import dataset

class KNNeu:
    def __init__(self, k=21,file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"):
        self.k = k
        self.data=dataset(file_path)
        self.data.train_data=self.data.train_data[:int(self.data.row*0.8*0.2),:]
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def absolute_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) )
    def chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2) )    
    def predict_val(self):
        pred = [self._predict(x) for x in self.data.val_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.val_label[i]:
                acc+=1
        print(f"EUC Val Accuracy: {acc/int(self.data.row*0.1)}")

    def predict_test(self):
        pred = [self._predict(x) for x in self.data.test_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.test_label[i]:
                acc+=1
        print(f"EUC Test Accuracy: {acc/int(self.data.row*0.1)}")
        
    def _predict(self, x):
        distances = [self.euclidean_distance(x, train) for train in self.data.train_data]
        # distances = [self.absolute_distance(x, train) for train in self.data.train_data]
        # distances = [self.chebyshev_distance(x, train) for train in self.data.train_data]

        # 前k小
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.data.train_label[i] for i in k_indices]
        
        #算誰多
        common = Counter(k_nearest_labels)
        if common[1] >common[0]:
            return 1
        elif common[1] < common[0]:
            return 0

class KNNabs:
    def __init__(self, k=21,file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"):
        self.k = k
        self.data=dataset(file_path)
        self.data.train_data=self.data.train_data[:int(self.data.row*0.8*0.2),:]
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def absolute_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) )
    def chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2) )    
    def predict_val(self):
        pred = [self._predict(x) for x in self.data.val_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.val_label[i]:
                acc+=1
        print(f"ABS Val Accuracy: {acc/int(self.data.row*0.1)}")

    def predict_test(self):
        pred = [self._predict(x) for x in self.data.test_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.test_label[i]:
                acc+=1
        print(f"ABS Test Accuracy: {acc/int(self.data.row*0.1)}")
        
    def _predict(self, x):
        #distances = [self.euclidean_distance(x, train) for train in self.data.train_data]
        distances = [self.absolute_distance(x, train) for train in self.data.train_data]
        # distances = [self.chebyshev_distance(x, train) for train in self.data.train_data]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.data.train_label[i] for i in k_indices]
        
        common = Counter(k_nearest_labels)
        if common[1] >common[0]:
            return 1
        elif common[1] < common[0]:
            return 0
#切比雪夫
class KNNche:
    def __init__(self, k=21,file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"):
        self.k = k
        self.data=dataset(file_path)
        self.data.train_data=self.data.train_data[:int(self.data.row*0.8*0.2),:]
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def absolute_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) )
    def chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2) )    
    def predict_val(self):
        pred = [self._predict(x) for x in self.data.val_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.val_label[i]:
                acc+=1
        print(f"CHE Val Accuracy: {acc/int(self.data.row*0.1)}")

    def predict_test(self):
        pred = [self._predict(x) for x in self.data.test_data]
        acc=0
        for i in range(len(pred)):
            if pred[i]==self.data.test_label[i]:
                acc+=1
        print(f"CHE Test Accuracy: {acc/int(self.data.row*0.1)}")
        
    def _predict(self, x):
        #distances = [self.euclidean_distance(x, train) for train in self.data.train_data]
        #distances = [self.absolute_distance(x, train) for train in self.data.train_data]
        distances = [self.chebyshev_distance(x, train) for train in self.data.train_data]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.data.train_label[i] for i in k_indices]
        
        common = Counter(k_nearest_labels)
        if common[1] >common[0]:
            return 1
        elif common[1] < common[0]:
            return 0  


eu = KNNeu()
ab=KNNabs()
ch=KNNche()

# eu.predict_val()
# eu.predict_test()

# ab.predict_val()
# ab.predict_test()

ch.predict_val()
ch.predict_test()