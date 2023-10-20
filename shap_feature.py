import shap
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from data_shap import dataset

file_path="C:\\Users\\acvlab\\Desktop\\ML\\homework1\\train.csv"
data=dataset(file_path)


classifier = SVC(kernel='linear', C=1) 
classifier.fit(data.train_data, data.train_label)

#取100個出來算
explainer = shap.KernelExplainer(classifier.predict, data.train_data[:100])

shap_values = explainer.shap_values(data.train_data[:100])

# Calculate feature importances
feature_importances = np.abs(shap_values).mean(axis=0)
sort=np.argsort(feature_importances)[::-1]

choose_frature=[]
for i in range(len(sort)):
    choose_frature.append(data.header[sort[i]])
print('Linear:')
print(choose_frature)


############################ Tree

model = DecisionTreeClassifier()
model.fit(data.train_data, data.train_label)

explainer = shap.KernelExplainer(model.predict, data.train_data[:100])

shap_values = explainer.shap_values(data.train_data[:100])

feature_importances = np.abs(shap_values).mean(axis=0)
sort=np.argsort(feature_importances)[::-1]
choose_frature=[]
for i in range(len(sort)):
    choose_frature.append(data.header[sort[i]])
print('Tree:')
print(choose_frature)

