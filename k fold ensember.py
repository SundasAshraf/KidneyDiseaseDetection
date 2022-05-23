import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import transformation
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('dataset_kaggle_14_featurs.csv')
#print(dataset)
#dataset.head()
Positive= dataset[dataset['Class'] == '1'].shape[0]
Negative = dataset[dataset['Class'] == '0'].shape[0]
print(Positive)
print(Negative)
# bar plot of the 3 classes
plt.bar(10,Positive,3, label="Positve ")
plt.bar(15,Negative,3, label="Negative")
plt.legend()
plt.ylabel('Patient Count')
plt.title('Data Distribution')
plt.xlabel('Kidney Disease Yes/No')
plt.show()
#data is divided into features and labels x contains all features
#y contains all labels
from numpy import mean

## feature scaling

print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,13].values
print(X)
print(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X)
X=np.nan_to_num(X)

classifier = DecisionTreeClassifier()
print(' Decision tree  kaggle dataset')
#print(np.where(np.isnan(X_train)))
log_model=LogisticRegression()
dec_model = DecisionTreeClassifier()
cv = KFold(n_splits=5, random_state=1, shuffle=True)
scores = cross_val_score(log_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(dec_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
m1=mean(scores)
m2=mean(scores2)
accuracy=(m1+m2)/2
   # report performance
print('For K = 5 ensemble on kaggle dataset')
print('Accuracy: %.3f' % accuracy)
