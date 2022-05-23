import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = pd.read_csv('dataset.csv')
#print(dataset)
#dataset.head()
Positive= dataset[dataset['classification'] == 'ckd'].shape[0]
Negative = dataset[dataset['classification'] == 'notckd'].shape[0]
#print(Positive)
#print(Negative)
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

## feature scaling

df = pd.DataFrame(dataset, columns = ['rbc',	'pc',	'pcc','ba',
    'htn',	'dm',	'cad',	'appet',	'pe',	'ane','classification'])
#print(df)




dataset['rbc'] = df['rbc'].apply(transformation.normal_to_numeric)
dataset['pc'] = df['pc'].apply(transformation.normal_to_numeric)
dataset['pcc'] = df['pcc'].apply(transformation.present_to_numeric)
dataset['ba'] = df['ba'].apply(transformation.present_to_numeric)
dataset['htn'] = df['htn'].apply(transformation.yes_to_numeric)
dataset['dm'] = df['dm'].apply(transformation.yes_to_numeric)
dataset['cad'] = df['cad'].apply(transformation.yes_to_numeric)
dataset['appet'] = df['appet'].apply(transformation.good_to_numeric)
dataset['pe'] = df['pe'].apply(transformation.yes_to_numeric)
dataset['ane'] = df['ane'].apply(transformation.yes_to_numeric)
dataset['classification'] = df['classification'].apply(transformation.labels_to_numeric)

print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,24].values
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(len(X_train))
print(len(X_test))
#feature scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)
X=np.nan_to_num(X)
X_train=np.nan_to_num(X_train)
X_test=np.nan_to_num(X_test)
classifier = KNeighborsClassifier(n_neighbors=7)
print(' k neighboours 7')
#print(np.where(np.isnan(X_train)))
k=[3,7,10]
for i in range(0,3):
   cv = KFold(n_splits=k[i], random_state=1, shuffle=True)
   scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
   # report performance
   print('For K =',k[i])
   print('Accuracy: %.3f' % (mean(scores) ))
