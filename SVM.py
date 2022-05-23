import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = svm.SVC()
X_train=np.nan_to_num(X_train)
X_test=np.nan_to_num(X_test)
#print(np.where(np.isnan(X_train)))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('classificaion report')
print(classification_report(y_test, y_pred))
print('accuracy for SVM model:',metrics.accuracy_score(y_test,y_pred))