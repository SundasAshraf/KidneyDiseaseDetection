import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,24].values
Positive=250
Negative=250
plt.bar(5,Positive,3, label="Positve ")
plt.bar(15,Negative,3, label="Negative")
plt.legend()
plt.ylabel('Patient Count')
plt.title('Data Distribution')
plt.xlabel('Kidney Disease Yes/No')
plt.show()