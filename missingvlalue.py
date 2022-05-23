import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('kidney_disease without preprocess.csv')
#print(dataset)
#dataset.head()
print(dataset.isnull().sum())
