
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