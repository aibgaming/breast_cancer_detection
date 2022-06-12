from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data['data']
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)

print(clf.score(X_test,Y_test))

column_data = np.concatenate([data['data'], data['target'][:, None]], axis=1)
column_names = np.concatenate([data['feature_names'], ["class"] ])
df =  pd.DataFrame(column_data,columns = column_names)

sns.heatmap(df.corr(), cmap = "coolwarm", annot= True, annot_kws = {"fontsize":8})
#plt.tight_layout()
plt.show()