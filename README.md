# breast_cancer_detection
Uses the python sklearn library to develop a correlation heat_map for prediction of breast cancer depending on various feautures

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
data = load_breast_cancer()
X = data['data']
Y = data['target']
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
```


```python
clf = KNeighborsClassifier()
clf.fit(X_train, Y_train)
```



```python
print(clf.score(X_test,Y_test))
```

    0.9035087719298246
    


```python
column_data = np.concatenate([data['data'], data['target'][:, None]], axis=1)
column_names = np.concatenate([data['feature_names'], ["class"] ])
```


```python
df =  pd.DataFrame(column_data,columns = column_names)
```


```python
sns.heatmap(df.corr(), cmap = "coolwarm", annot= True, annot_kws = {"fontsize":8})
```




    <AxesSubplot:>




    
![png](output_8_1.png)
    



```python

```
