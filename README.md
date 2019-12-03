# Breast Cancer Classification

## Introduction
BLAH BLAH




## Data Collection and Cleaning
blah blah


```python
#imports
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


#import the data.
data = pd.read_csv('data.csv')

#Show first 5 entries.
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_radius</th>
      <th>mean_texture</th>
      <th>mean_perimeter</th>
      <th>mean_area</th>
      <th>mean_smoothness</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



As stated above, the data consits of 5 features, and a binary choice of 0, or 1, for the label 'diagnosis'. In total we have 569 entries, each with 5 features and 1 label.

### Removing 'NaN' Entries

Some of the data could include entries with the data 'NaN' this is data that is missing, and needs to be removed.


```python
#Remove 'NaN'.
data.dropna(axis=0, inplace = True)
data.shape
```




    (569, 6)



After removing the entries with 'NaN' in the fields. The data is still left with 569 entires and of course still 5 features and 1 label. Therfore, no entries contaied 'NaN'.

### Sperating X and Y
In order to train our models, the data needs to be serperated into inputs 'x_values' and the coresponding outputs 'y_values'. To acheive this, the python library 'Panda' has supplied us with a function to extract all the features and labels, to create a NxM matrix of inputs and a Nx1 matrix of outputs. Where N is the number of training data, and M is the number of features. 


```python
#Seperating

#Get all, but the last columns, these are the input valuse.The last column, is the y_values.
x_values = data.iloc[:, :-1].values

#Only get the last column, these are the coresponding outputs.
y_values = data.iloc[:, -1].values

print 'x_values matrix dimensions (NxM): ', x_values.shape
print('')
print 'y_values matrix dimensions (Nx1): ', y_values.shape

```

    x_values matrix dimensions (NxM):  (569, 5)
    
    y_values matrix dimensions (Nx1):  (569,)


### Do Visual Stuff

### Seperate into training and testing fields


```python
# Test size will be a quater. Also, this means the traing size will be three quaters.

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.25, random_state = 0)

##add more here


```

## Algorithms Used in Analysis

## Aanalysis

### Logistic Regression


```python
logReg = LogisticRegression(solver='liblinear')
logReg.fit(x_train, y_train)

predictions = logReg.predict(x_test)
print(predictions)
print(y_test)
score = logReg.score(x_test, y_test)
print(score)



```

    [1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1 0 1 1 0 1 0 1 0 1 0 1 1 1
     0 1 0 0 1 0 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 0 0 0 1 1 0 1 1
     0 1 1 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 0 1
     0 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 0]
    [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 1 0 1 0 1
     0 1 0 0 1 0 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 0 0 0 1 1 0 1 0
     0 1 1 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 1 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 0 0 1
     0 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0]
    0.9230769230769231


### SVM


```python
svc = svm.SVC(gamma='scale')
svc.fit(x_train, y_train)

#svc.predict(x_test)
svc_score = svc.score(x_test, y_test)
print(svc_score)
```

    0.8811188811188811


### Naive Bayesian


```python
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_score = nb.score(x_test, y_test)
print(nb_score)
```

    0.9300699300699301


## Results
BLAH BLAH


## Conclusions

## References
