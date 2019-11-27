# Breast Cancer Classification

## Background
BLAH BLAH




## Methods and Findings
blah blah


```python
#imports
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression


#import the data.
data = pd.read_csv('data.csv')

#preview the first 5 lines of the data.
data.shape
```




    (569, 6)



Notice we get 4238 entreis. Now we need to eliminate no good data.

### Eliminating no good data

Remove 'NaN' from desc and genre.


```python
#Remove 'NaN'.
data.dropna(axis=0, inplace = True)
data.shape
```




    (569, 6)



Now giving us 3656 entries.

### Sperating X and Y


```python
#Seperating

#need the -1 because we dont want the last value, that is our y_values.
x_values = data.iloc[:, :-1].values
y_values = data.iloc[:, -1].values

print(x_values.shape)
print(y_values.shape)

```

    (569, 5)
    (569,)


### Do Visual Stuff

### Seperate into training and testing fields


```python
# Test size will be a quater. Also, this means the traing size will be three quaters.

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size = 0.25, random_state = 0)

##add more here


```

### Now we do Log Regression and Evaluate.


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


## Conclusions
BLAH BLAH

