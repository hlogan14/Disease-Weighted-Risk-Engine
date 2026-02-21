# Feature Engineering for Categorical Health Data

## Overview
This script provides comprehensive feature engineering techniques and encoding methods for categorical health data points, which are commonly encountered in health-related datasets. The following methods will be covered: 

1. **One-Hot Encoding**  
2. **Label Encoding**  
3. **Frequency Encoding**  
4. **Target Encoding**  
5. **Binary Encoding**  

## Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
```

## Methods

### 1. One-Hot Encoding
```python
def one_hot_encode(df, column):
    return pd.get_dummies(df, columns=[column], drop_first=True)
```

### 2. Label Encoding
```python
def label_encode(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df
```

### 3. Frequency Encoding
```python
def frequency_encode(df, column):
    freq = df[column].value_counts()
    df[column + '_freq'] = df[column].map(freq)
    return df
```

### 4. Target Encoding
```python
def target_encode(df, target_column, categorical_columns):
    for column in categorical_columns:
        mean_target = df.groupby(column)[target_column].mean()
        df[column + '_target_enc'] = df[column].map(mean_target)
    return df
```

### 5. Binary Encoding
```python
import category_encoders as ce

def binary_encode(df, column):
    encoder = ce.BinaryEncoder(cols=[column])
    df = encoder.fit_transform(df)
    return df
```

## Example Usage
```python
if __name__ == '__main__':
    # Example DataFrame
    df = pd.DataFrame({'health_condition': ['Condition A', 'Condition B', 'Condition A', 'Condition C'],
                       'target': [1, 0, 1, 0]})
    
    # Apply feature engineering methods
    df = one_hot_encode(df, 'health_condition')
    df = label_encode(df, 'health_condition')
    df = frequency_encode(df, 'health_condition')
    df = target_encode(df, 'target', ['health_condition'])
    df = binary_encode(df, 'health_condition')
    
    print(df)
```

## Conclusion
The above methods provide a robust toolkit for feature engineering and encoding for categorical health data, ensuring improved predictive modeling performance.