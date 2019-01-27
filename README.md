# House-Price-Prediction-Advanced-Regression

## Imported all the required library
```
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

```
### Loading and Viewing the data

~~~
data=pd.read_csv('housing.csv')
data.head()
~~~
# Data Visualisation

### Ploting the Heatmap

![alt Survived](https://rahuljadli.github.io/House-Price-Prediction-Advanced-Regression/screen_shots/heatmap.png)

### Ploting Ocean Proximity

![alt Age](https://github.com/rahuljadli/Housing-Price-Prediction/blob/master/screen_shots/OceanHouses.png)

### Ploting Lattitude effect on house price

![alt Sex Survival ](https://github.com/rahuljadli/Housing-Price-Prediction/blob/master/screen_shots/latitude-price.png)

### Ploting Ocean Median Price

![alt Sex Survival ](https://github.com/rahuljadli/Housing-Price-Prediction/blob/master/screen_shots/ocean-effect-on-median-value.png)

### Ploting Income effect on house price

![alt Sex Survival ](https://github.com/rahuljadli/Housing-Price-Prediction/blob/master/screen_shots/Income-effect-on-house-value.png)

## Data filling

~~~
housing_mean=housing_data.fillna(housing_data['total_bedrooms'].mean())
~~~

# Using Different Model's 

## Creating Training and Testing Data set

~~~
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

~~~
# Training the model

~~~
model=LogisticRegression()
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~

## Getting the accuracy score

~~~
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(prediction, y_test))
rmse
~~~
## Got RMSE value of 69140.009

