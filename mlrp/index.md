 MIT Applied Data Science Program 
 
 Machine Learning Regression Project
 
Boston House Price Prediction

Hellen S Momoh

October 21 2022

# Context

Housing is one of our most essential needs, along with other fundamental needs. 
Over the years, people's living standards have  improved so as the demand for housing. 

We will use machine learning algorithms to build a prediction model for house prices that may provide a lot of information and knowledge to home buyers, property investors, and housebuilders, such as the valuation of house prices in the present market, which will help them determine house prices. Meanwhile, this model can help potential buyers decide the characteristics of a house they want according to their budget.

# Objective

In this project, we were asked to predict the housing prices of a town or a suburb based on the features of the locality provided to us. In the process, we need to identify the most important features affecting the price of the house. We need to employ techniques of data preprocessing and build a linear regression model that predicts the prices for the unseen data.


```python
# libraries for data manipulation
import pandas as pd

import numpy as np

# libraries for data visualization
import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.graphics.gofplots import ProbPlot

# libraries for building linear regression model
from statsmodels.formula.api import ols

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

# library for preparing data
from sklearn.model_selection import train_test_split

# library for data preprocessing
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
```


```python
df = pd.read_csv('Boston.csv')
df.head()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



# Observation

The variable MEDV is the target variable and the rest of the variables are independent variables based on which we will predict the house price (MEDV).


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 13 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    int64  
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    int64  
     9   TAX      506 non-null    int64  
     10  PTRATIO  506 non-null    float64
     11  LSTAT    506 non-null    float64
     12  MEDV     506 non-null    float64
    dtypes: float64(10), int64(3)
    memory usage: 51.5 KB
    

# Observation
According to the info of the data, there are a total of 506 non-null observations in each of the columns indicating that 
there are no missing values in the data. There are 13 columns in the dataset and every column is of numeric data type.



```python
#QUESTION1
df.describe()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#QUESTION1
df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>506.0</td>
      <td>3.613524</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>0.082045</td>
      <td>0.25651</td>
      <td>3.677083</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>506.0</td>
      <td>11.363636</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>12.500000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>506.0</td>
      <td>11.136779</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>5.190000</td>
      <td>9.69000</td>
      <td>18.100000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>506.0</td>
      <td>0.069170</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>506.0</td>
      <td>0.554695</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.449000</td>
      <td>0.53800</td>
      <td>0.624000</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>506.0</td>
      <td>6.284634</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>5.885500</td>
      <td>6.20850</td>
      <td>6.623500</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>506.0</td>
      <td>68.574901</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>45.025000</td>
      <td>77.50000</td>
      <td>94.075000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>506.0</td>
      <td>3.795043</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>2.100175</td>
      <td>3.20745</td>
      <td>5.188425</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>506.0</td>
      <td>9.549407</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>5.00000</td>
      <td>24.000000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>506.0</td>
      <td>408.237154</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>279.000000</td>
      <td>330.00000</td>
      <td>666.000000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>506.0</td>
      <td>18.455534</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>17.400000</td>
      <td>19.05000</td>
      <td>20.200000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>506.0</td>
      <td>12.653063</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>6.950000</td>
      <td>11.36000</td>
      <td>16.955000</td>
      <td>37.9700</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>506.0</td>
      <td>22.532806</td>
      <td>9.197104</td>
      <td>5.00000</td>
      <td>17.025000</td>
      <td>21.20000</td>
      <td>25.000000</td>
      <td>50.0000</td>
    </tr>
  </tbody>
</table>
</div>



# Observation:


The summary statistics of this data computes the five number summary.

It also counts the number of variables in the dataset.

The statistical components of these numbers are their measures of dispersions.

The MEDV column/variable has a total count of 506.0 which indicates the sample studied.

The mean is 22.532806 which is the sample average, the std is 9.197104	, and the min is 5.00000

In terms of quartiles, 17.025000 is the lowest quartile and 25.000000 is the upper quartile.

The interquartile range is 8 and the range itself is 45. The meadian is 21.20000 and it measures the average unit.



```python
#QUESTION2
#Plotting all the columns to look at their distributions
for i in df.columns:
    
    plt.figure(figsize = (7, 4))
    
    sns.histplot(data = df, x = i, kde = True)
    
    plt.show()
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



    
![png](output_11_3.png)
    



    
![png](output_11_4.png)
    



    
![png](output_11_5.png)
    



    
![png](output_11_6.png)
    



    
![png](output_11_7.png)
    



    
![png](output_11_8.png)
    



    
![png](output_11_9.png)
    



    
![png](output_11_10.png)
    



    
![png](output_11_11.png)
    



    
![png](output_11_12.png)
    


# Observation

Theere are diverse relationships between these distributions. The dependent varibale MEDV is seen to be slightly skewed which means we must apply a log transformation on the 'MEDV' column and check the distribution of the transformed column.


```python
df['MEDV_log'] = np.log(df['MEDV'])
```


```python
sns.histplot(data = df, x = 'MEDV_log', kde = True)
```




    <AxesSubplot:xlabel='MEDV_log', ylabel='Count'>




    
![png](output_14_1.png)
    


# Observation

Now that we have applied the log transformation on the dependent variable MEDV, the variable appears to have a nearly normal distribution without skew, which means we can proceed.
Best practice is to check the bivariate relationship between the variables before creating the linear regression model. 
We will use the heatmap and scatter plot to understand this relationship. 

```python
#QUESTION3
corr_matrix = df.corr()

corr_matrix

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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
      <th>MEDV</th>
      <th>MEDV_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>1.000000</td>
      <td>-0.200469</td>
      <td>0.406583</td>
      <td>-0.055892</td>
      <td>0.420972</td>
      <td>-0.219247</td>
      <td>0.352734</td>
      <td>-0.379670</td>
      <td>0.625505</td>
      <td>0.582764</td>
      <td>0.289946</td>
      <td>0.455621</td>
      <td>-0.388305</td>
      <td>-0.527946</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>-0.200469</td>
      <td>1.000000</td>
      <td>-0.533828</td>
      <td>-0.042697</td>
      <td>-0.516604</td>
      <td>0.311991</td>
      <td>-0.569537</td>
      <td>0.664408</td>
      <td>-0.311948</td>
      <td>-0.314563</td>
      <td>-0.391679</td>
      <td>-0.412995</td>
      <td>0.360445</td>
      <td>0.363344</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>0.406583</td>
      <td>-0.533828</td>
      <td>1.000000</td>
      <td>0.062938</td>
      <td>0.763651</td>
      <td>-0.391676</td>
      <td>0.644779</td>
      <td>-0.708027</td>
      <td>0.595129</td>
      <td>0.720760</td>
      <td>0.383248</td>
      <td>0.603800</td>
      <td>-0.483725</td>
      <td>-0.541556</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>-0.055892</td>
      <td>-0.042697</td>
      <td>0.062938</td>
      <td>1.000000</td>
      <td>0.091203</td>
      <td>0.091251</td>
      <td>0.086518</td>
      <td>-0.099176</td>
      <td>-0.007368</td>
      <td>-0.035587</td>
      <td>-0.121515</td>
      <td>-0.053929</td>
      <td>0.175260</td>
      <td>0.158412</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>0.420972</td>
      <td>-0.516604</td>
      <td>0.763651</td>
      <td>0.091203</td>
      <td>1.000000</td>
      <td>-0.302188</td>
      <td>0.731470</td>
      <td>-0.769230</td>
      <td>0.611441</td>
      <td>0.668023</td>
      <td>0.188933</td>
      <td>0.590879</td>
      <td>-0.427321</td>
      <td>-0.510600</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>-0.219247</td>
      <td>0.311991</td>
      <td>-0.391676</td>
      <td>0.091251</td>
      <td>-0.302188</td>
      <td>1.000000</td>
      <td>-0.240265</td>
      <td>0.205246</td>
      <td>-0.209847</td>
      <td>-0.292048</td>
      <td>-0.355501</td>
      <td>-0.613808</td>
      <td>0.695360</td>
      <td>0.632021</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.352734</td>
      <td>-0.569537</td>
      <td>0.644779</td>
      <td>0.086518</td>
      <td>0.731470</td>
      <td>-0.240265</td>
      <td>1.000000</td>
      <td>-0.747881</td>
      <td>0.456022</td>
      <td>0.506456</td>
      <td>0.261515</td>
      <td>0.602339</td>
      <td>-0.376955</td>
      <td>-0.453422</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-0.379670</td>
      <td>0.664408</td>
      <td>-0.708027</td>
      <td>-0.099176</td>
      <td>-0.769230</td>
      <td>0.205246</td>
      <td>-0.747881</td>
      <td>1.000000</td>
      <td>-0.494588</td>
      <td>-0.534432</td>
      <td>-0.232471</td>
      <td>-0.496996</td>
      <td>0.249929</td>
      <td>0.342780</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.625505</td>
      <td>-0.311948</td>
      <td>0.595129</td>
      <td>-0.007368</td>
      <td>0.611441</td>
      <td>-0.209847</td>
      <td>0.456022</td>
      <td>-0.494588</td>
      <td>1.000000</td>
      <td>0.910228</td>
      <td>0.464741</td>
      <td>0.488676</td>
      <td>-0.381626</td>
      <td>-0.481971</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>0.582764</td>
      <td>-0.314563</td>
      <td>0.720760</td>
      <td>-0.035587</td>
      <td>0.668023</td>
      <td>-0.292048</td>
      <td>0.506456</td>
      <td>-0.534432</td>
      <td>0.910228</td>
      <td>1.000000</td>
      <td>0.460853</td>
      <td>0.543993</td>
      <td>-0.468536</td>
      <td>-0.561466</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>0.289946</td>
      <td>-0.391679</td>
      <td>0.383248</td>
      <td>-0.121515</td>
      <td>0.188933</td>
      <td>-0.355501</td>
      <td>0.261515</td>
      <td>-0.232471</td>
      <td>0.464741</td>
      <td>0.460853</td>
      <td>1.000000</td>
      <td>0.374044</td>
      <td>-0.507787</td>
      <td>-0.501729</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>0.455621</td>
      <td>-0.412995</td>
      <td>0.603800</td>
      <td>-0.053929</td>
      <td>0.590879</td>
      <td>-0.613808</td>
      <td>0.602339</td>
      <td>-0.496996</td>
      <td>0.488676</td>
      <td>0.543993</td>
      <td>0.374044</td>
      <td>1.000000</td>
      <td>-0.737663</td>
      <td>-0.805034</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>-0.388305</td>
      <td>0.360445</td>
      <td>-0.483725</td>
      <td>0.175260</td>
      <td>-0.427321</td>
      <td>0.695360</td>
      <td>-0.376955</td>
      <td>0.249929</td>
      <td>-0.381626</td>
      <td>-0.468536</td>
      <td>-0.507787</td>
      <td>-0.737663</td>
      <td>1.000000</td>
      <td>0.953155</td>
    </tr>
    <tr>
      <th>MEDV_log</th>
      <td>-0.527946</td>
      <td>0.363344</td>
      <td>-0.541556</td>
      <td>0.158412</td>
      <td>-0.510600</td>
      <td>0.632021</td>
      <td>-0.453422</td>
      <td>0.342780</td>
      <td>-0.481971</td>
      <td>-0.561466</td>
      <td>-0.501729</td>
      <td>-0.805034</td>
      <td>0.953155</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#QUESTION3

plt.figure(figsize = (12, 8))

cmap = sns.diverging_palette(230, 20, as_cmap = True)

sns.heatmap(corr_matrix, annot = True, fmt = '.2f', cmap = cmap)

plt.show()

```


    
![png](output_18_0.png)
    


# Observation

The heatmap show significant correlations between the different pairs of features. i.e. "LSTAT" and "MEDV." 

The correlation between these varibales violates the linear assumptions. We will visualize these realtionships on a scatterplot to understand their patterns. 


```python
#Scatterplot to visualize the relationship between AGE and DIS
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'AGE', y = 'DIS', data = df)

plt.show()
```


    
![png](output_20_0.png)
    


# Observations

The distance of the houses to the Boston employment centers appears to decrease moderately as the the proportion of the old houses increase in the town. It is possible that the Boston employment centers are located in the established towns where proportion of owner-occupied units built prior to 1940 is comparatively high.


```python
# Scatterplot to visulaize the relationship between RAD and TAX
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'RAD', y = 'TAX', data = df)

plt.show()
```


    
![png](output_22_0.png)
    


# Observations:

The correlation between RAD and TAX is very high. But, no trend is visible between the two variables.
We assume that te strong correlation might be due to outliers.


```python
# Scatterplot to visulaize the relationship between RAD and TAX
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'RAD', y = 'TAX', data = df)

plt.show()
```


    
![png](output_24_0.png)
    

Now we check the correlation after removing the outliers.

```python
# Removing the data corresponding to high tax rate
df1 = df[df['TAX'] < 600]

# the required function
from scipy.stats import pearsonr

# Calculating the correlation
print('The correlation between TAX and RAD is', pearsonr(df1['TAX'], df1['RAD'])[0])
```

    The correlation between TAX and RAD is 0.249757313314292
    

# Observation



Well we were right. The high correlation between TAX and RAD is due to the outliers. 
The tax rate for some properties might be higher due to some other reason.

Now we visualize the relationship between the remaining features.


```python

plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'INDUS', y = 'TAX', data = df)

plt.show()
```


    
![png](output_29_0.png)
    


# Observation

The tax rate appears to increase with an increase in the proportion of non-retail business acres per town. This might be due to the reason that the variables TAX and INDUS are related with a third variable.


```python

plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'RM', y = 'MEDV', data = df)

plt.show()
```


    
![png](output_31_0.png)
    


# Observations:

The price of the house seems to increase as the value of RM increases. This is expected as the price is generally higher for more rooms.

There are a few outliers in a horizontal line as the MEDV value seems to be capped at 50.


```python
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'LSTAT', y = 'MEDV', data = df)

plt.show()
```


    
![png](output_33_0.png)
    


# Observations:

The price of the house tends to decrease with an increase in LSTAT. This is also possible as the house price is lower in areas where lower status people live.
There are few outliers and the data seems to be capped at 50.


```python
#QUESTION4

plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'INDUS', y = 'NOX', data = df)

plt.show()
```


    
![png](output_35_0.png)
    


# Observation

The plot shows that the Proportion of non-retail business acres per town (INDUS) falls in areas with higher Nitric Oxide concentration (parts per 10 million)(NOX), and places where there are more NOX, there are less INDUS. 


```python
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'AGE', y = 'NOX', data = df)

plt.show()
```


    
![png](output_37_0.png)
    


# Observation

The scatterplot shows a mix realtionship between "AGE and NOX." Older and younger people live in places with high levels of NOX. Other older people seem to care not much about NOX hence, places where there are non much NOX older people are less likely to live in those places.


```python
plt.figure(figsize = (6, 6))

sns.scatterplot(x = 'DIS', y = 'NOX', data = df)

plt.show()
```


    
![png](output_39_0.png)
    


# Observation

There is also a mix relationship between "DIS and NOX." Weighted distances to Boston employment centers are not much in places with lower levels of "NOX."

# Observation
We have seen that the variables LSTAT and RM have a linear relationship with the dependent variable MEDV. Also, there are significant relationships among few independent variables, which is not desirable for a linear regression model. Let's first split the dataset and see. 
Now we split the dataset into the dependent and independent variables and further split it into train and test set in a ratio of 70:30 for train and test sets.

```python
#separating the dependent variable and indepedent variables
Y = df['MEDV_log']

X = df.drop(columns = {'MEDV', 'MEDV_log'})

X = sm.add_constant(X)
```


```python
# splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)
```
Now we check for multicollinearity in the training dataset.

We will use the Variance Inflation Factor (VIF), to check if there is multicollinearity in the data.

Features having a VIF score > 5 will be dropped / treated till all the features have a VIF score < 5

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# checking the VIF
def checking_vif(train):
    vif = pd.DataFrame()
    vif["feature"] = train.columns

    # Calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(train.values, i) for i in range(len(train.columns))
    ]
    return vif


print(checking_vif(X_train))
```

        feature         VIF
    0     const  535.372593
    1      CRIM    1.924114
    2        ZN    2.743574
    3     INDUS    3.999538
    4      CHAS    1.076564
    5       NOX    4.396157
    6        RM    1.860950
    7       AGE    3.150170
    8       DIS    4.355469
    9       RAD    8.345247
    10      TAX   10.191941
    11  PTRATIO    1.943409
    12    LSTAT    2.861881
    

# Observations:

There are two variables with a high VIF - RAD and TAX (greater than 5).

So we remove TAX as it has the highest VIF values and check the multicollinearity again.


```python
#QUESTION5
#Creating the model after dropping TAX
X_train = X_train.drop(['TAX'], axis = 1)

# Checking for VIF again
print(checking_vif(X_train))
```

        feature         VIF
    0     const  532.025529
    1      CRIM    1.923159
    2        ZN    2.483399
    3     INDUS    3.270983
    4      CHAS    1.050708
    5       NOX    4.361847
    6        RM    1.857918
    7       AGE    3.149005
    8       DIS    4.333734
    9       RAD    2.942862
    10  PTRATIO    1.909750
    11    LSTAT    2.860251
    

# Observation

Now the VIF is less than 5 for all the independent variables so we will create the linear regression model. We can also assume that multicollinearity has been removed between the variables.


```python
#QUESTION6
#Creating the model
model1 = sm.OLS(y_train, X_train).fit()

# Get the model summary
model1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>MEDV_log</td>     <th>  R-squared:         </th> <td>   0.769</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.761</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   103.3</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 21 Nov 2022</td> <th>  Prob (F-statistic):</th> <td>1.40e-101</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:31:46</td>     <th>  Log-Likelihood:    </th> <td>  76.596</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   354</td>      <th>  AIC:               </th> <td>  -129.2</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   342</td>      <th>  BIC:               </th> <td>  -82.76</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    4.6324</td> <td>    0.243</td> <td>   19.057</td> <td> 0.000</td> <td>    4.154</td> <td>    5.111</td>
</tr>
<tr>
  <th>CRIM</th>    <td>   -0.0128</td> <td>    0.002</td> <td>   -7.445</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.009</td>
</tr>
<tr>
  <th>ZN</th>      <td>    0.0010</td> <td>    0.001</td> <td>    1.425</td> <td> 0.155</td> <td>   -0.000</td> <td>    0.002</td>
</tr>
<tr>
  <th>INDUS</th>   <td>   -0.0004</td> <td>    0.003</td> <td>   -0.148</td> <td> 0.883</td> <td>   -0.006</td> <td>    0.005</td>
</tr>
<tr>
  <th>CHAS</th>    <td>    0.1196</td> <td>    0.039</td> <td>    3.082</td> <td> 0.002</td> <td>    0.043</td> <td>    0.196</td>
</tr>
<tr>
  <th>NOX</th>     <td>   -1.0598</td> <td>    0.187</td> <td>   -5.675</td> <td> 0.000</td> <td>   -1.427</td> <td>   -0.692</td>
</tr>
<tr>
  <th>RM</th>      <td>    0.0532</td> <td>    0.021</td> <td>    2.560</td> <td> 0.011</td> <td>    0.012</td> <td>    0.094</td>
</tr>
<tr>
  <th>AGE</th>     <td>    0.0003</td> <td>    0.001</td> <td>    0.461</td> <td> 0.645</td> <td>   -0.001</td> <td>    0.002</td>
</tr>
<tr>
  <th>DIS</th>     <td>   -0.0503</td> <td>    0.010</td> <td>   -4.894</td> <td> 0.000</td> <td>   -0.071</td> <td>   -0.030</td>
</tr>
<tr>
  <th>RAD</th>     <td>    0.0076</td> <td>    0.002</td> <td>    3.699</td> <td> 0.000</td> <td>    0.004</td> <td>    0.012</td>
</tr>
<tr>
  <th>PTRATIO</th> <td>   -0.0452</td> <td>    0.007</td> <td>   -6.659</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.032</td>
</tr>
<tr>
  <th>LSTAT</th>   <td>   -0.0298</td> <td>    0.002</td> <td>  -12.134</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.025</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>30.699</td> <th>  Durbin-Watson:     </th> <td>   1.923</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  83.718</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.372</td> <th>  Prob(JB):          </th> <td>6.62e-19</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.263</td> <th>  Cond. No.          </th> <td>2.09e+03</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.09e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



# Observation

There are variables with p-values greater than 5 that are statistically irrelevant that we need to drop in other to build a better model. 

As we observe and determine the significance of the model, we drop the insignificant variables with p-value > 0.05 and create the regression model again.

Significance in this problems means whether the population regression parameters are significantly different from zero.

From the above, it may be noted that the regression coefficients corresponding to ZN, AGE, and INDUS are not statistically significant at level α = 0.05. In other words, we say that the regression coefficients corresponding to these three are not significantly different from 0 in the population. Hence, we will eliminate the three features and create a new model.



```python
#QUESTION7

#Creating a new model after dropping columns 'MEDV', 'MEDV_log', 'TAX', 'ZN', 'AGE', 'INDUS' from df DataFrame
Y = df['MEDV_log']

X = df.drop(['MEDV', 'MEDV_log', 'TAX', 'ZN', 'AGE', 'INDUS'], axis =1) 

X = sm.add_constant(X)

# Splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30 , random_state = 1)

# Creating the new model
model2 = sm.OLS(y_train, X_train).fit()

model2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>MEDV_log</td>     <th>  R-squared:         </th> <td>   0.767</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.762</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   142.1</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 21 Nov 2022</td> <th>  Prob (F-statistic):</th> <td>2.61e-104</td>
</tr>
<tr>
  <th>Time:</th>                 <td>20:52:59</td>     <th>  Log-Likelihood:    </th> <td>  75.486</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   354</td>      <th>  AIC:               </th> <td>  -133.0</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   345</td>      <th>  BIC:               </th> <td>  -98.15</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>    4.6494</td> <td>    0.242</td> <td>   19.242</td> <td> 0.000</td> <td>    4.174</td> <td>    5.125</td>
</tr>
<tr>
  <th>CRIM</th>    <td>   -0.0125</td> <td>    0.002</td> <td>   -7.349</td> <td> 0.000</td> <td>   -0.016</td> <td>   -0.009</td>
</tr>
<tr>
  <th>CHAS</th>    <td>    0.1198</td> <td>    0.039</td> <td>    3.093</td> <td> 0.002</td> <td>    0.044</td> <td>    0.196</td>
</tr>
<tr>
  <th>NOX</th>     <td>   -1.0562</td> <td>    0.168</td> <td>   -6.296</td> <td> 0.000</td> <td>   -1.386</td> <td>   -0.726</td>
</tr>
<tr>
  <th>RM</th>      <td>    0.0589</td> <td>    0.020</td> <td>    2.928</td> <td> 0.004</td> <td>    0.019</td> <td>    0.098</td>
</tr>
<tr>
  <th>DIS</th>     <td>   -0.0441</td> <td>    0.008</td> <td>   -5.561</td> <td> 0.000</td> <td>   -0.060</td> <td>   -0.028</td>
</tr>
<tr>
  <th>RAD</th>     <td>    0.0078</td> <td>    0.002</td> <td>    3.890</td> <td> 0.000</td> <td>    0.004</td> <td>    0.012</td>
</tr>
<tr>
  <th>PTRATIO</th> <td>   -0.0485</td> <td>    0.006</td> <td>   -7.832</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.036</td>
</tr>
<tr>
  <th>LSTAT</th>   <td>   -0.0293</td> <td>    0.002</td> <td>  -12.949</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.025</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>32.514</td> <th>  Durbin-Watson:     </th> <td>   1.925</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  87.354</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.408</td> <th>  Prob(JB):          </th> <td>1.07e-19</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.293</td> <th>  Cond. No.          </th> <td>    690.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



# Observation

We drop the irrelevant features with p-values greater than 5 that were statistically insignificant to the model. 
Now we check to see if we meet the linear regression assumptions.


Mean of residuals should be 0
No Heteroscedasticity
Linearity of variables
Normality of error terms

```python
#QUESTION8
residuals = model2.resid

np.mean(residuals)
```




    5.532297783725356e-16



# Observation
Residuals are diagnostisc measures used when assessing the quality of a machine learning model. 
The mean of residuals is not very far from 0. Hence, the corresponding assumption is satisfied.

Next we check for homoscedasticity since we anticipate that the residuals are symmetrically distributed across the regression line.
To determine homoscedasticity, We'll use Goldfeldquandt Test to test the following hypothesis with alpha = 0.05:

Null hypothesis: Residuals are homoscedastic 
Alternate hypothesis: Residuals have heteroscedastic

```python
from statsmodels.stats.diagnostic import het_white

from statsmodels.compat import lzip

import statsmodels.stats.api as sms
```


```python
name = ["F statistic", "p-value"]

test = sms.het_goldfeldquandt(y_train, X_train)

lzip(name, test)

```




    [('F statistic', 1.083508292342529), ('p-value', 0.30190120067668275)]



# Observation

Since the p-value < 0.05, we can reject the Null Hypothesis that the residuals are homoscedastic and the corresponding assumption is satisfied. 
Now we check the Linearity of variables which states that the predictor variables must have a linear relation with the dependent variable.

To test the assumption, we'll plot residuals and the fitted values on a plot and ensure that residuals do not form a strong pattern. They should be randomly and uniformly scattered on the x-axis.

```python
# Predicted values
fitted = model2.fittedvalues

# sns.set_style("whitegrid")
sns.residplot(x = fitted, y = residuals, color = "lightblue", lowess = True)

plt.xlabel("Fitted Values")

plt.ylabel("Residual")

plt.title("Residual PLOT")

plt.show()
```


    
![png](output_63_0.png)
    


# Observation

There is no pattern in the residual vs fitted values plot. Hence, the corresponding assumption is satisfied.
Now we check the normality of error terms
The residuals should be normally distributed.

```python
# Plotting the histogram of residuals

sns.histplot(residuals, kde = True)
```




    <AxesSubplot:ylabel='Count'>




    
![png](output_66_1.png)
    


# Observation

The histogram of the residuals is normally distributed.


```python
# Plot q-q plot of residuals
import pylab

import scipy.stats as stats

stats.probplot(residuals, dist = "norm", plot = pylab)

plt.show()
```


    
![png](output_68_0.png)
    


# Observation

From the above plots, the residuals seem to follow a normal distribution. Hence, the corresponding assumption is satisfied. Now, we will check the model performance on the train and test dataset.
Next, we check the performance of the model on the train and test data set and compare the model performance of train and test dataset.

```python
#QUESTION9

#RMSE
def rmse(predictions, targets):
    return np.sqrt(((targets - predictions) ** 2).mean())


# MAPE
def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions)) / targets) * 100


# MAE
def mae(predictions, targets):
    return np.mean(np.abs((targets - predictions)))


# Model Performance on test and train data
def model_pref(olsmodel, x_train, x_test):

    # In-sample Prediction
    y_pred_train = olsmodel.predict(x_train)
    y_observed_train = y_train

    # Prediction on test data
    y_pred_test = olsmodel.predict(x_test)
    y_observed_test = y_test

    print(
        pd.DataFrame(
            {
                "Data": ["Train", "Test"],
                "RMSE": [
                    rmse(y_pred_train, y_observed_train),
                    rmse(y_pred_test, y_observed_test),
                ],
                "MAE": [
                    mae(y_pred_train, y_observed_train),
                    mae(y_pred_test, y_observed_test),
                ],
                "MAPE": [
                    mape(y_pred_train, y_observed_train),
                    mape(y_pred_test, y_observed_test),
                ],
            }
        )
    )


# Checking model performance
model_pref(model2, X_train, X_test)  
```

        Data      RMSE       MAE      MAPE
    0  Train  0.195504  0.143686  4.981813
    1   Test  0.198045  0.151284  5.257965
    

# Observation

After checking the performance of the model on the train and test data set and comparing the model performance of train and test data, we see that the Mean Absolute Percentage Error (MAPE) which is the mean of all absolute percentage errors between the predicted and actual values is > 5.0 which places doubt on the performance of the model hence, we apply cross validation to improve the model and evaluate it using different evaluation metrics. 


```python
# Importing the required function

from sklearn.model_selection import cross_val_score

# Building the regression model and cross-validating
linearregression = LinearRegression()                                    

cv_Score11 = cross_val_score(linearregression, X_train, y_train, cv = 10)
cv_Score12 = cross_val_score(linearregression, X_train, y_train, cv = 10, 
                             scoring = 'neg_mean_squared_error')                                  


print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std() * 2))
print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1*cv_Score12.mean(), cv_Score12.std() * 2))
```

    RSquared: 0.729 (+/- 0.232)
    Mean Squared Error: 0.041 (+/- 0.023)
    


```python
#Question 11: 

#The conclusions and business recommendations derived from the model.
```

In this project, we used Machine Learning algorithms to identify the most important features affecting the price of housing in Boston. We employ the techniques of data preprocessing and built a linear regression model that predicts the prices for the unseen data. This prediction model will provide a lot of information and knowledge to home buyers, property investors, and potential buyers to decide the characteristics of the type of housing they prefer.

Recommendation: After conducting a proper engineering on the data by following all the necessary steps of gathering the summary statistics, data preprocessing and so on, It is recommended that property investors, and potential home buyers consider the listed independent features before making a business informed decision. This recommendation is based on the performance of our regression model with a Mean Squared Errro that is close to zero representing better quality of the regression model and an RSquared that measures the Goodness of Fit of the regression model determining a strong relationship between our dependent and independent features with a convenient score of 70+/100. 
