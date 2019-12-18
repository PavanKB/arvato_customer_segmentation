# Arvato Customer Segmentation

# 1. Introduction
###*"How can the business client acquire more customers efficiently."*
In this project, we use the attributes and demographics of the existing customers and
check against the broader population to identify new customers for our client.

# 2. Files
The project is organised as follows:

## 2.1 Data
* `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
* `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
* `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
* `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
* `DIAS Attributes - Values 2017.xlsx`: 
* `DIAS Information Levels - Attributes 2017.xlsx`: 
Attributes meta data file has some data as merged cells that needs to be handled.

There also columns in data that are not represented in the meta data at all!
We drop those columns- why> becasue we dont know how to treat them. 

## 2.2 Code
### `arvato_model`
Contains the functions to use for EDA, plotting, model training and feature extraction 

1. **`plot`**
1. **`stats`**
1. **`utils`**

### The following packages were used. Python version 3.7.5
1. `pandas 0.25.3`
1. `numpy 1.17.4`
1. `matplotlib 3.1.1`
1. `seaborn 0.9.0`
1. `sklearn 0.21.3`
1. `scipy 1.3.1`
1. `feather`
1. `joblib`

# 3. How do we run the code?
The entire journey of data extraction, EDA, identifying cluster and training the classification
model have been coded into the Jupyter notebook. 

When running the for the first time, ensure that `USE_CACHE` and `USE_MODEL_CACHE` are set to `False`

The notebook saves the intermediate data sets and the fitted transforms as the code progress.
Subsequent runs can then use these cached values.
  


