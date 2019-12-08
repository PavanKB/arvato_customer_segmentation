# Arvato Customer Segmentation

# Introduction
Explain the business case and questions we are trying to answer

# Files
The project is organised as follows:
* `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
* `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
* `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
* `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

**TODO** Add the codes as well.

# Packages 
What are the packages needed?
`pandas`

How do we run the code?

# EDA
Explore the data

Plots etc
NA analysis
Correlation?
Uniqueness of the data in each columns
- Shannon
- Simpson 


# CleanUP
1. Fix the error of 18, 19 columns.
```python
/opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (18,19) have mixed types. Specify dtype option on import or set low_memory=False.
  interactivity=interactivity, compiler=compiler, result=result)
```
We forced all columns to be of type str and checked the unique values for columns
18 and 19 in azdias and customer.  Turns out there are X, XX. 
We shall force them to be NA?

Also there is `OST_WEST_KZ` O,W

1. NA Clean Up
1. Drop the non-diverse columns.
1. Drop the correlated columns. (use spearman)
1. One-hot encoding

# Dimensionality Reduction

# Decide on the ML algo for customer segmentation
1. K-means clustering
1. Hierarchical clustering
1. SVD
1. PCA

Combined?

The challenge is how do we identify the factors which differentiate the
customers from the population.
1. Do a pairwise correlation of each column between data sets?
1. Can we use PCA to see which factors explain the difference?
1. What about latent variables?


# Testing
Test and Train data split
Grid search for the best  - K-Fold

Give the metrics
Give the ROC

# Applications
Must be able to run from command prompt.  


# References
1. https://medium.com/machine-learning-for-humans/unsupervised-learning-f45587588294
1. http://andrew.gibiansky.com/blog/mathematics/cool-linear-algebra-singular-value-decomposition/
1. http://www.tiem.utk.edu/~gross/bioed/bealsmodules/simpsonDI.html
1. http://www.tiem.utk.edu/~gross/bioed/bealsmodules/shannonDI.html
 
