# Apriori Algorithm for Market Basket Analysis

## Overview

This code implements the Apriori algorithm, a popular algorithm in Machine Learning, for Market Basket Analysis. Market Basket Analysis is used to discover interesting associations and relationships between items in large datasets, particularly in the context of retail transactions.

## Libraries Used

- **NumPy:** Numerical computing library in Python.
- **Pandas:** Data manipulation and analysis library.
- **mlxtend:** Machine learning extensions library containing tools for frequent pattern mining and association rule learning.

## Loading and Preprocessing Data

```python
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Loading the data from an Excel file
data = pd.read_excel('Online_Retail.xlsx')
data.head()

# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping rows without an invoice number and converting InvoiceNo to string
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping transactions done on credit
data = data[~data['InvoiceNo'].str.contains('C')]
```

## Creating Transaction Baskets

```python
# Transactions done in different countries
basket_France = ...
basket_UK = ...
basket_Por = ...
basket_Sweden = ...
```

## Hot Encoding and Data Preparation

```python
# Defining the hot encoding function
def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# Encoding the datasets
basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded

basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded

basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded
```

## Building the Apriori Model and Association Rules

```python
# Building the model for transactions in France
frq_items = apriori(basket_France, min_support=0.05, use_colnames=True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
rules.head()

# Similar steps for transactions in the United Kingdom, Portugal, and Sweden
```

## Results

The final output includes the discovered association rules for each country, sorted by confidence and lift. These rules provide insights into items frequently purchased together, aiding in strategic decision-making for product recommendations and marketing strategies.

Feel free to customize the parameters and experiment with different support thresholds to explore varying levels of item associations.

Last updated on: 2024-02-13

Last updated on: 2024-02-17

Last updated on: 2024-02-17

Last updated on: 2024-02-17

Last updated on: 2024-02-18

Last updated on: 2024-02-19

Last updated on: 2024-02-20

Last updated on: 2024-02-29

Last updated on: 2024-03-02

Last updated on: 2024-03-06

Last updated on: 2024-03-08

Last updated on: 2024-03-15

Last updated on: 2024-03-17

Last updated on: 2024-03-20

Last updated on: 2024-03-23

Last updated on: 2024-04-03

Last updated on: 2024-04-08