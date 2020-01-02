# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Load Offers
offers = pd.read_excel(path,sheet_name=0)

# Load Transactions
transactions = pd.read_excel(path, sheet_name=1)
transactions['n'] = 1

# Merge dataframes
df = pd.merge(offers,transactions)

# Look at the first 5 rows
print(df.head())






# --------------
# Code starts here

# create pivot table
matrix = df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')

# replace missing values with 0
matrix.fillna(0, inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])

matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns denoting observation locations in decomposed form
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,[0,33,34,35]]

# visualize clusters
plt.scatter(x=clusters['x'], y=clusters['y'], c=clusters['cluster'], cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Customer Segmentation")
plt.show()

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data = pd.merge(clusters, transactions)

# merge `data` and `offers`
data = pd.merge(offers, data)
data.head()
# initialzie empty dictionary
champagne = {}

# iterate over every cluster
for i in range(0,5):
    new_df = data[data['cluster'] == i]
    new_df.head()
    # observation falls in that cluster
    
    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)

    # check if 'Champagne' is ordered mostly
    if counts.index[0] == 'Champagne':
        # add it to 'champagne'
        champagne[i] = counts[0]

# get cluster with maximum orders of 'Champagne' 
cluster_champagne = max(champagne, key= champagne.get)

# print out cluster number
print("The Cluster which ordered the most Champagne is : ",cluster_champagne)



# --------------
# Code starts here

# empty dictionary
discount = {}

# iterate over cluster numbers
for i in range(0,5):
    # dataframe for every cluster
    new_df = data[data['cluster'] == i]
    # average discount for cluster
    counts = new_df['Discount (%)'].sum() / len(new_df)
    # adding cluster number as key and average discount as value 
    discount[i] = counts

# cluster with maximum average discount
cluster_discount = max(discount, key= discount.get)
print("The cluster number with the maximum average discount percentage : ",cluster_discount)
# Code ends here


