# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here

# read the dataset
dataset = pd.read_csv(path)

# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset = dataset.drop('Id', 1)

# check the statistical description
dataset.describe()




# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns.values

#number of attributes (exclude target)
size = len(cols) -1

#x-axis has target attribute to distinguish between classes
#x = dataset.iloc[:,-1]
x= cols[size]

#y-axis shows values of an attribute
#y = dataset.iloc[:,:-1]
y= cols[0:size]

#Plot violin for all attributes
for i in range(0,size):
    sns.violinplot(x=x, y=y[i], data=dataset)
    plt.show()





# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here

subset_train = dataset.iloc[:,:10]
data_corr = subset_train.corr()
sns.heatmap(data_corr, annot=True, fmt='.2f')
correlation = data_corr.unstack().sort_values(kind='quicksort')
corr_var_list = correlation[(correlation<lower_threshold) | (correlation>upper_threshold) & (correlation != 1)]
print(corr_var_list)

# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.head())
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.

#Standardized
#Apply transform only for continuous data       
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp = scaler.fit_transform(X_test.iloc[:,:10])

#Concatenate scaled continuous data and categorical

X_train1 = np.concatenate((X_train_temp,X_train.iloc[:,10:]), axis=1)
X_test1 = np.concatenate((X_test_temp,X_test.iloc[:,10:]), axis=1)

scaled_features_train_df = pd.DataFrame(X_train1, columns=X_train.columns, index=X_train.index)
scaled_features_test_df = pd.DataFrame(X_test1, columns=X_test.columns, index=X_test.index)




# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

# Write your solution here:

skb = SelectPercentile(score_func=f_classif , percentile=90)
predictors = skb.fit_transform(X_train1,y_train)
scores = skb.scores_
#print(scores)
Features = X_train.columns
dataframe= pd.DataFrame(data={'Features':Features,'Scores':scores})
dataframe.sort_values(['Scores'],ascending=False, inplace=True)
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])
print("Top 90 Percentile Features : \n")
print(top_k_predictors)




# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features = clf1.fit(X_train,y_train)
predictions_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(y_test,predictions_all_features)
print("One vs All Classifier Accuracy Score with all features : ",score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(y_test,predictions_top_features)
print("One vs All Classifier Accuracy Score with top features : ",score_top_features)




