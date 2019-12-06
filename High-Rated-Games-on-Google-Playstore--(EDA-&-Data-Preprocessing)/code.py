# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

#Code starts here

data = pd.read_csv(path)
fig, (axes_1,axes_2) = plt.subplots(nrows = 2 , ncols=1)
data = data[data['Rating'] >= 0]
axes_1.hist(data['Rating'], bins = 30)

# Cleaning Rating<=5
data = data[data['Rating'] <= 5]
axes_2.hist(data['Rating'], bins = 30)

#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()
#print(total_null)
percent_null = (total_null / data.isnull().count())
#print(percent_null)
missing_data= pd.concat([total_null, percent_null], axis=1, keys=['Total','Percent'])
print(missing_data)

#Dropping null values
data.dropna(inplace = True)

total_null_1 = data.isnull().sum()
#print(total_null_1)
percent_null_1 = (total_null_1 / data.isnull().count())
#print(percent_null_1)
missing_data_1= pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total','Percent'])
print(missing_data_1)


# code ends here


# --------------
#Code starts here

g = sns.catplot(x='Category', y='Rating', data=data, kind='box', height=10)
g.set_titles("Rating vs Category [BoxPlot]")
g.set_xticklabels(rotation=90)

 #Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data.Installs.value_counts()
data['Installs'] = data['Installs'].str.replace(',', '')
data['Installs'] = data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].astype(int)
data.Installs.value_counts()

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x=data.Installs, y=data.Rating, data=data)
plt.title("Rating vs Installs [RegPlot]")
#Code ends here



# --------------
#Code starts here

print(data.Price.value_counts())
data['Price'] = data['Price'].str.replace('$', '')
data['Price'] = data['Price'].astype(float)
print(data.Price.value_counts())

sns.regplot(x=data['Price'], y=data['Rating'], data=data)
plt.title("Rating vs Price [RegPlot]")

#Code ends here


# --------------
#Code starts here

print("Original Data")
print(data['Genres'].unique())
print("_"*50)

data['Genres'] = data['Genres'].apply(lambda x: x.split(';')[0])
print("Data After Splitting")
print(data['Genres'].unique())

gr_mean = data.groupby(['Genres'], as_index= False)[['Rating']].mean()
print("_"*50)
print("Describe")
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(by = 'Rating')

print("_"*50)
print("Lowest Average Rating : ",gr_mean.iloc[0])
print("_"*50)
print("Highest Average Rating : " ,gr_mean.iloc[-1])

#Code ends here


# --------------
from datetime import datetime
#Code starts here

print(data['Last Updated'])
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
data['Last Updated Days']

sns.regplot(x= "Last Updated Days", y= "Rating", data=data)
plt.title("Rating vs Last Updated [RegPlot]")

#Code ends here


