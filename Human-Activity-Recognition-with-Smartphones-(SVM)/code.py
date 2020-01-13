# --------------
import pandas as pd
from collections import Counter

# Load dataset
data = pd.read_csv(path)

print(data.isnull().sum())

print('Statistical Description : \n', data.describe())


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data['Activity']

# plot the countplot
sns.countplot(x=label)
plt.title("Distribution of Target Variable")
plt.xticks(rotation=90)
plt.show()



# --------------
# make the copy of dataset

data_copy = data.copy()

# Create an empty column 
data_copy['duration'] = ""
label.head()
# Calculate the duration

duration_df = (data_copy.groupby([label[(label=='WALKING_UPSTAIRS') | (label=='WALKING_DOWNSTAIRS')], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
plot_data=duration_df.reset_index()

plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS': 'Upstairs', 'WALKING_DOWNSTAIRS': 'Downstairs'})

# Plot the barplot
plt.figure(figsize=(10,7))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title("Subject Vs Duration")
plt.show()






# --------------
import numpy as np

#exclude the Activity column and the subject column

feature_cols = data.drop(['Activity', 'subject'],1).columns
feature_cols

#Calculate the correlation values
correlated_values = data[feature_cols].corr()

#stack the data and convert to a dataframe
correlated_values =  correlated_values.stack().to_frame().reset_index().rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'})
# correlated_values = pd.DataFrame(correlated_values.stack().sort_values(kind='quicksort')).reset_index()
# correlated_values.rename(columns={'level_0':'Feature_1', 'level_1':'Feature_2', 0:'Correlation_score'}, inplace=True)

#create an abs_correlation column

correlated_values['abs_correlation'] = abs(correlated_values['Correlation_score'])

#Picking most correlated features without having self correlated pairs

top_corr_fields = correlated_values[correlated_values['abs_correlation']>=0.8]
top_corr_fields = top_corr_fields[top_corr_fields['Feature_1'] != top_corr_fields['Feature_2']].reset_index(drop=True)
top_corr_fields




# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
data.Activity = le.fit_transform(data.Activity)

# split the dataset into train and test
X = data.drop(['Activity'],1)
y = data['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train,y_train)
y_pred = clf.predict(X_test)

precision = error_metric(y_test, y_pred, average='weighted')[0]
recall = error_metric(y_test, y_pred, average='weighted')[1]
f_score = error_metric(y_test, y_pred, average='weighted')[2]

model1_score = accuracy_score(y_test, y_pred)

print("Accuracy SVC : ", model1_score)
print("Precision SVC : ", precision)
print("Recall SVC : ", recall)
print("F1 Score SVC : ", f_score)



# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score

# Feature selection using Linear SVC
lsvc = LinearSVC(C=0.01, penalty='l1', dual=False, random_state=42)
lsvc.fit(X_train,y_train)

model_2 = SelectFromModel(estimator=lsvc, prefit=True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)

# model building on reduced set of features
classifier_2 = SVC()
clf_2 = classifier_2.fit(new_train_features, y_train)
y_pred_new = clf_2.predict(new_test_features)

model2_score = accuracy_score(y_test, y_pred_new)

precision = error_metric(y_test, y_pred_new, average='weighted')[0]
recall = error_metric(y_test, y_pred_new, average='weighted')[1]
f_score = error_metric(y_test, y_pred_new, average='weighted')[2]

print("Accuracy using LinearSVC Feature Selection : ", model2_score)
print("Precision using LinearSVC Feature Selection : ", precision)
print("Recall using LinearSVC Feature Selection : ", recall)
print("F1 Score using LinearSVC Feature Selection : ", f_score)





# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = { 'kernel': ['linear','rbf'],
            'C': [100,20,1,0.1] }

# Usage of grid search to select the best hyperparmeters

selector = GridSearchCV(SVC(), scoring='accuracy', param_grid =parameters)
selector.fit(new_train_features, y_train)
params = selector.best_params_
print("Best Parameters set: ", params)

print('Detailed grid scores:')
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, selector.cv_results_['params']):
    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    print()

# Model building after Hyperparameter tuning

classifier_3 = SVC(C=20, kernel='rbf')
clf_3 = classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)

model3_score = accuracy_score(y_test, y_pred_final)

precision = error_metric(y_test, y_pred_final, average='weighted')[0]
recall = error_metric(y_test, y_pred_final, average='weighted')[1]
f_score = error_metric(y_test, y_pred_final, average='weighted')[2]

print("Accuracy after GridSearchCV : ", model3_score)
print("Precision after GridSearchCV : ", precision)
print("Recall after GridSearchCV : ", recall)
print("F1 Score after GridSearchCV : ", f_score)




