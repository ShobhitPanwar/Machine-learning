#Importing the desired modules which will be used for EDA and model creation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

#Saving the Breast Cancer Wisconsin (Diagnostic) Data set , dataset.csv, to a dataframe.
breast_cancer_data = pd.read_csv('breast_cancer_dataset.csv')

## Exploratory Data Analysis

#Getting the information about the features of our dataset
breast_cancer_data.info()


'''
Diagnosis is the label for this dataset. <br>
"B" stands for benign. <br>
"M" stands for malignant.
'''

#Checking if there are any nulll values for features.
breast_cancer_data.isnull().sum()

# As diagnosis is of type object. Lets convert it into integer type.
# Lets map B to 0
# And Map M to 1
mapping_dict = {'B': 0, 'M':1}
breast_cancer_data['diagnosis'] = breast_cancer_data['diagnosis'].map(mapping_dict)
breast_cancer_data['diagnosis']

breast_cancer_data.describe()

#First lets drop the feature, 'id'. As it does not have any impact on the diagnosis result.
breast_cancer_data = breast_cancer_data.drop(labels='id', axis=1)
breast_cancer_data.columns

#Checking for the outliers in each feature, except for id and diagnosis feature.
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20,15))
axes = axes.flatten() #Converting 2D array to 1D so that we can iterate over it using single loop.

#Iterate over each column and create a boxplot
for i in range(1, len(breast_cancer_data.columns)):
    axes[i-1].boxplot(breast_cancer_data[breast_cancer_data.columns[i]])
    axes[i-1].set_title(breast_cancer_data.columns[i])

plt.tight_layout()
plt.show()

'''
As indicated above, it is observed that there exist outliers in each feature. However, the elimination of outlier data is infeasible due to the potential inclusion of crucial information regarding the classification of the tumor as benign or malignant. These outliers may potentially represent unique characteristics or distinguishing features of either type of tumor.
Hence, lets check for more details like what is the value of each features for the type of tumor. This can be made visually clear by plotting each feature against the type of tumor.
'''

#We need to check the number of benign and malignant entries in our dataset.
breast_cancer_data['diagnosis'].value_counts()

So we have 357 enteries of Begnin tumor and 212 enteries of Malignant.
Lets plot a graph between diagnosis and other features to get visual understanding about how the value of features get impacted based on the tumor.

#lets plot the graph between diagnosis and the rest of the 30 features
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(15,45))
axes = axes.flatten()
for i in range(1, len(breast_cancer_data.columns)):
    #fig, ax = plt.subplots()
    axes[i-1].scatter(breast_cancer_data['diagnosis'], breast_cancer_data[breast_cancer_data.columns[i]])
    axes[i-1].set_xlabel('diagnosis')
    axes[i-1].set_ylabel(breast_cancer_data.columns[i])
plt.show()

#Checking for duplicate enteries in the dataset
breast_cancer_data.duplicated().sum()

#Lets check for the values of each feature, and if there is requirement of normalization.
#Set maximum no. of rows to display
pd.set_option('display.max_columns', 31)
breast_cancer_data.head(10)

# Hypothesis Testing using two sample t-test
'''
**Null Hypothesis:** *There is **NO** significant difference between feature selected.* <br>
**Alternate Hypothesis:** *There **IS** significant difference between feature selected.*
'''
#We have selected 14 features for our model buidling, we will check if all 14 features should be choosed for model training based on Hypothesis testing.

# Creating two groups based on diagnosis type and radius_mean.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['radius_mean']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['radius_mean']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis. \nThere is a significant difference in the radius_mean of begnin and malignant. Hence, we can consider radius_mean feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the radius_mean of begnin and malignant.')

# Creating two groups based on diagnosis type and area_mean.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['area_mean']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['area_mean']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis. \nThere is a significant difference in the area_mean of begnin and malignant. Hence, we can consider area_mean feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the area_mean of begnin and malignant.')

# Creating two groups based on diagnosis type and compactness_mean.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['compactness_mean']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['compactness_mean']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis. \nThere is a significant difference in the compactness_mean of begnin and malignant. Hence, we can consider compactness_mean feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the compactness_mean of begnin and malignant.')

# Creating two groups based on diagnosis type and concavity_mean.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['concavity_mean']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['concavity_mean']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis. \nThere is a significant difference in the concavity_mean of begnin and malignant. Hence, we can consider concavity_mean feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the concavity_mean of begnin and malignant.')

# Creating two groups based on diagnosis type and points_mean.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['points_mean']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['points_mean']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the points_mean of begnin and malignant. Hence, we can consider points_mean feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the points_mean of begnin and malignant.')

# Creating two groups based on diagnosis type and radius_se.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['radius_se']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['radius_se']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the radius_se of begnin and malignant. Hence, we can consider radius_se feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the radius_se of begnin and malignant.')

# Creating two groups based on diagnosis type and perimeter_se.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['perimeter_se']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['perimeter_se']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the perimeter_se of begnin and malignant. Hence, we can consider perimeter_se feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the perimeter_se of begnin and malignant.')

# Creating two groups based on diagnosis type and area_se.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['area_se']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['area_se']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the area_se of begnin and malignant. Hence, we can consider area_se feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the area_se of begnin and malignant.')

# Creating two groups based on diagnosis type and radius_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['radius_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['radius_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the radius_worst of begnin and malignant. Hence, we can consider radius_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the radius_worst of begnin and malignant.')

# Creating two groups based on diagnosis type and perimeter_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['perimeter_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['perimeter_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the perimeter_worst of begnin and malignant. Hence, we can consider perimeter_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the perimeter_worst of begnin and malignant.')
# Creating two groups based on diagnosis type and area_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['area_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['area_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the area_worst of begnin and malignant. Hence, we can consider area_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the area_worst of begnin and malignant.')

# Creating two groups based on diagnosis type and compactness_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['compactness_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['compactness_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the compactness_worst of begnin and malignant. Hence, we can consider compactness_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the compactness_worst of begnin and malignant.')

# Creating two groups based on diagnosis type and points_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['points_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['points_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the points_worst of begnin and malignant. Hence, we can consider points_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the points_worst of begnin and malignant.')

# Creating two groups based on diagnosis type and symmetry_worst.
begnin = breast_cancer_data[breast_cancer_data['diagnosis']==0]['symmetry_worst']
malignant = breast_cancer_data[breast_cancer_data['diagnosis']==1]['symmetry_worst']

#Performing t-test
t_stat, p_val = ttest_ind(begnin, malignant)

print('t-statistic:', t_stat)
print('p-value:', p_val)

if p_val < 0.01:
    print('Alternative Hypothesis: \nThere is a significant difference in the symmetry_worst of begnin and malignant. Hence, we can consider symmetry_worst feature for our model training.')
else:
    print('Null Hypothesis. \n\t There is no significant difference in the symmetry_worst of begnin and malignant.')

## Data Cleaning
'''
From the above Hypothesis testing, it is clear that we can choose all the selected 14 features for our model building. <br>
Lets drop the other features from our data frame.
'''
breast_cancer_data = breast_cancer_data.drop(labels=['texture_mean', 'perimeter_mean', 'smoothness_mean', 'symmetry_mean', 'dimension_mean', 'texture_se', 'smoothness_se','compactness_se', 'concavity_se', 'points_se', 'symmetry_se', 'dimension_se', 'texture_worst', 'smoothness_worst', 'concavity_worst', 'dimension_worst'], axis=1)
breast_cancer_data.columns

#Lets check for no. of rows and columns
breast_cancer_data.shape

breast_cancer_data['diagnosis'].value_counts()

#As from the above box plot plot we saw that each feature contains some outliers, lets remove the outliers using Z-score.

# Calculating Z-score for each column
z_scores = np.abs(stats.zscore(breast_cancer_data))

# Filter out the rows that contain outliers in any column
# Setting a threshold of 3.
breast_cancer_data = breast_cancer_data[(z_scores < 3).all(axis=1)]

breast_cancer_data.shape

breast_cancer_data.isnull().sum()

## Normalizing

breast_cancer_data.head(5)

#From above we can see that there is difference in magnitude across the features. So we need to normalize the features and bring them in same scale so that some features do not dominate over the other features.

#Normalizing all the rows using Min-Max normalization technique. Using this all the features will be in same magnitude range.

#For using the Min-Max normalization technique, we are using MinMaxScaler class from the sklearn.preprocessing module.
scaler = MinMaxScaler()

#normalizing the data stored in the breast_cancer_data dataframe, but we dont want to normalize our label, i.e., diagnosis.
df_to_normalize = breast_cancer_data.drop('diagnosis', axis=1)
print(df_to_normalize.shape)
normalized_data = scaler.fit_transform(df_to_normalize)

#Converting normalized_data to dataframe.
normalized_breast_cancer_data = pd.DataFrame(normalized_data, columns=df_to_normalize.columns)
normalized_breast_cancer_data = normalized_breast_cancer_data.reset_index(drop=True) # reset indices
breast_cancer_data = breast_cancer_data.reset_index(drop=True) # reset indices

#Adding label i.e., diagnosis, back to our dataframe
normalized_breast_cancer_data['diagnosis'] = breast_cancer_data['diagnosis']

## Model Building

#I am using **Random Forest** algorithm to build model. Thereafter, checking its performance on the basis of accuracy.

# separating the features and label
X = normalized_breast_cancer_data.drop(['diagnosis'], axis=1)
y = normalized_breast_cancer_data['diagnosis']

# spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

#**Random Forest Classifier**
rf_classifier = RandomForestClassifier()

# Training the model using the training set
rf_classifier.fit(X_train, y_train)

# Making predictions on the testing set
rf_classifier_pred = rf_classifier.predict(X_test)

# Evaluating the accuracy of the model
rf_accuracy = accuracy_score(y_test, rf_classifier_pred)

#**KNearestNeighbor(KNN) Classifier**
knn_classifier = KNeighborsClassifier()

# Training the model using the training set
knn_classifier.fit(X_train, y_train)

# Making predictions on the testing set
knn_classifier_pred = knn_classifier.predict(X_test)

# Evaluating the accuracy of the model
knn_accuracy = accuracy_score(y_test, knn_classifier_pred)

#**Performance of the model**

print("Random Forest Accuracy:", rf_accuracy)
print("KNN Accuracy:", knn_accuracy)

'''
**Summary:** <br>
Though random forest model is performing very well with an accuracy of 95% but KNN is performming better than RF model with an accuracy of 97%.
'''

