# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('GUL001_Appt_Data_fixed.csv')

# Splits Appt Date and Start time into 2 separate columns. 
data2 = dataset['ApptStartTime'].apply(lambda x: pd.Series(x.split(' ')))
data2.rename(columns={0:'ApptDate',1:'StartTime'},inplace=True)

# Splits appt end time. We only need endTime and can get rid of Date
data3 = dataset['ApptEndTime'].apply(lambda x: pd.Series(x.split(' ')))
data3.rename(columns={0:'ApptEndDate', 1:'EndTime'}, inplace=True)

#drop unecessary columns
ds = dataset.drop(['Id','ApptStartTime','ApptEndTime','ApptCreatedDate','ApptStatus','ApptType','ApptLocation','ApptProviderName','PatientId','AccountNumber','PatientZip'], axis=1)
# drop unnamed column and dependent var
cols_of_interest = [1,2,3,4,5,6,7,8,9,10]
X = ds.iloc[:,cols_of_interest].values

#set dependent var for appt Status
y = ds.iloc[:, 1].values
# set appt status to either 0(no show) or 1(completed) for simplicity
for i in range(len(y)):
    if y[i] == 6 or y[i] == 10:
        y[i] = 1
    else:
        y[i] = 0    

#take care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 6] = labelEncoder_X.fit_transform(X[:, 6]) # Gender
X[:, 7] = labelEncoder_X.fit_transform(X[:, 7]) # Marital Status
X[:, 8] = labelEncoder_X.fit_transform(X[:, 8]) # Language
X[:, 9] = labelEncoder_X.fit_transform(X[:, 9]) # Ethnicity
oneHotEncoder = OneHotEncoder(categorical_features = [6,7,8,9])
X = oneHotEncoder.fit_transform(X).toarray()
y = labelEncoder_X.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)