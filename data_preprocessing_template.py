# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('GUL001_Appt_Data_fixed.csv')

# Splits Appt Date and Start tIime into 2 separate columns. 
data2 = dataset['ApptStartTime'].apply(lambda x: pd.Series(x.split(' ')))
data2.rename(columns={0:'ApptDate',1:'StartTime'},inplace=True)

# Splits appt end time. We only need endTime and can get rid of Date
data3 = dataset['ApptEndTime'].apply(lambda x: pd.Series(x.split(' ')))
data3.rename(columns={0:'ApptEndDate', 1:'EndTime'}, inplace=True)


print(dataset.describe())

X = dataset.iloc[:, :-1].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""