# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:29:16 2018

@author: A 
"""
    

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_excel('GUL001_Appt_Data_fixed.xlsx').drop(['Unnamed: 22'], axis=1)
data = data[data['Age'] > 61]
data = data[data['Age'] < 109]
data['Age'].value_counts()
data['ApptStatusId'] = data['ApptStatusId'].map(lambda x: 1 if x in [6,15,3,4,17,14] else 0)
data = data.drop(['Id','ApptStartTime','ApptEndTime','ApptCreatedDate','ApptStatus','ApptTypeId','ApptFacilityId','ApptLocation','ApptProviderName','ApptProviderId','PatientId','AccountNumber','PatientZip'], axis=1)


"""%matplotlib inline
pd.crosstab(data.MaritalStatus,data.ApptStatusId).plot(kind='bar')
plt.title('Makes Appt for Marital Status')
plt.xlabel('MaritalStatus')
plt.ylabel('ApptStatusId')
plt.savefig('purchase_fre_job')"""

# get dummy vars 
cat_vars= ['ApptType','Gender','MaritalStatus','Language','Ethnicity']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    
cat_vars= ['ApptType','Gender','MaritalStatus','Language','Ethnicity']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

data_final_vars=data_final.columns.values.tolist()
y=['ApptStatusId']
X=[i for i in data_final_vars if i not in y]


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[np.asarray(y)] )
print(rfe.support_)
print(rfe.ranking_)


cols=['Age','TotalPatientCancelled','ApptType_1ST POST OP','ApptType_6 MONTHS',
      'ApptType_CANCEL','ApptType_NEW PATIENT','ApptType_POST OP','ApptType_SAME DAY POST OP',
      'ApptType_SURGERY SIGN UP','MaritalStatus_S','Language_eng','Ethnicity_HS'] 
X=data_final[cols]
y=data_final['ApptStatusId']

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
