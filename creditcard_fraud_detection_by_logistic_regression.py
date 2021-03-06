# -*- coding: utf-8 -*-
"""creditcard-fraud-detection-by-logistic-regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ggt5KPrCnWKiIzhFplH_c-WpVXiJsaZf
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter


pd.set_option('display.max_columns', None)
card =pd.read_csv(r'E:\FILES\VARIOUS_PYTHON_TASKS\CREDIT CARD FRAUD DETECTION\creditcard.csv')

card.head()

card.info()

card.describe()

card.shape

"""Finding :
Dataset has 284807 rows and 31 columns.

# DATA QUALITY CHECK

## Check for NULL/MISSING values
"""

# percentage of missing values in each column
round(100 * (card.isnull().sum()/len(card)),2).sort_values(ascending=False)

# percentage of missing values in each row
round(100 * (card.isnull().sum(axis=1)/len(card)),2).sort_values(ascending=False)

"""### Note:
- There are no missing / Null values either in columns or rows

## Duplicate Check
"""

card_d=card.copy()
card_d.drop_duplicates(subset=None, inplace=True)

card.shape

card_d.shape

"""### Note:
- Duplicate are found in the records
"""

## Assigning removed duplicate datase to original 
card=card_d
card.shape

del card_d

"""# EXPLORATORY DATA ANALYSIS"""

card.info()

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()  
    plt.show()
draw_histograms(card,card.columns,8,4)

card.Class.value_counts()

ax=sns.countplot(x='Class',data=card);
ax.set_yscale('log')

"""### Note

- There are 283253 records with no fraud status and 473 records with fraud status.

# Correlation Matrix
"""

plt.figure(figsize = (40,10))
sns.heatmap(card.corr(), annot = True, cmap="tab20c")
plt.show()

"""### Note
- The heatmap clearly shows which all variable are multicollinear in nature,
 and which variable have high collinearity with the target variable.
- We will refer this map back-and-forth while building the linear model so as
 to validate different correlated values along with p-value, for identifying
 the correct variable to select/eliminate from the model.

# Logistic Regression
"""

card.shape

card.info()

estimators=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X1 = card[estimators]
y = card['Class']

col=X1.columns[:-1]
col

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()

results_logit.summary()

"""### Note:

The results above show some of the attributes with P value higher than the 
preferred alpha(5%) and thereby showing low statistically significant relationship
 with the probability of heart disease. Backward elemination approach is used 
 
 here to remove those attributes with highest Pvalue one at a time follwed by
 running the regression repeatedly until all attributes have P Values less than 0.05.

## Feature Selection: Backward elemination (P-value approach)
"""

def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column 
    names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with
    all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.0001):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(X,card.Class,col)

result.summary()

"""Logistic regression equation

P=e^(β0+β1X1)/1+e^(β0+β1X1)

# Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues
"""

params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))

new_features=card[['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20','V21', 'V22', 'V23',
       'V25', 'V26', 'V27','Class']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,stratify=y)

##### SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)

# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_sample(x_train, y_train)


##### SMOTEENN Technique (OverSampling & Under sampling)
'''
Over-sampling using SMOTE and cleaning using ENN.

Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
'''
sme = SMOTEENN(sampling_strategy='minority',random_state=42)
X_res, y_res = sme.fit_resample(x_train, y_train)

# Distribution of Original dataset
print('Original dataset shape : {}'.format(Counter(y_train)))

# Distribution of SMOTE (Just to see how it distributes the labels we won't use these variables)
print('SMOTE Label Distribution: {}'.format(Counter(ysm_train)))

# Distribution of SMOTEENN (Just to see how it distributes the labels we won't use these variables)
print('SMOTEENN Label Distribution: {}'.format(Counter(y_res)))

#Distribution of data
'''
Original dataset shape : Counter({0: 198277, 1: 331})
SMOTE Label Distribution: Counter({0: 198277, 1: 198277})
SMOTEENN Label Distribution: Counter({1: 191429, 0: 189194})
'''

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

logreg_sm=LogisticRegression()
logreg_sm.fit(Xsm_train,ysm_train)
y_pred_sm=logreg_sm.predict(x_test)

logreg_sme=LogisticRegression()
logreg_sme.fit(X_res,y_res)
y_pred_sme=logreg_sme.predict(x_test)

"""# Model Evaluation

## Model accuracy
"""

from sklearn.metrics import classification_report
print("Classsification report normal:")
print(classification_report(y_test,y_pred))
print("Classsification report SMOTE:")
print(classification_report(y_test,y_pred_sm))
print("Classsification report SMOTEEN:")
print(classification_report(y_test,y_pred_sme))

from sklearn.metrics import accuracy_score
print("Accuracy score for normal:")
accuracy_score(y_test,y_pred)

print("Accuracy score for SMOTE:")
accuracy_score(y_test,y_pred_sm)

print("Accuracy score for SMOTEEN:")
accuracy_score(y_test,y_pred_sme)

"""* Accuracy of the model is 0.9982
But accuracy is not a good metric for measuring the performance of a model 
especially when the data is highly skewed or dataset is highly imbalanced as 
in this case. A better metric would be a high Recall and F1 score with high 
sensitivity along with high AUC score.
# Confusion matrix
"""

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

cm=confusion_matrix(y_test,y_pred_sm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

cm=confusion_matrix(y_test,y_pred_sme)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

"""The confusion matrix shows 84940+99 = 85039 correct predictions and 43+36= 79 incorrect ones.

True Positives: 99

True Negatives: 84940

False Positives: 36 (Type I error)

False Negatives: 43 ( Type II error)
"""

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

"""# Model Evaluation - Statistics"""

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) =       ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy =                  ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) =       ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) =       ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) =               ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) =               ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

"""- From the above statistics it is clear that the model is highly specific 
than sensitive. The negative values are predicted more accurately than the positives."""

y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of Not Fraud (0)','Prob of Fraud (1)'])
y_pred_prob_df.head()

"""## Predicted probabilities of 0 (No Fraud) and 1 ( Fraud) for the test data
 with a default classification threshold of 0.5

### Lower the threshold

Since the model is predicting Fraud, too many type II errors is not advisable. 
A False Negative ( ignoring the probability of Fraud when there actualy is one)
 is more dangerous than a False Positive in this case. Hence inorder to 
 increase the sensitivity, threshold can be lowered.
"""

from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes_sm=logreg_sm.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes_sm,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes_sme=logreg_sme.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes_sme,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

"""## ROC Curve"""

from sklearn.metrics import roc_curve
fig = plt.figure()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr, label='roc normal')
fpr_sm, tpr_sm, thresholds = roc_curve(y_test, y_pred_prob_yes_sm[:,1])
plt.plot(fpr_sm,tpr_sm, label='roc SMOTE')
fpr_sme, tpr_sme, thresholds = roc_curve(y_test, y_pred_prob_yes_sme[:,1])
plt.plot(fpr_sme,tpr_sme, label='roc SMOTEEN')
plt.legend(loc='lower right')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Fraud classifier Using various sampling techniques')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

"""A common way to visualize the trade-offs of different thresholds is by using 
an ROC curve, a plot of the true positive rate
 (# true positives/ total # positives) versus the false positive rate
  (# false positives / total # negatives) for all possible choices of thresholds.
   A model with good classification accuracy should have significantly more true
   positives than false positives at all thresholds.

The optimum position for roc curve is towards the top left corner where the 
specificity and sensitivity are at optimum levels

Area Under The Curve (AUC)
The area under the ROC curve quantifies model classification accuracy; the higher
 the area, the greater the disparity between true and false positives,
 and the stronger the model in classifying members of the training dataset.
 An area of 0.5 corresponds to a model that performs no better than random 
 classification and a good classifier stays as far away from that as possible. 
 An area of 1 is ideal. The closer the AUC to 1 the better
"""

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_prob_yes[:,1])
roc_auc_score(y_test,y_pred_prob_yes_sm[:,1])
roc_auc_score(y_test,y_pred_prob_yes_sme[:,1])

"""# Conclusion

- All attributes selected after the elimination process show Pvalues lower than
 5% and thereby suggesting significant role in the fraud Prediction.

- The best Area under the ROC curve is 0.973780 which is from SMOTE 
- We choose the threshold at 0.2 for SMOTE which gives the optimal 
  sensitivity and specificity.

- Overall model could be improved with more data.
"""

###############################################################
"""
                            MACHINE LEARNING PIPELINE

"""
###############################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from collections import Counter

pd.set_option('display.max_columns', None)
card =pd.read_csv(r'E:\FILES\VARIOUS_PYTHON_TASKS\CREDIT CARD FRAUD DETECTION\creditcard.csv')

card.info()

card.shape

"""Finding :
- Dataset has 284807 rows and 31 columns.
- There are no missing / Null values either in columns or rows

## Duplicate Check
"""
card_d=card.copy()
card_d.drop_duplicates(subset=None, inplace=True)
card.shape
card_d.shape

"""### Note:
- Duplicate are found in the records
"""
## Assigning removed duplicate datase to original 
card=card_d
card.shape
del card_d

# Logistic Regression
card.shape
card.info()

estimators=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X1 = card[estimators]
y = card['Class']

col=X1.columns[:-1]
col

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()

results_logit.summary()

"""### Note:

The results above show some of the attributes with P value higher than the 
preferred alpha(5%) and thereby showing low statistically significant relationship
 with the probability of heart disease. Backward elemination approach is used 
 
 here to remove those attributes with highest Pvalue one at a time follwed by
 running the regression repeatedly until all attributes have P Values less than 0.05.

## Feature Selection: Backward elemination (P-value approach)
"""
def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column 
    names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with
    all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.0001):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(X,card.Class,col)

result.summary()

new_features=card[['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20','V21', 'V22', 'V23',
       'V25', 'V26', 'V27','Class']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,stratify=y)

##### SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)

# This will be the data where we are going to train
Xsm_train, ysm_train = sm.fit_sample(x_train, y_train)

# Distribution of Original dataset
print('Original dataset shape : {}'.format(Counter(y_train)))

# Distribution of SMOTE (Just to see how it distributes the labels we won't use these variables)
print('SMOTE Label Distribution: {}'.format(Counter(ysm_train)))

#Distribution of data
'''
OOriginal dataset shape : Counter({0: 198277, 1: 331})
SMOTE Label Distribution: Counter({0: 198277, 1: 198277})

'''

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

logreg_sm=LogisticRegression()
logreg_sm.fit(Xsm_train,ysm_train)
y_pred_sm=logreg_sm.predict(x_test)

"""# Model Evaluation

## Model accuracy
"""

from sklearn.metrics import classification_report
print("Classsification report normal:")
print(classification_report(y_test,y_pred))
print("Classsification report SMOTE:")
print(classification_report(y_test,y_pred_sm))

from sklearn.metrics import accuracy_score
print("Accuracy score for normal:")
accuracy_score(y_test,y_pred)
#0.9990013863107686

print("Accuracy score for SMOTE:")
accuracy_score(y_test,y_pred_sm)
#0.9850090462651848
"""* Accuracy of the model is 0.9982
But accuracy is not a good metric for measuring the performance of a model 
especially when the data is highly skewed or dataset is highly imbalanced as 
in this case. A better metric would be a high Recall and F1 score with high 
sensitivity along with high AUC score.
# Confusion matrix
"""

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

cm=confusion_matrix(y_test,y_pred_sm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu");

TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

"""- From the above statistics it is clear that the model is highly specific 
than sensitive. The negative values are predicted more accurately than the positives."""

y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of Not Fraud (0)','Prob of Fraud (1)'])
y_pred_prob_df.head()


from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

from sklearn.preprocessing import binarize
for i in range(0,11):
    cm2=0
    y_pred_prob_yes_sm=logreg_sm.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes_sm,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

"""# Conclusion

- All attributes selected after the elimination process show Pvalues lower than
 5% and thereby suggesting significant role in the fraud Prediction.

- The best Area under the ROC curve is 0.973780 which is from SMOTE 
- We choose the threshold at 0.2 for SMOTE which gives the optimal 
  sensitivity and specificity.

- Overall model could be improved with more data.
"""

"""# Save XGBoost Classifier model using Pickel"""

## Pickle
import pickle
import os

# Set the working directory
os.chdir(r"E:\FILES\VARIOUS_PYTHON_TASKS\CREDIT CARD FRAUD DETECTION")
# save model
pickle.dump(logreg_sm, open('credit_card_fraud_detection.pickle', 'wb'))

# load model
model = pickle.load(open('credit_card_fraud_detection.pickle', 'rb'))

# predict the output
 
cm = 0
y_pred = model.predict((x_test.iloc[0]).to_numpy().reshape(1, -1))
y_pred
z=pd.DataFrame(y_pred)
z.shape
'''
 (85118, 1)
'''
z.value_counts()
'''
0    83740
1     1378
'''
'''
The following input gives fraud i.e. label is predicted as 1 using predict function
x_test.iloc[1170]
Out[136]: 
Time    118784.000000
V1           2.055322
V2           0.798811
V3          -2.173338
V4           3.593431
V5           1.438928
V6          -0.472701
V7           0.981887
V8          -0.322479
V9          -1.414681
V10          1.650586
V11         -1.746579
V12         -1.386654
V13         -1.725005
V14          1.097197
V15         -0.951814
V16          0.268918
V17         -0.563112
V20         -0.393930
V21          0.228582
V22          0.633130
V23         -0.136209
V25          0.695671
V26          0.351991
V27         -0.117525

The following input gives non-fraud i.e. label is predicted as 0 using predict function

x_test.iloc[0]
Out[141]: 
Time    132808.000000
V1           1.188351
V2          -1.334561
V3          -1.337815
V4           1.368390
V5          -0.110603
V6           0.461728
V7           0.270842
V8          -0.010393
V9           0.347166
V10          0.162762
V11          0.503131
V12          1.143950
V13          0.546509
V14          0.294444
V15         -0.376230
V16          0.370331
V17         -0.921944
V20          0.629989
V21          0.318764
V22          0.140389
V23         -0.213994
V25         -0.108040
V26         -0.725272
V27         -0.043551
'''

# confusion matrix
print('Confusion matrix of Logistic Regression model: \n',confusion_matrix(y_test, y_pred),'\n')
cm=confusion_matrix(y_test,y_pred)
print ('With',2/10,'threshold the Confusion Matrix is ','\n',cm,'\n',
        'with',cm[0,0]+cm[1,1],'correct predictions and',cm[1,0],'Type II errors( False Negatives)','\n\n',
      'Sensitivity: ',cm[1,1]/(float(cm[1,1]+cm2[1,0])),'\n','Specificity: ',cm[0,0]/(float(cm[0,0]+cm[0,1])),'\n\n\n')


from sklearn.preprocessing import binarize
cm2=0
y_pred_prob_yes_sm= model.predict_proba((x_test.iloc[0]).to_numpy().reshape(1, -1))

"""
Get the prediction and binarize it as per the threshold we want. Convert it to a 
dataframe
submit_smote = pd.DataFrame(y_pred2)
label the output as per submission format
"""
y_pred2=binarize(y_pred_prob_yes_sm,2/10)[:,1]
zz=pd.DataFrame(y_pred2)
zz.shape
'''
 (85118, 1)
'''
zz.value_counts()
'''
0.0    80558
1.0     4560
'''
'''
The following input gives fraud i.e. label is predicted as
 1 using predict_proba function
x_test.iloc[12]
Out[156]: 
Time    78007.000000
V1         -2.768853
V2         -3.520503
V3         -0.509607
V4         -1.941450
V5          4.110432
V6          1.855543
V7         -0.812920
V8          0.827134
V9         -1.048479
V10        -0.523567
V11        -0.279187
V12        -0.795883
V13         0.198563
V14        -1.576411
V15         0.265539
V16         0.149483
V17        -0.373117
V20         0.971757
V21         0.145182
V22        -0.495217
V23         0.996258
V25         0.151155
V26         0.268713
V27        -0.272149

The following input gives non-fraud i.e. label is predicted as
 0 using predict_proba function

x_test.iloc[0]
Out[160]: 
Time    132808.000000
V1           1.188351
V2          -1.334561
V3          -1.337815
V4           1.368390
V5          -0.110603
V6           0.461728
V7           0.270842
V8          -0.010393
V9           0.347166
V10          0.162762
V11          0.503131
V12          1.143950
V13          0.546509
V14          0.294444
V15         -0.376230
V16          0.370331
V17         -0.921944
V20          0.629989
V21          0.318764
V22          0.140389
V23         -0.213994
V25         -0.108040
V26         -0.725272
V27         -0.043551
'''
cm2=confusion_matrix(y_test,y_pred2)
print ('With',2/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
        'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
      'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'\n','Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')


# show the accuracy
print('Accuracy of Regression model = ',accuracy_score(y_test, y_pred))
print('Accuracy of Regression model = ',accuracy_score(y_test, y_pred2))

"""End ==================================================  """