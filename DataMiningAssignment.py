# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:54:06 2021

@author: Rutger
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


#################### Q2.6 ########################
chol1 = pd.read_spss('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/Data Mining/voorbeeld7_1.sav')

#### a)
plot1 = sns.lmplot(x='leeftijd',y='chol',data=chol1,fit_reg=True)

#### b)
model = smf.ols(formula='chol ~ leeftijd',data=chol1) 
fit1 = model.fit()
#print(fit1.summary())

### #c)
#string = f'{chol_dummies.columns[2]}' '+' f'{chol_dummies.columns[3]}' '+' f'{chol_dummies.columns[4]}' '+' f'{chol_dummies.columns[5]}' '+' f'{chol_dummies.columns[6]}' '+' f'{chol_dummies.columns[7]}' '+' f'{chol_dummies.columns[8]}' '+' f'{chol_dummies.columns[9]}' '+' f'{chol_dummies.columns[10]}' '+' f'{chol_dummies.columns[11]}' '+' f'{chol_dummies.columns[12]}' '+' f'{chol_dummies.columns[13]}'
chol_dummies = pd.get_dummies(chol1, columns=['actief','roken','sekse','alcohol'],prefix_sep='_')
chol_dummies.columns=['id', 'chol', 'leeftijd', 'bmi', 'heel_actief', 'inactief', 'matig_actief', 'niet_roker', 'roker', 'man', 'vrouw', 'matige_drinkers', 'niet_drinkers', 'zware_drinkers']
model2 = smf.ols(formula='chol ~ leeftijd+bmi+heel_actief+inactief+matig_actief+niet_roker+roker+man+vrouw+matige_drinkers+niet_drinkers+zware_drinkers',data=chol_dummies) 
fit2 = model2.fit()
#print(fit2.summary())
chol1['residuals'] = fit2.resid

fig, ax = plt.subplots()
plt.hist(chol1['residuals'],bins=12)
plt.title('Histogram of residuals regarding linear regression fit')
plt.xlabel('Residual value')
plt.ylabel('Number of occurences')


#################### Q2.7 ########################
births_check = pd.read_csv('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/Data Mining/births.csv',delimiter=',')
births = pd.read_csv('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/Data Mining/births.csv',delimiter=',')

#### a) + b) + c)

# Change column names and values:
for x in range(0,len(births)):
    
    if 'first line child birth, at home' in births.iloc[x,births.columns.get_loc('child_birth')]:
        births.loc[x,'child_birth'] = 'at_home'
    else:
        births.loc[x,'child_birth'] = 'not_at_home'
    
    if births.loc[x,'parity'] == 1:
        births.loc[x,'parity'] = 'primi'
    else:
        births.loc[x,'parity'] = 'multi'
    
    if births.loc[x,'etnicity'] != 'Dutch':
        births.loc[x,'etnicity'] = 'Not Dutch'
    
    if x == len(births)-1:
        births = births.rename(columns={'child_birth':'home','parity':'pari','etnicity':'etni'})      #rename columns

#### d) (based on: https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python)
# Make dataframe suitable for analysis by changing values
births_reg = births.copy()
for x in range(0,len(births_reg)):
    if births_reg.loc[x,'pari'] == 'primi':
        births_reg.loc[x,'pari'] = 0
    else:
        births_reg.loc[x,'pari'] = 1
    
    if births_reg.loc[x,'etni'] == 'Dutch':
        births_reg.loc[x,'etni'] = 0
    else:
        births_reg.loc[x,'etni'] = 1

# make dummies of columns with more than 2 possible values
births_reg = pd.get_dummies(births_reg, columns=['urban','age_cat'])

# Split dataset in features and the target variable
# Get demanded columns, and split between depedent variable y and independent variables X
columns_to_use = []
for x in births_reg.columns:
    if x == 'provmin' or x == 'age' or x == 'home':
        columns_to_use.append(False)
    else:
        columns_to_use.append(True)
X = births_reg.loc[:, columns_to_use]                       # exclude home column
y = births_reg['home']                                      # dependent variable at_home or not_at_home
logreg = LogisticRegression()                               # instantiate the model with default parameters
logreg.fit(X,y)                                             # fit the model with the data
y_pred =logreg.predict(X)                                   # predict at_home / not_at_home 
clas_rep = metrics.classification_report(y,y_pred)          # classification report
print(clas_rep)

#### e)
# decision tree model:
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
prediction = clf.predict(X)                                 # prediction
probs = clf.predict_proba(X)                                # probabilities
plot = tree.plot_tree(clf,max_depth=1)

#### f)
# performing cross validation on Logistic regression analysis: 
outcome_lreg = cross_val_score(logreg, X, y, cv=4)
# performing a accuracy test on test vs. training set for Logistic regression analysis, with confusing matrix:
# Split X and y into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

logreg2 = LogisticRegression()                              # instantiate the model with default parameters
logreg2.fit(X_train,y_train)                                # fit the model with the data
y_pred=logreg2.predict(X_test)                              # predict at_home / not_at_home 
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)       # make confusion matrix

class_names=['at home','not at home']                       # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap, matrix plot:
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for linear regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# performing cross validation on decision tree: 
outcome_tree = cross_val_score(clf, X, y, cv=4)
# performing a accuracy test on test vs. training set for Decision tree analysis, with confusing matrix:
# Split X and y into training and testing sets
X_train_tree,X_test_tree,y_train_tree,y_test_tree = train_test_split(X,y,test_size=0.25,random_state=0)

clf2 = tree.DecisionTreeClassifier()                            # instantiate the model with default parameters
clf2.fit(X_train_tree,y_train_tree)                                # fit the model with the data
y_pred_tree=logreg2.predict(X_test_tree)                              # predict at_home / not_at_home 
cnf_matrix_tree = metrics.confusion_matrix(y_test_tree, y_pred_tree)       # make confusion matrix

class_names=['at home','not at home']                       # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap, matrix plot:
sns.heatmap(pd.DataFrame(cnf_matrix_tree), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for decision tree', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')