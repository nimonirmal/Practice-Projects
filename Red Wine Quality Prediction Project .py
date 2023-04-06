#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Prediction Project

# # Project Description
Project Description
The dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

This dataset can be viewed as classification task. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
Attribute Information
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)

What might be an interesting thing to do, is to set an arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1' and the remainder as 'not good/0'.
This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at the ROC curve and the AUC value.

You need to build a classification model. 
Inspiration
Use machine learning to determine which physiochemical properties make a wine 'good'!

Dataset Link-
https://github.com/dsrscientist/DSData/blob/master/winequality-red.csv

# # Attribute Information
Input variables (based on physicochemical tests):

1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol

Output variable (based on sensory data):

12 - quality (score between 0 and 10)
# In[3]:


#Importing necessary libraries.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# # Data collection

# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df.shape


# In[7]:


#Checking random records of dataset

df.sample(5)


# In[8]:


# show top five records

df.head()


# In[9]:


#checking the unique label values

df.quality.unique()

As we can see, the quality ranges from 03 to 08 .

As suggested in the problem discription, converting the labels into only two variables. 
Reference text (What might be an interesting thing to do, is to set an arbitrary cutoff 
for your dependent variable (wine quality) at e.g. 7 or higher getting classified as 'good/1'
and the remainder as 'not good/0'.)
# In[10]:


#  For " good /1" label variable

df.loc[df.quality >= 7, 'quality'] = 1


# In[12]:


#checking the unique label values

df.quality.unique()


# In[13]:


#  For " not good /0" label variable

df.loc[df.quality > 1, 'quality'] = 0


# In[14]:


#checking the unique label values

df.quality.unique()


# In[16]:


#number of "good" and "not good "quality of wine

df.quality.value_counts()

There is an imbalance in the data w.r.t label.
# In[17]:


df.describe().T #since no.of columns are more, therefore using the 'transpose' method

1-No null values confirmed.

2-Skewness observed in total sulfur dioxide, free sulfur dioxide
possibility of skewness.
# In[18]:


#checking the corelation using heatmaps (only absolute values)

plt.figure(figsize = (15,7))
sns.heatmap(df.corr().abs(), annot = True, linewidths=0.5, linecolor  = "black", fmt='.2f')

1-residual sugar, free sufur dioxide and ph have very low corelation with the label.

2-alcohol, volatile acidity and citric acidicty have high corelation with label.
# In[19]:


#checking the correlations with label in ascending order
df_corr = df.corr().abs()['quality'].sort_values() 
df_corr

Alcohol, volatile acidity, citric acid have great correlations with label.
# In[20]:


#graphichal representation of correlation with label
plt.figure (figsize = (15,12))
df_corr.plot().bar 


# In[21]:


#Checking for data distribution

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (20,15), facecolor = 'yellow')
plotnumber = 1 #this is an initiator

for column in df:
    if plotnumber <12: #here 11 is the number of features
        ax = plt.subplot (3,4, plotnumber)
        sns.distplot (df [column])
        plt.xlabel (column, fontsize =20)
    plotnumber +=1
plt.show()

Most of the features have positive skewness
Some features are multimodal.
# In[22]:


#Checking for outliers

plt.figure (figsize = (20,25))
graph = 1 #Initiator

for column in df:
    if graph <=12: 
        plt.subplot (4,3, graph)
        ax = sns.boxplot (data = df[column],)
        plt.xlabel (column, fontsize = 15)
    graph +=1
plt.show()


# In[23]:


df.info()


# In[24]:


from scipy.stats import zscore

z_score = zscore (df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
                      'total sulfur dioxide','density','pH','sulphates','alcohol']])
abs_z_score = np.abs(z_score)
filtering_entry = (abs_z_score < 3).all(axis=1)
df = df[filtering_entry]
df


# In[25]:


#Checking for Skewness
df.skew().sort_values()


# In[26]:


df.columns


# In[27]:


#splitting features & labels

x=df.drop('quality', axis=1)
y= df.quality


# In[28]:


x.head()


# In[29]:


y.head()


# In[30]:


#Using Quantile transformer for skewness removal

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer()
np_array = qt.fit_transform(x) #this will result in numpy array
np_array


# In[31]:


x.columns


# In[32]:


#converting array into dataframe
xt = pd.DataFrame(np_array, columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'])


# In[33]:



xt.head()


# In[34]:


plt.figure(figsize = (20,15), facecolor = 'yellow')
plotnumber = 1 #this is an initiator

for column in xt:
    if plotnumber <12: #here 11 is the number of features
        ax = plt.subplot (3,4, plotnumber)
        sns.distplot (xt [column])
        plt.xlabel (column, fontsize =20)
    plotnumber +=1
plt.tight_layout()


# In[30]:


sns.countplot(y)


# # Train -Test split

# In[35]:



from sklearn.model_selection import train_test_split
# Split the data into training and testing sets

x_train,x_test,y_train,y_test = train_test_split (xt, y, test_size = 0.2, random_state = 42)


# In[36]:


print(xt.shape, x_train.shape, x_test.shape)


# In[37]:


print(y.shape, y_train.shape, y_test.shape)


# # Model Training

# In[38]:


from sklearn.linear_model import LogisticRegression
# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = logreg.predict(x_test)


# # Model Evaluation

# In[39]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[40]:


x_train, x_test, y_train, y_test = train_test_split (xt, y, test_size = 0.2, random_state = 11)


# In[41]:


import six
import joblib
import sys
sys.modules ['sklearn.externals.six'] = six
sys.modules ['sklearn.externals.joblib'] = joblib


# In[42]:


get_ipython().system('pip install imbalanced-learn')


# In[43]:


from imblearn.over_sampling import SMOTE


# In[44]:


ovr_spl = SMOTE()
x_train_ns, y_train_ns = ovr_spl.fit_resample (x_train, y_train)

import warnings
warnings.filterwarnings('ignore')


# # XGBClassifier

# In[45]:


pip install xgboost


# In[46]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(x_train_ns, y_train_ns)


# In[47]:


y_pred = model.predict(x_test)

y_pred 


# In[48]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# #  KNNClassifier

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()


# In[50]:


knn.fit(x_train_ns, y_train_ns)
y_pred = knn.predict (x_test)


# In[51]:


y_pred


# In[52]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# #  Decision Tree Classifier

# In[53]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train_ns, y_train_ns)
y_pred = dtc.predict (x_test)


# In[54]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# # SVM

# In[55]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_ns, y_train_ns)
y_pred = svc.predict (x_test)


# In[56]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# # Adaboost

# In[57]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()


# In[58]:


ada.fit(x_train_ns, y_train_ns)
y_pred = ada.predict (x_test)


# In[59]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# # Here we see that XGBClassifier works the best in this dataset with accuracy 89%.

# # AUC- RUC Curve

# In[60]:


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import plot_roc_curve


# In[61]:


#For Training Data
disp = plot_roc_curve (model, x_train_ns, y_train_ns)
plot_roc_curve (knn, x_train_ns, y_train_ns, ax = disp.ax_)
plot_roc_curve (dtc, x_train_ns, y_train_ns, ax = disp.ax_)
plot_roc_curve (svc, x_train_ns, y_train_ns, ax = disp.ax_)
plot_roc_curve (ada, x_train_ns, y_train_ns, ax = disp.ax_)
plt.legend (prop = {'size' : 10}, loc = 'lower right')
plt.show()


# In[62]:


#For Tetsing Data
disp = plot_roc_curve (model, x_test, y_test)
plot_roc_curve (knn, x_test, y_test, ax = disp.ax_)
plot_roc_curve (dtc, x_test, y_test, ax = disp.ax_)
plot_roc_curve (svc, x_test, y_test, ax = disp.ax_)
plot_roc_curve (ada, x_test, y_test, ax = disp.ax_)
plt.legend (prop = {'size' : 10}, loc = 'lower right')
plt.show()

Clearly XGB Classifier is the best in this dataset
# # Hyper Parameter Tuning

# In[63]:


from sklearn.model_selection import GridSearchCV, KFold


# In[66]:


params = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01,0.05,0.1],
    'booster': ['gbtree', 'gblinear'],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.5, 1, 5],
    'base_score': [0.2, 0.5, 1]
}


# In[65]:



gs2 = GridSearchCV(XGBClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=3))
gs2.fit(x_train_ns, y_train_ns)

print('Best score:', gs2.best_score_)
print('Best score:', gs2.best_params_)


# In[67]:


xgb = XGBClassifier(n_estimator = 500, base_score = 0.5,learning_rate = 0.1, reg_aplha = 0, reg_lamdba = 0.5)
xgb.fit(x_train_ns, y_train_ns)
y_pred = xgb.predict (x_test)


# In[69]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# # The Accuracy is Same as before.

# # END

# In[ ]:




