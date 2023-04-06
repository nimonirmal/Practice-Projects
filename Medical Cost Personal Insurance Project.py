#!/usr/bin/env python
# coding: utf-8

#   # Medical Cost Personal Insurance Project

# # Project Description
Health insurance is a type of insurance that covers medical expenses that arise due to an illness. These expenses could be related to hospitalisation costs, cost of medicines or doctor consultation fees. The main purpose of medical insurance is to receive the best medical care without any strain on your finances. Health insurance plans offer protection against high medical costs. It covers hospitalization expenses, day care procedures, domiciliary expenses, and ambulance charges, besides many others. Based on certain input features such as age , bmi,,no of dependents ,smoker ,region  medical insurance is calculated .
Columns                                            
•	age: age of primary beneficiary
•	sex: insurance contractor gender, female, male
•	bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
•	children: Number of children covered by health insurance / Number of dependents
•	smoker: Smoking
•	region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
•	charges: Individual medical costs billed by health insurance

Predict : Can you accurately predict insurance costs?

Dataset Link-
https://github.com/dsrscientist/dataset4
https://github.com/dsrscientist/dataset4/blob/main/medical_cost_insurance.csv

# In[118]:


#importing Essential libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# # Data collection

# In[119]:


df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv")
df.shape


# In[121]:


#Checking random records of dataset
df.sample(10)


# In[122]:


# info of dataset
df.info()

Here categorical features are :
1)Sex
2)region
3)smoker
# # Converting Cateogrical into Binary number

# In[123]:


# encoding sex column
df.replace({'sex':{'male' : 0,'female': 1}},inplace =True)

# encoding smoker column
df.replace({'smoker':{'yes' : 1 , 'no' : 0}},inplace = True)

# encoding region column
df.replace({'region':{'southeast':0, 'southwest': 1, 'northeast': 2,
                          'northwest': 3}}, inplace = True)


# In[124]:


df.describe()


# In[125]:


#Checking for data distribution

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20,15), facecolor = 'yellow')
plotnumber = 1 #this is an initiator

for column in df:
    if plotnumber <8: #here 7 is the number of features
        ax = plt.subplot (4,2, plotnumber)
        sns.distplot (df [column])
        plt.xlabel (column, fontsize =20)
    plotnumber +=1
plt.tight_layout()


# In[126]:


#Checking for outliers

plt.figure (figsize = (20,25))
graph = 1 #Initiator

for column in df:
    if graph <=6: 
        plt.subplot (4,3, graph)
        ax = sns.boxplot (data = df[column],)
        plt.xlabel (column, fontsize = 15)
    graph +=1
plt.show()


# In[127]:


# splitting features & labels

x=df.drop('charges', axis=1)
y= df.charges


# In[128]:


x


# In[129]:


y


# In[130]:


#visualizing relationship


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (20,15), facecolor = 'grey')
plotnumber = 1 #this is an initiator

for column in x:
    if plotnumber <7: #here 6 is the number of features
        ax = plt.subplot (4,2, plotnumber)
        sns.barplot (x [column],y)
        
        plt.xlabel (column, fontsize =20)
        plt.ylabel ('charges', fontsize =20)
    plotnumber +=1
plt.show()


# In[131]:


#checking the corelation using heatmaps (only absolute values)

plt.figure(figsize = (15,7))
sns.heatmap(df.corr().abs(), annot = True, linewidths=0.5, linecolor  = "black", fmt='.2f')

Conclusion -

1) "region" does not have much impact on medical cost.
2) "Smoker" spend a lot on medical cost.
3)"Charges" are not affected by Gender.
4)People with two children have more medical expenses.

# In[132]:


# dropping "region " column 

x  = x.drop(columns=['region'])
x


# # Using Standard Scaler

# In[133]:


# data scaling 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#y_scaled = scaler.fit_transform(y)
x_scaled
#y_scaled


# In[134]:


#train test split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

x_train,x_test,y_train,y_test = train_test_split (x_scaled, y, test_size = 0.2, random_state = 42)


# In[135]:


print(x_scaled.shape, x_train.shape, x_test.shape)


# In[136]:


print(y.shape, y_train.shape, y_test.shape)


# # Model Training

# In[137]:


# Loading linear regression model
lr = LinearRegression()
lr.fit(x_train,y_train)


# # Model Evaluation

# In[138]:


# prediction on training data
training_data_prediction =lr.predict(x_train)


# In[139]:


# R squared value
r2_train = metrics.r2_score(y_train,training_data_prediction)
print('R squared value: ',r2_train)


# In[140]:


# prediction on testing data
test_data_prediction = lr.predict(x_test)


# In[141]:


# R squared value
r2_test = metrics.r2_score(y_test,test_data_prediction)
print('R squared value: ',r2_test)


# # Building a Predictive System

# In[152]:


input_data = (33,1,36.29,3,0)

#changing input_data to numpy array
input_data_as_numpy_array = np.array(input_data)

#reshaping array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = lr.predict(input_data_reshaped)
print(prediction)
print('The insurance cost is USD ', prediction[0])


# # END

# In[ ]:




