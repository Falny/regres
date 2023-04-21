#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')

plt.style.use('ggplot')


# In[2]:


df = pd.read_csv(r'C:\Users\vinem\OneDrive\Рабочий стол\Concrete_Data_Yeh.csv')
df


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


plt.figure(figsize=(8, 8))
plt.boxplot(df)
plt.show()


# In[6]:


from scipy import stats
import numpy as np


# In[7]:


df.shape


# In[8]:


z = stats.zscore(df)
z.head()


# In[9]:


z_mask = (np.abs(z) < 2).all(axis=1)
df_1 = df[z_mask]
df_1.shape


# In[10]:


plt.figure(figsize=(8, 8))
plt.boxplot(df_1)
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
sns.heatmap(df_1.corr(), annot= True)
plt.show()


# In[12]:


import graphviz


# In[13]:


plt.figure(figsize=(15, 15))
sns.pairplot(df, kind = 'scatter', diag_kind = 'kde')
plt.show()


# In[14]:


plt.figure(figsize=(10, 10))
sns.kdeplot(df['csMPa'], fill=True, color = 'maroon')
sns.kdeplot(df['age'], fill=True, color = 'orangered')
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[16]:


X = df_1.drop('csMPa', axis = 1)
y = df_1['csMPa']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[18]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


lr = LinearRegression()
model_lr = lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)
print('MAE:', mean_absolute_error(y_pred, y_test).round(2))
print('MSE:', mean_squared_error(y_pred, y_test).round(2))
print('RMSE:', np.sqrt(mean_absolute_error(y_pred, y_test)).round(2))
print('R2_score:', r2_score(y_pred, y_test).round(2))
print('-'*20)
df_lr = pd.DataFrame({
    'Test':y_test,
    'Predict': y_pred
})
df_lr


# In[20]:


params = {
    'learning_rate':0.03,
    'n_estimators':200,
    'criterion':'mse',
    'min_samples_split':16,
    'min_samples_leaf':16
}

gbr = GradientBoostingRegressor(**params)
model_gbr = gbr.fit(X_train, y_train)
y_pred_gbr = model_gbr.predict(X_test)
print('MAE_gbr:', mean_absolute_error(y_pred_gbr, y_test).round(2))
print('MSE_gbr:', mean_squared_error(y_pred_gbr, y_test).round(2))
print('RMSE_gbr:', np.sqrt(mean_absolute_error(y_pred_gbr, y_test)).round(2))
print('R2_score_gbr:', r2_score(y_pred_gbr, y_test).round(2))
print('-'*20)
df_gbr = pd.DataFrame({
    'Test':y_test,
    'Predict': y_pred_gbr
})
df_gbr


# In[21]:


xgb = XGBRegressor()
model_xgb = xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
print('MAE_xgb:', mean_absolute_error(y_pred_xgb, y_test).round(2))
print('MSE_xgb:', mean_squared_error(y_pred_xgb, y_test).round(2))
print('RMSE_xgb:', np.sqrt(mean_absolute_error(y_pred_xgb, y_test)).round(2))
print('R2_score_xgb:', r2_score(y_pred_xgb, y_test).round(2))
print('-'*20)
df_xgb = pd.DataFrame({
    'Test':y_test,
    'Predict': y_pred_xgb
})
df_xgb


# In[23]:


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators = [('lr',lr), ('gbr',gbr), ('xgb',xgb)])
model = vr.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MAE:', mean_absolute_error(y_pred, y_test).round(2))
print('MSE:', mean_squared_error(y_pred, y_test).round(2))
print('RMSE:', np.sqrt(mean_absolute_error(y_pred, y_test)).round(2))
print('R2_score:', r2_score(y_pred, y_test).round(2))
print('-'*20)
df = pd.DataFrame({
    'Test':y_test,
    'Predict': y_pred
})
df


# In[24]:


print('Train set score: ', cross_val_score(xgb, X_train, y_train, cv = 4, scoring='r2').mean())
print('Test set score: ', cross_val_score(xgb, X_test, y_test, cv = 4, scoring='r2').mean())


# In[25]:


print('Train set score: ', cross_val_score(vr, X_train, y_train, cv = 4, scoring='r2').mean())
print('Test set score: ', cross_val_score(vr, X_test, y_test, cv = 4, scoring='r2').mean())


# In[26]:


print('Train set score: ', cross_val_score(gbr, X_train, y_train, cv = 4, scoring='r2').mean())
print('Test set score: ', cross_val_score(gbr, X_test, y_test, cv = 4, scoring='r2').mean())


# In[27]:


print('Train set score: ', cross_val_score(lr, X_train, y_train, cv = 4, scoring='r2').mean())
print('Test set score: ', cross_val_score(lr, X_test, y_test, cv = 4, scoring='r2').mean())


# In the end VotingRegressor showed the best result on the test data, 
# literally dropping 0.1 on the training data compared to the training data on XGBRegressor
