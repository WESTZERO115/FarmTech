#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/8305-01/Desktop/aipSources/cabbage.csv",  encoding='cp949')
df.head()


# In[120]:


df = df.replace(0, np.NaN)   ##################
df.head()


# In[121]:


# nan 값이 얼마나 있는지 column별로 확인하기
df.isnull().sum()
#################################################################### [중, 하]가 압도적으로 NaN이 많아서 삭제. 이상치라 판단해 많아삭제


# In[122]:


df.isnull().sum() / len(df)


# In[6]:


import matplotlib
matplotlib.font_manager._rebuild()
x = ['특', '상', '중','하' ]
y = [204, 132, 408, 435]
plt.bar(x,y)

for i, v in enumerate(x):
    plt.text(v, y[i], y[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 9, 
             color='blue',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

plt.show()


############################# 여기 중, 하 는 너무 NaN 값이 많아 이상치라 판단해 삭제. ###############################


# In[38]:


SC = df ['봄배추 반입량']


# In[26]:


SC


# In[27]:


SC = SC.dropna()


# In[28]:


SC


# In[29]:


SC.isnull().sum()


# In[123]:


df = df.drop(['봄배추 10.0 kg 상자 /중', '봄배추 10.0 kg 상자 /하','여름배추 10.0 kg 상자 /중','여름배추 10.0 kg 상자 /하','김장(가을)배추 10.0 kg 상자 /중','김장(가을)배추 10.0 kg 상자 /하','월동배추 10.0 kg 상자 /중','월동배추 10.0 kg 상자 /하'],axis=1)


# In[49]:


df.isnull().sum()


# In[9]:


df


# In[124]:


df = df.drop(['고냉지배추 10.0 kg 상자 /중', '고냉지배추 10.0 kg 상자 /하','저장배추 10.0 kg 상자 /중','저장배추 10.0 kg 상자 /하'],axis=1)


# In[51]:


df


# In[46]:


################################################################ 여기 보류
df_thresh = df.dropna(axis = 1, thresh = 80)
df_thresh.info()


# In[125]:


df = df.drop([120])   ##### 마지막 행 삭제


# In[126]:


df


# In[127]:


df.isnull().sum()


# In[128]:


df = df.drop(['고냉지배추 10.0 kg 상자 /특', '고냉지배추 10.0 kg 상자 /상','저장배추 10.0 kg 상자 /특','저장배추 10.0 kg 상자 /상','고냉지배추 반입량','저장배추 반입량'],axis=1)


# In[55]:


df


# In[129]:


df.isnull().sum()


# In[130]:


df = df.dropna(subset=["김장(가을)배추 반입량"])


# In[131]:


df.isnull().sum()


# In[132]:


df = df.drop(['봄배추 10.0 kg 상자 /특', '여름배추 10.0 kg 상자 /특','김장(가을)배추 10.0 kg 상자 /특','월동배추 10.0 kg 상자 /상'],axis=1)


# In[59]:


df.isnull().sum()


# In[133]:


df


# In[134]:


df = df.dropna(subset=["여름배추 10.0 kg 상자 /상"])


# In[135]:


df.isnull().sum()


# In[29]:


df


# In[136]:


df = df.dropna(subset=["월동배추 10.0 kg 상자 /특"])


# In[63]:


df.isnull().sum()


# In[137]:


df


# In[138]:


df.isnull().sum()


# In[139]:


df


# In[140]:


df = df.fillna(0)


# In[141]:


df


# In[85]:


df = df.drop(['가격'],axis=1)


# In[142]:


df['가격'] = (df['봄배추 10.0 kg 상자 /상'] + df['여름배추 10.0 kg 상자 /상'] + df['김장(가을)배추 10.0 kg 상자 /상'] + df['월동배추 10.0 kg 상자 /특'])/4


# In[143]:


df


# In[144]:


df = df.drop(['봄배추 10.0 kg 상자 /상', '여름배추 10.0 kg 상자 /상','김장(가을)배추 10.0 kg 상자 /상','월동배추 10.0 kg 상자 /특'],axis=1)


# In[90]:


df


# In[145]:


df['가격'].describe()


# In[146]:


df['가격'].hist(bins=50)


# In[147]:


plt.boxplot(df['가격'])
plt.show()


# In[95]:


numerical_columns=['봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량']
fig = plt.figure(figsize = (16, 20))
ax = fig.gca()
df[numerical_columns].hist(ax=ax)
plt.show()


# In[96]:


cols = ['가격','봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량']

corr = df[cols].corr(method = 'pearson')
corr


# In[97]:


fig = plt.figure(figsize = (16, 12))
ax = fig.gca()
sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정 
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = cols, xticklabels = cols, ax=ax, cmap = "RdYlBu")
plt.tight_layout()
plt.show()


# In[98]:


sns.scatterplot(data=df, x='봄배추 반입량', y='가격', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()


# In[99]:


sns.scatterplot(data=df, x='여름배추 반입량', y='가격', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()


# In[100]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # 평균 0, 표준편차 1
scale_columns = ['봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량']
df[scale_columns] = scaler.fit_transform(df[scale_columns])


# In[101]:


df.head()


# In[102]:


df[numerical_columns].head()


# In[103]:


from sklearn.model_selection import train_test_split

# split dataset into training & test
X = df[numerical_columns]
y = df['가격']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[104]:


X_train.shape, y_train.shape


# In[105]:


X_test.shape, y_test.shape


# In[106]:


y_train


# In[107]:


X_train


# In[108]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.round(1) 
### 소수점 첫째자리까지 표시합니다. 즉, 소수점 둘째짜리에서 반올림 합니다.


# In[109]:


from sklearn import linear_model

# fit regression model in training set
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# predict in test set
pred_test = lr.predict(X_test)


# In[110]:


### print coef 
### 계수를 출력합니다.
print(lr.coef_)


# In[111]:


coefs = pd.DataFrame(zip(df[numerical_columns].columns, lr.coef_), columns = ['feature', 'coefficients'])
coefs


# In[112]:


coefs_new = coefs.reindex(coefs.coefficients.abs().sort_values(ascending=False).index)
coefs_new


# In[113]:


### coefficients 를 시각화 합니다. 

### figure size
plt.figure(figsize = (8, 8))

### bar plot : matplotlib.pyplot 모듈의 barh() 함수를 사용해서 수평 막대 그래프를 그릴 수 있습니다. 
plt.barh(coefs_new['feature'], coefs_new['coefficients'])
plt.title('"feature - coefficient" Graph')
plt.xlabel('coefficients')
plt.ylabel('features')
plt.show()


# In[114]:


import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)
### 회귀분석모형 수식을 간단하게 만들기 위해 다음과 같이 상수항을 독립변수 데이터에 추가하는 것을 상수항 결합(bias augmentation)작업이라고 합니다.
### ordinary least square 의 약자로, 거리의 최소값을 기준으로 구하는 함수입니다. 

model2 = sm.OLS(y_train, X_train2).fit()
model2.summary()


# In[115]:


### 예측 결과 시각화 (test set)
df = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df = df.sort_values(by='actual').reset_index(drop=True)
df.head()

### reset_index() : 아무래도 데이터프레임의 다양한 전처리 과정을 거치게 되면 인덱스가 뒤죽박죽인 경우가 많다. 이럴때 인덱스를 다시 처음부터 재배열 해주는 유용한 함수다.
### drop=True옵션을 주면 기존 인덱스를 버리고 재배열해준다.
### https://yganalyst.github.io/data_handling/Pd_2/


# In[116]:


plt.figure(figsize=(12, 9))
plt.scatter(df.index, df['prediction'], marker='x', color='r')
plt.scatter(df.index, df['actual'], alpha=0.3, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)
plt.show()


# In[117]:


### R square
print(model.score(X_train, y_train))  # training set
print(model.score(X_test, y_test))  # test set


# In[118]:


# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

# training set
pred_train = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, pred_train)))

# test set
print(sqrt(mean_squared_error(y_test, pred_test)))


# In[ ]:




