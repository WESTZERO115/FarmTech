#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/8305-01/Desktop/aipSources/cabbage.csv",  encoding='cp949')
df.head()


# In[88]:


df = df.replace(0, np.NaN)   ##################
df.head()


# In[89]:


df2 = pd.read_csv("C:/Users/8305-01/Desktop/aipSources/cabbage_weather.csv",  encoding='cp949')
df2.head()


# In[90]:


df_last = pd.concat([df,df2],axis=1)
df_last


# In[91]:


df_last = df_last.drop(['고냉지배추 10.0 kg 상자 /중', '고냉지배추 10.0 kg 상자 /하','저장배추 10.0 kg 상자 /중','저장배추 10.0 kg 상자 /하'],axis=1)


# In[92]:


df_last


# In[93]:


df_last = df_last.drop([130])   ##### 마지막 행 삭제


# In[95]:


df_last = df_last.drop(['고냉지배추 10.0 kg 상자 /특', '고냉지배추 10.0 kg 상자 /상','저장배추 10.0 kg 상자 /특','저장배추 10.0 kg 상자 /상','고냉지배추 반입량','저장배추 반입량'],axis=1)


# In[97]:


df_last.isnull().sum()


# In[98]:


df_last = df_last.drop(['봄배추 10.0 kg 상자 /중', '봄배추 10.0 kg 상자 /하','여름배추 10.0 kg 상자 /중','여름배추 10.0 kg 상자 /하','김장(가을)배추 10.0 kg 상자 /중','김장(가을)배추 10.0 kg 상자 /하','월동배추 10.0 kg 상자 /중','월동배추 10.0 kg 상자 /하'],axis=1)


# In[99]:


df_last.isnull().sum()


# In[100]:


df_last = df_last.drop(['봄배추 10.0 kg 상자 /특', '여름배추 10.0 kg 상자 /특','김장(가을)배추 10.0 kg 상자 /특','월동배추 10.0 kg 상자 /상'],axis=1)


# In[101]:


df_last.isnull().sum()


# In[103]:


df_last = df_last.dropna(subset=["봄배추 반입량","여름배추 반입량","김장(가을)배추 반입량","월동배추 반입량"])


# In[104]:


df_last.isnull().sum()


# In[106]:


df_last = df_last.dropna(subset=["여름배추 10.0 kg 상자 /상","월동배추 10.0 kg 상자 /특"])


# In[107]:


df_last.isnull().sum()


# In[108]:


df_last = df_last.drop(['일자'],axis=1)


# In[109]:


df_last.isnull().sum()


# In[112]:


df_last = df_last.fillna(0)
df_last.isnull().sum()


# In[114]:


df_last['가격'] = (df_last['봄배추 10.0 kg 상자 /상'] + df_last['여름배추 10.0 kg 상자 /상'] + df_last['김장(가을)배추 10.0 kg 상자 /상'] + df_last['월동배추 10.0 kg 상자 /특'])/4


# In[115]:


df_last


# In[116]:


df_last = df_last.drop(['봄배추 10.0 kg 상자 /상', '여름배추 10.0 kg 상자 /상','김장(가을)배추 10.0 kg 상자 /상','월동배추 10.0 kg 상자 /특'],axis=1)


# In[117]:


df_last


# In[ ]:


## ------------------------------------------------------------------- 전처리 끝


# In[ ]:





# In[123]:


df_last['가격'].describe()


# In[124]:


### 시각화를 해서 살펴봅니다. 데이터의 분포를 파악할때, 시각화각 가장 좋은 방법 중 한개 입니다.!! 
### .hist(): 히스토그램을 의미합니다. bins=50:주머니가 50개 이다. x가 50개로 나누어 진다라고 이해하셔도 좋습니다. 
### y축은 frequency 빈도수입니다. x 축은 실제 갑습니다. 
df_last['가격'].hist(bins=50)


# In[125]:


plt.boxplot(df_last['가격'])
plt.show()


# In[ ]:


설명변수(독립변수, features, attributes, x) 살펴보기


# In[126]:


numerical_columns=['봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량', '최고기온', '평균기온', '최저기온', '강수량']

# 가격은 반영 안했음

### figsize()는 plot()의 기본 크기를 지정합니다. 
fig = plt.figure(figsize = (16, 20))
ax = fig.gca()  # Axes 생성

### gca(), gcf(), axis()
### gca()로 현재의 Axes를, gcf()로 현재의 Figure 객체를 구할 수 있다.
### ax=plt.gica(): 축의 위치를 호출하여 ax로 설정(축 위치 변경을 위해 필요한 과정)

df_last[numerical_columns].hist(ax=ax)
plt.show()


# In[73]:


설명변수(x) 와 종속변수(y) 간의 관계 탐색


# In[127]:


### Person 상관계수 : 대표적으로 상관관계 분석시 사용하는 지표입니다.
### -1 에서 1 사이의 값을 가진다는 특징이 있습니다.
### 1일 때는 완전 양의 상관(perfect positive correlation), -1일 때는 완전 음의 상관관계(perfect negative correlation)관계를 보입니다.
### https://m.blog.naver.com/istech7/50153047118

cols = ['가격','봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량', '최고기온', '평균기온', '최저기온', '강수량']

corr = df_last[cols].corr(method = 'pearson')
corr


# In[128]:


### 상관관계를 직관적으로 살펴보기 위해 Heatmap 으로 돌려봅니다.
### heatmap (seaborn): 여기서는 seaborn 시각화 라이브러리를 사용해서 표현합니다. 
### 시각화의 대표적인 라이브러리가 matplot(https://matplotlib.org/)과 seaborn(https://seaborn.pydata.org/)이 있습니다.

fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정 
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = cols, xticklabels = cols, ax=ax, cmap = "RdYlBu")
plt.tight_layout()
plt.show()


# In[ ]:


가격이랑 평균기온 관계


# In[129]:


### scatter plot 산점도, https://seaborn.pydata.org/generated/seaborn.scatterplot.html
sns.scatterplot(data=df_last, x='평균기온', y='가격', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()


# In[ ]:


가격이랑 강수량 관계


# In[130]:


sns.scatterplot(data=df_last, x='강수량', y='가격', markers='o', color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.show()


# In[ ]:


## ------------------------------------------------------------------- 선형회귀 시작


# In[131]:


df_last.head()


# In[132]:


df_last.info()


# In[133]:


from sklearn.preprocessing import StandardScaler

# feature standardization  (numerical_columns except dummy var.-"CHAS")

scaler = StandardScaler()  # 평균 0, 표준편차 1
scale_columns = ['봄배추 반입량', '여름배추 반입량', '김장(가을)배추 반입량', '월동배추 반입량', '최고기온', '평균기온', '최저기온', '강수량']
df_last[scale_columns] = scaler.fit_transform(df_last[scale_columns])


# In[134]:


df_last.head()


# In[135]:


df_last[scale_columns].head()


# In[136]:


df_last[numerical_columns].head()


# In[137]:


from sklearn.model_selection import train_test_split

# split dataset into training & test
X = df_last[numerical_columns]
y = df_last['가격']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[138]:


X_train.shape, y_train.shape


# In[139]:


X_test.shape, y_test.shape


# In[140]:


y_train
### 가격 값을 의미합니다.


# In[141]:


X_train


# In[142]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.round(1) 
### 소수점 첫째자리까지 표시합니다. 즉, 소수점 둘째짜리에서 반올림 합니다.


# In[ ]:


#일반적으로, VIF > 10인 feature들은 다른 변수와의 상관관계가 높아, 다중공선성이 존재하는 것으로 판단합니다. 
#즉, VIF > 10인 feature들은 설명변수에서 제거하는 것이 좋을 수도 있습니다.


# In[ ]:


회귀 모델링


# In[ ]:


(1) 빼지 않았을 경우


# In[143]:


from sklearn import linear_model

# fit regression model in training set
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# predict in test set
pred_test = lr.predict(X_test)


# In[144]:


print(lr.coef_)


# In[149]:


coefs = pd.DataFrame(zip(df_last[numerical_columns].columns, lr.coef_), columns = ['feature', 'coefficients'])
coefs


# In[150]:


coefs_new = coefs.reindex(coefs.coefficients.abs().sort_values(ascending=False).index)
coefs_new


# In[151]:


### coefficients 를 시각화 합니다. 

### figure size
plt.figure(figsize = (8, 8))

### bar plot : matplotlib.pyplot 모듈의 barh() 함수를 사용해서 수평 막대 그래프를 그릴 수 있습니다. 
plt.barh(coefs_new['feature'], coefs_new['coefficients'])
plt.title('"feature - coefficient" Graph')
plt.xlabel('coefficients')
plt.ylabel('features')
plt.show()


# In[152]:


import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)
### 회귀분석모형 수식을 간단하게 만들기 위해 다음과 같이 상수항을 독립변수 데이터에 추가하는 것을 상수항 결합(bias augmentation)작업이라고 합니다.
### ordinary least square 의 약자로, 거리의 최소값을 기준으로 구하는 함수입니다. 

model2 = sm.OLS(y_train, X_train2).fit()
model2.summary()


# In[153]:


### 예측 결과 시각화 (test set)
df_last = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df_last = df_last.sort_values(by='actual').reset_index(drop=True)
df_last.head()


# In[156]:


plt.figure(figsize=(12, 9))
plt.scatter(df_last.index, df_last['prediction'], marker='x', color='r')
plt.scatter(df_last.index, df_last['actual'], alpha=0.3, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)
plt.show()


# In[ ]:




