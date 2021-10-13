# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop
from sklearn.model_selection import KFold
import pandas as pd 

# df1 = pd.read_excel('C:\\Users\\8305-01\\source\\source1\\data1.xlsx', sheet_name=0)
data = pd.read_csv('C:/Users/8305-01/Desktop/onion_test.csv', encoding='cp949')

xy = np.array(data['Quantity'], dtype=np.float32)
# xy.describe()
print(xy)


#EXCEL 데이터를 읽고 신경망에 입력할 형태로 변환
(train_data), (test_data) = xy
train_data = train_data.shape(60, 1)
test_data = test_data.shape(60, 1)

#신경망 구조 설정
n_input=60
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_output=10

#하이퍼 매개변수 설정
batch_siz=512
n_epoch=30
k=4 
#4-겹

#모델을 설계해주는 함수(모델을 나타내는 객채 model을 반환)
def build_model():
    model=Sequential()
    model.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
    model.add(Dense(units=n_hidden2, activation='relu'))
    model.add(Dense(units=n_hidden3, activation='relu'))
    model.add(Dense(units=n_output, activation='softmax'))
    return model

#교차 검증을 해주는 함수(서로 다른 옵티마이저(opt)에 대해)
def cross_validation(opt):
    accuracy=[]
    for train_index, val_index in KFold(k).split(train_data):
        xtrain,xval=train_data[train_index], train_data[val_index]
        ytrain, yval=test_data[train_index], test_data[val_index]
        dmlp=build_model()
        dmlp.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        dmlp.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch, verbose=0)
        accuracy.append(dmlp.evaluate(xval,yval,verbose=0)[1])
    return accuracy
    
#옵티마이저 4개에 대해 교차 검증을 실행
acc_sgd=cross_validation(SGD())
end=time.time()

acc_adam=cross_validation(Adam())
end=time.time()

acc_adagrad=cross_validation(Adagrad())
end=time.time()

acc_rmsprop=cross_validation(RMSprop())
end=time.time()

#옵티마이저 4개의 정확률을 비교
print("SGD:",np.array(acc_sgd).mean())
print("Adam:",np.array(acc_adam).mean())
print("Adagrad:",np.array(acc_adagrad).mean())
print("RMSprop:",np.array(acc_rmsprop).mean())

import matplotlib.pyplot as plt

#네 옵티마이저의 정확률을 박스플롯으로 비교
plt.boxplot([acc_sgd, acc_adam, acc_adagrad, acc_rmsprop], labels=["SGD","Adam","Adagrad","RMSprop"])
plt.grid()