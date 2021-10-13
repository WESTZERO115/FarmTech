import numpy as np
import tensorflow as tf
import time
import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/8305-01/Desktop/onion_test.csv', encoding='cp949')
# xy = np.array(data['Quantity'], dtype=np.float32)
# xy = np.array(data, dtype=np.float32)

t = np.array(data)

#fashion MNIST를 읽고 신경망에 입력할 형태로 변환
train_data = t
test_data = t

#신경망 구조 설정
n_input=784
n_hidden1=1024
n_hidden2=512
n_hidden3=512
#n_hidden4=512
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
    #model.add(Dense(units=n_hidden4, activation='relu'))
    model.add(Dense(units=n_output, activation='softmax'))
    return model

#교차 검증을 해주는 함수(서로 다른 옵티마이저(opt)에 대해)
def cross_validation(opt):
    accuracy=[]
    for train_index, val_index in KFold(k).split(train_data):
        _train,val=train_data[train_index], train_data[val_index]
        # ytrain, yval=y_train[train_index], y_train[val_index]
        dmlp=build_model()
        dmlp.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        dmlp.fit(train_data,batch_size=batch_siz,epochs=n_epoch, verbose=0)
        accuracy.append(dmlp.evaluate(val,None,verbose=0)[1])
    return accuracy
    
#옵티마이저 4개에 대해 교차 검증을 실행
start=time.time() #시작시간
acc_sgd=cross_validation(SGD())
end=time.time()
print("SGD 실행 결과의 수행 시간은",end-start,"초 입니다.")

start=time.time() #시작시간
acc_adam=cross_validation(Adam())
end=time.time()
print("Adam 실행 결과의 수행 시간은",end-start,"초 입니다.")

start=time.time() #시작시간
acc_adagrad=cross_validation(Adagrad())
end=time.time()
print("Adagrad 실행 결과의 수행 시간은",end-start,"초 입니다.")

start=time.time() #시작시간
acc_rmsprop=cross_validation(RMSprop())
end=time.time()
print("RMSprop실행 결과의 수행 시간은",end-start,"초 입니다.")

#옵티마이저 4개의 정확률을 비교
print("SGD:",np.array(acc_sgd).mean())
print("Adam:",np.array(acc_adam).mean())
print("Adagrad:",np.array(acc_adagrad).mean())
print("RMSprop:",np.array(acc_rmsprop).mean())

import matplotlib.pyplot as plt

#네 옵티마이저의 정확률을 박스플롯으로 비교
plt.boxplot([acc_sgd, acc_adam, acc_adagrad, acc_rmsprop], labels=["SGD","Adam","Adagrad","RMSprop"])
plt.grid()