import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl

#Variable의 랜덤함수 시드를 설정한다.
tf.random.set_seed(777)

#MinMaxScaler 정의 -> data를 0부터 1사이의 값으로 변환(normalize)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


#train Parameter 상수값 설정
seq_length = 1 # 월 단위로 학습시키고 다음 달을 예측  (현재 데이터는 월 단위로 있음)
input_dim = 2 #input 데이터의 개수
hidden_dim = 16 #은닉층의 개수
output_dim = 1 # target label의 개수(양파 가격)
learning_rate = 0.01
iterations = 10001


# 값 설정이 끝나면 위에서 만든 데이터 셋을 불러온다. 
# 그 후 데이터양에 따라 적절하게 Training Data와 Test Data로 분할한다.

# df_sheet_index = pd.read_excel('C:\\Users\\8305-01\\Desktop\\onion.xlsx', sheet_name=0)
# print(df_sheet_index)

#data load
#만들어둔 데이터셋을 Load한다.
xy = pd.read_excel('C:\\Users\\8305-01\\Desktop\\onion_test.xlsx', sheet_name=0)
xy = MinMaxScaler(xy) #Normalize
x = xy #전체 데이터(input)
y = xy[:,[-1]] #target Label(양파가격)


#build dataset
#Normalize가 끝난 데이터를 월 단위로 슬라이스 해서 3차원형태로 만든다.
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i+seq_length]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)
    
    
#train/test set 나누기
train_size = int(len(dataY) * 0.7) #train size= 70%
test_size = len(dataY) - train_size #test size = 30%
trainX, testX = np.array(dataX[0:train_size]),np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]),np.array(dataY[train_size:len(dataY)])



# 데이터 셋 분할이 끝났으면 해당 데이터를 받을 Placeholder와 인공신경망 구조인 LSTM을 구축한다.


#input placeholders
#input placeholder의 파라미터로 seq_length,input_dim를 넘겨준다.
X = tf.placeholder(tf.float32, [None,seq_length, input_dim]) #3차원 형태의 input placeholder
Y = tf.placeholder(tf.float32, [None, 1])                    #2차원 형태의 output placeholder


#build a LSTM network (build Rnn)
#tensorflow를 이용하면 LSTM 구조를 cell 형태로 빠르고 쉽게 생성해준다.
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#생성된 cell을 dynamic_rnn함수로 활성화시켜준다. 파라미터로 생성된 cell과 input
# placeholder를 넘겨주고 리턴값으로 output을 받는다.
outputs, _states = tf.nn.dynamic_rnn(cell, X,dtype = tf.float32)
#그 후 출력된 output을 fully_connected에 통과 시켜 최종 예측값을 리턴 받는다.
Y_pred =tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=tf.tanh)



















