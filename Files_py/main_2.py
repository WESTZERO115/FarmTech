import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
# 엑셀 데이터를 읽어올 수 있도록 하기 위해 import
from pandas.io.parsers import read_csv

model = tf.compat.v1.global_variables_initializer()

# def get_dataset(file_path, **kwargs):
#   dataset = tf.data.experimental.make_csv_dataset(
#       file_path,
#       batch_size=5, # Artificially small to make examples easier to show.
#       label_name=LABEL_COLUMN,
#       na_value="?",
#       num_epochs=1,
#       ignore_errors=True, 
#       **kwargs)
#   return dataset

# data = read_csv('C:/Users/8305-01/Desktop/onion_test.csv', encoding='cp949', 
#                 dtype = {"Date": str, 
#                          "Quantity": int, 
#                          "Price": int})

data = read_csv('C:/Users/8305-01/Desktop/onion_test_.csv', encoding='cp949', dtype=np.float32)


####
xy = np.array(data)
print(xy)

x_data = xy[:, 0:-1]  ## A,B 날짜, 반입량.
y_data = xy[:, [-1]]  ## C 가격.

tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 가중치 값 초기화
W = tf.Variable(tf.compat.v1.random_normal([2, 1]), name="weight")

# bias 값 초기화
b = tf.Variable(tf.compat.v1.random_normal([1]), name="bias")

# 선형 회귀의 경우 행렬의 곱 연산을 이용하여 결과식을 세울 수 있다
hypothesis = tf.matmul(X, W) + b

# 비용함수 : tensorflow에서 기본적으로 제공해주는 reduce_mean 사용
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 함수 : 경사하강법 : 학습률 0.000005로 세팅
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.000005)

# cost를 최소화하는 최적값 계산
train = optimizer.minimize(cost)

# 세션 생성 및 초기화
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# 1만번 트레이닝 진행
for step in range(10001):
    hypo2, cost2, third = sess.run([hypothesis, cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("#", step, "손실 비용: ", cost)
        print("- 양파 가격: ", hypo2[0])


# 학습한 결과를 파일로 저장, 추후 저장된 모델을 언제든지 불러들여 사용할 수 있도록 함
saver = tf.compat.v1.train.Saver()
save_path = saver.save(sess, "./saved.cpkt")
print("학습된 모델을 저장하였습니다.")



