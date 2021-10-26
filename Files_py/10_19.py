####### 다변인 선형회귀 모델 개발 #######

# 텐서플로우와 넘파이 import
import tensorflow as tf
import numpy as np
# Info성 불필요 메시지 미출력을 위한 작업
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 엑셀 데이터를 읽어올 수 있도록 하기 위해 import
from pandas.io.parsers import read_csv

# 모델 초기화
model = tf.compat.v1.global_variables_initializer()

# csv 파일 읽어들여 data에 대입 ','를 토큰으로 하여 데이터를 분리하여 담는다.
# ( 해당 파일을 메모장으로 열어보면 ','를 기준으로 데이터들이 나열되어있기 때문 )
data = read_csv('/Users/parkseoyoung/Desktop/aipSources/price_data_sample.csv', encoding='UTF8', dtype=np.float32, sep=',')

# 행렬 형태로 해당 데이터를 xy에 대입
xy = np.array(data)

#정상적으로 들어갔는지 확인 (중간 점검, 확인만 해보고 삭제할 것)
print(xy)



# xy 배열에서 모든 배열의 avgTemp, minTemp, maxTemp, rainFall 값을 x_data로 설정
# B, C, D, E 값을 가져옴
x_data = xy[:, 1:-1]

# xy 배열에서 모든 배열의 avgPrice 값을 y_data로 설정
# F 값을 가져옴
y_data = xy[:, [-1]]

# X는 총 4개의 열이 담길 수 있도록 설정
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

# Y는 총 1개의 열이 담길 수 있도록 설정
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 가중치 값 초기화
W = tf.Variable(tf.compat.v1.random_normal([4, 1]), name="weight")
#W = tf.Variable(tf.random_normal([4, 1]), name="weight")

# bias 값 초기화
b = tf.Variable(tf.compat.v1.random_normal([1]), name="bias")
#b = tf.Variable(tf.random_normal([1]), name="bias")


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
# for step in range(10001):
#     hypo2, cost2, third = sess.run([hypothesis, cost, train], feed_dict={X: x_data, Y: y_data})
#     if step % 500 == 0:
#         print("#", step, "손실 비용: ", cost)
#         print("- 배추 가격: ", hypo2[0])
for step in range(10001):
    cost_,  hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("#", step, " 손실 비용: ", cost_)
        print("- 배추 가격: ", hypo_[0])

avg_temp = float(input('평균 온도: '))
min_temp = float(input('최저 온도: '))
max_temp = float(input('최고 온도: '))
rain_fall = float(input('강수량: '))

sess.run(init)

data_ = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
arr = np.array(data_, dtype=np.float32)
x_data_ = arr[0:4]
dict = sess.run(hypothesis, feed_dict={X: x_data_})
print("예상되는 배추 가격 : " + str(dict[0]))

# # 학습한 결과를 파일로 저장, 추후 저장된 모델을 언제든지 불러들여 사용할 수 있도록 함
# saver = tf.compat.v1.train.Saver()
# save_path = saver.save(sess, "./saved.cpkt")
# print("학습된 모델을 저장하였습니다.")