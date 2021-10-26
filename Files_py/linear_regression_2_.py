import tensorflow as tf
import numpy as np
# Info성 불필요 메시지 미출력을 위한 작업
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.reset_default_graph()   ##############
# 플레이스 홀더 세팅
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.compat.v1.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.compat.v1.random_normal([1]), name="bias")

# 가설 세팅
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러올 객체 선언 및 초기화를 위한 변수 model 세팅
saver = tf.compat.v1.train.Saver()
model = tf.compat.v1.global_variables_initializer()

# 사용자로부터 값 입력받음 - 4개 변수
avg_temp = float(input('평균 온도: '))
min_temp = float(input('최저 온도: '))
max_temp = float(input('최고 온도: '))
rain_fall = float(input('강수량: '))

# # 텐서플로우의 세션을 이용해서 데이터를 모델에 적용한 결과값을 가져오기
# with tf.compat.v1.Session() as sess:
#     # 세션 초기화
#     sess.run(model)
    
#     # restore 함수를 이용해 학습 모델을 그대로 가져옴
#     save_path = "/Users/parkseoyoung/Desktop/aipSources/saved.cpkt"
#     saver.restore(sess, save_path)  
#     # saver = tf.compat.v1.train.import_meta_graph('saved.cpkt.meta')
#     # saver.restore(sess, save_path)  
#     # saver.restore(sess, tf.train.latest_checkpoint('./'))
    
#     #tf.compat.v1.reset_default_graph()  
    
#     #new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
#     #new_saver.restore(sess, tf.train.latest_checkpoint('./'))

#     # 사용자의 입력값을 통해 2차원 배열을 생성
#     data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
#     # 튜플 형태 data와 그가 포함한 튜플을 모두 list 형태로 변경
#     arr = np.array(data, dtype=np.float32)
#     x_data = arr[0:4]
#     #### 해당 부분에서 data에 (0, 0, 0, 0) 누락시 Error 발생
#     #### 해당 부분에서 x_data 에 arr[0] 으로 입력시 Error 발생

    
#     dict = sess.run(hypothesis, feed_dict={X: x_data})
#     # print(dict[0])
#     print("예상되는 배추 가격 : " + str(dict[0][0])) 

# 텐서플로우의 세션을 이용해서 데이터를 모델에 적용한 결과값을 가져오기
with tf.compat.v1.Session() as sess:
    # 세션 초기화
    sess.run(model)
    # restore 함수를 이용해 학습 모델을 그대로 가져옴
    # save_path = "/Users/parkseoyoung/Desktop/aipSources/saved.cpkt.meta"
    # # saver = tf.compat.v1.train.import_meta_graph('saved.cpkt.meta')
    # saver.restore(sess, save_path)  
    saver = tf.compat.v1.train.import_meta_graph('./saved.cpkt.meta')
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))
    

    # 사용자의 입력값을 통해 2차원 배열을 생성
    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    # 튜플 형태 data와 그가 포함한 튜플을 모두 list 형태로 변경
    arr = np.array(data, dtype=np.float32)
    x_data = arr[0:4]
    #### 해당 부분에서 data에 (0, 0, 0, 0) 누락시 Error 발생
    #### 해당 부분에서 x_data 에 arr[0] 으로 입력시 Error 발생

    # dict = sess.run(hypothesis, feed_dict={X: x_data})
    # dict = sess.run(hypothesis, feed_dict={X: x_data})
    # model.add(x_data.Dense(16, activation='relu', input_shape=(4,)))
    # x_data = x_data.reshape(2,4)
    
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print("예상되는 배추 가격 : " + str(dict[0]))