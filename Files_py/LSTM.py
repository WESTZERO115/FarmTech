import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = open('price_data_sample.csv', 'r')
coindesk_data = pd.read_csv(f, header=0)
seq = coindesk_data[['avgTemp', 'minTemp', 'maxTemp','rainFall', 'avgPrice']].to_numpy()
##### πμ μ½λμ λ¬λΌμ§ μ . #####

def seq2dataset(seq, window, horizon):
  X = []; Y = []
  for i in range(len(seq)-(window+horizon)+1):
    x = seq[i:(i+window)]
    y = (seq[i+window+horizon-1])
    X.append(x); Y.append(y)
  return np.array(X), np.array(Y)

w=7
h=1

X,Y = seq2dataset(seq,w,h)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

split = int(len(X)*0.7)
x_train = X[0:split]; y_train = Y[0:split]
x_test = X[split:]; y_test = Y[split:]

model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(5))
##### πμ μ½λμ λ¬λΌμ§ μ . #####
model.compile(loss='mae', optimizer ='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data = (x_test,y_test), verbose = 2)

ev = model.evaluate(x_test, y_test, verbose = 0)
print("μμ€ ν¨μ:", ev[0], "MAE:", ev[1])

pred = model.predict(x_test)
# print("νκ· μ λκ°λ°±λΆμ¨μ€μ°¨(MAPE):", sum(abs(y_test-pred)/y_test)/len(x_test))

plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.ylim([100,600])
##### πμ μ½λμ λ¬λΌμ§ μ . #####
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()

x_range = range(len(y_test))
plt.plot(x_range,y_test[x_range], color='red')
plt.plot(x_range, pred[x_range], color='blue')
plt.legend(['True prices', 'Predicted prices'], loc='best')
plt.grid()
plt.show()