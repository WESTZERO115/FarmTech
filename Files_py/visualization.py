import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd 

data = pd.read_csv('C:/Users/8305-01/Desktop/onion_test.csv', encoding='cp949')
# xy = np.array(data['Quantity', 'Price'], dtype=np.float32)
x = np.array(data['Quantity'])
y = np.array(data['Price'])
plt.plot(x,y)

plt.show()