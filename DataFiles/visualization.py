import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

data = pd.read_csv('C:/Users/8305-01/Desktop/data.csv', encoding='cp949')
w_data = pd.read_csv('C:/Users/8305-01/Desktop/w_data.csv', encoding='cp949')
# xy = np.array(data['Quantity', 'Price'], dtype=np.float32)
# p_data = pd.read_csv('C:/Users/8305-01/Desktop/data_p.csv', encoding='cp949')

x = np.array((data['total']))
# x = np.array((p_data['sum_produced']))
y = np.array(w_data['temp'])
plt.xlabel('Import quantity')
# plt.xlabel('produced_quantity')
plt.ylabel('Temperature')

plt.scatter(x,y)

plt.show()


