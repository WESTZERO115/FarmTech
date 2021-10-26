import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

f = open('price_data_sample.csv', 'r')
desk_data = pd.read_csv(f, header=0)
seq = desk_data[['avgTemp', 'minTemp', 'maxTemp','rainFall', 'avgPrice']].to_numpy()


coindesk_data.describe() 

transform = MinMaxScaler()
desk_data = transform.fit_transform(X, fit_params)


