import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras . models import Sequential
from keras . layers import Dense, GRU
from keras . optimizers import SGD
from sklearn . preprocessing import StandardScaler
from sklearn . model_selection import train_test_split
import joblib


# Read the merged dataset (Weather + Stock)
df = pd.read_csv('merged_data.csv')


# Transfer Date to 'Year, Month, Day'
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date', axis=1)


labels = df.loc[:, 'mean_temp']
features = df.drop(['mean_temp'], axis=1)
x_scaler = StandardScaler()
features = x_scaler.fit_transform(features)

def create_sequences(features, labels, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(features, labels, time_steps)

y = y.reshape(-1,1)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, y_train.shape)


# Create the model
model = Sequential([
    GRU(64, input_shape=(time_steps, features.shape[1])),
    Dense(1, activation='linear')
    ])

model.compile(optimizer=SGD(), loss='mse')
batch_size = 32
epochs = 100

model . fit(X_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_split=0.1)

mse = model . evaluate(X_test, y_test, verbose=0)

print('MSE:', mse)
