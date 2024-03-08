import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras . models import Sequential
from keras . layers import Dense, LSTM
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



"""
# Create the correlation Heatmap
correlation = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()
"""

# Features and label

y = df.loc[:, 'mean_temp']
X = df.drop(['mean_temp', 'Year', 'Month', 'Day'], axis=1)
y = y.to_numpy()
X = X.to_numpy()

# Standard the dataset
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
joblib.dump(scaler_X, 'LSTM_x_scaler.save')
joblib.dump(scaler_y, 'LSTM_y_scaler.save')
# Reshape the data
time_steps = 5
samples = len(X) - time_steps
X_reshaped = np.zeros((samples, time_steps, X.shape[1]))
y_reshaped = np.zeros((samples, 1))

for i in range(samples):
    X_reshaped[i] = X_scaled[i:i+time_steps]
    y_reshaped[i] = y_scaled[i+time_steps]

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X_reshaped,
                                                    y_reshaped, test_size=0.1)

# Reshape y_train and y_test
y_train = y_train . reshape((-1, 1))
y_test = y_test . reshape((-1, 1))

print(x_train.shape, y_train.shape)
# define the batch size and the number of epochs
batch_size = 32
epochs = 50

# Create the model
num_input = X.shape[1]
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(time_steps, num_input)))
model.add(Dense(1, activation='linear'))  #
model . compile(loss="mse", optimizer=SGD())

# Define the ModelCheckpoint
checkpoint = ModelCheckpoint('LSTM_best_model.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')


model . fit(x_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_split=0.1,callbacks=[checkpoint])

# Evaluation
mse = model . evaluate(x_test, y_test, verbose=0)

print ('MSE:', mse)





