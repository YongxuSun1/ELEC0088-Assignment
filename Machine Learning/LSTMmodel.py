import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras . models import Sequential
from keras . layers import Dense, LSTM, Dropout
from sklearn . preprocessing import StandardScaler
import joblib
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Read the merged dataset (Weather + Stock)
df = pd.read_csv('merged_data.csv')


# Transfer Date to 'Year, Month, Day'
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date', axis=1)

# Features and label

y = df.loc[:, 'mean_temp']
X = df.drop(['mean_temp'], axis=1)
y = y.to_numpy()
X = X.to_numpy()

# Standard the dataset
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
joblib.dump(scaler_X, 'LSTM_x_scaler.save')
joblib.dump(scaler_y, 'LSTM_y_scaler.save')

# define the batch size and the number of epochs
batch_size =32
epochs = 50
time_steps = 1

# Reshape the data

samples = len(X) - time_steps
X_reshaped = np.zeros((samples, time_steps, X.shape[1]))
y_reshaped = np.zeros((samples, 1))

for i in range(samples):
    X_reshaped[i] = X_scaled[i:i+time_steps]
    y_reshaped[i] = y_scaled[i+time_steps]

# Split the dataset

test_size = int(len(X) * 0.1)
x_train = X_reshaped[:-test_size]
x_test = X_reshaped[-test_size:]
y_train = y_reshaped[:-test_size]
y_test = y_reshaped[-test_size:]
print(x_train.shape,X_reshaped.shape)
y_train = y_train . reshape((-1, 1))
y_test = y_test . reshape((-1, 1))

# Create the model
num_input = X.shape[1]
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(time_steps, num_input),return_sequences=False))
#model.add(Dropout(0.2))
#model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))  #

model . compile(loss="mse", optimizer='sgd',)

# Define the ModelCheckpoint
checkpoint = ModelCheckpoint('LSTM_best_model.keras',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')


history = model . fit(x_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_split=0.1,callbacks=[checkpoint])

# Plot the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation
mse = model . evaluate(x_test, y_test, verbose=0)

# model = load_model('LSTM_best_model.h5')
predictions = model.predict(x_test)
predictions = scaler_y.inverse_transform(predictions)
y_test = scaler_y.inverse_transform(y_test)

print('LSTM MSE:', mse)

# Plot the actual value and predicted value
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values')
plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.5, label='Predicted values')
plt.title('Actual vs. Predicted values')
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.legend()
plt.show()
