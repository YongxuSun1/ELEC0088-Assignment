import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras . models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras . optimizers import SGD
from sklearn . preprocessing import StandardScaler
from sklearn . model_selection import train_test_split
import joblib

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

# Reshape X & y
def create_sequences(features, labels, time_steps=5):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)
time_steps = 20
X, y = create_sequences(features, labels, time_steps)

y = y.reshape(-1,1)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)

# Split the dataset

test_size = int(len(X) * 0.1)  # Split size is 0.1
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create model
batch_size = 32
epochs = 20

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
# model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='sgd', loss='mse')

checkpoint = ModelCheckpoint('CNN_best_model.keras',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

history = model . fit(X_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_split=0.1,callbacks=[checkpoint])

# Evaluation
mse = model . evaluate(X_test, y_test, verbose=0)

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
mse = model . evaluate(X_test, y_test, verbose=0)
# model = load_model('LSTM_best_model.h5')
predictions = model.predict(X_test)
predictions = y_scaler.inverse_transform(predictions)
y_test = y_scaler.inverse_transform(y_test)

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
