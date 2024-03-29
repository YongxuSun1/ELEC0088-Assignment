import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.callbacks import ModelCheckpoint
from keras . models import Sequential
from keras . layers import Dense, GRU
from keras . optimizers import SGD
from sklearn . preprocessing import StandardScaler


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


time_steps = 5
batch_size = 32
epochs = 50

X, y = create_sequences(features, labels, time_steps)

print(X.shape)

y = y.reshape(-1,1)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)

print(max(y),min(y))
# Split the dataset

test_size = int(len(X) * 0.1)  # Split size is 0.1
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)



# Create the model

model = Sequential([
    GRU(64, activation='relu', input_shape=(time_steps, features.shape[1])),
    Dense(1, activation='linear')])

model.compile(optimizer=SGD(), loss='mse')

checkpoint = ModelCheckpoint('GRU_best_model.keras',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

history = model . fit(X_train, y_train, batch_size=batch_size,
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


mse = model . evaluate(X_test, y_test, verbose=0)

predictions = model.predict(X_test)
predictions = y_scaler.inverse_transform(predictions)
y_test = y_scaler.inverse_transform(y_test)
print('GRU MSE:', mse)

# Plot the actual value and predicted value
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values')
plt.scatter(range(len(predictions)), predictions, color='red', alpha=0.5, label='Predicted values')
plt.title('Actual vs. Predicted values')
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.legend()
plt.show()

