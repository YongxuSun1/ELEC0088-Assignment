import socket
from _thread import *
import threading

import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load data and preprocessing
df = pd.read_csv('merged_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df_feature = df.drop(['mean_temp', 'Date'], axis=1)

step = 5
# Load the model and scaler
model = load_model('GRU_best_model.keras')
x_scaler = joblib.load('GRU_x_scaler.save')
y_scaler = joblib.load('GRU_y_scaler.save')


def Predict(date_input):
    global x_scaler, y_scaler, step, df, df_feature
    receive_date_dt = pd.to_datetime(date_input)
    print(receive_date_dt)
    row_number = None

    # Find the date location
    if receive_date_dt in df['Date'].values:
        row_number = df.index[df['Date'] == receive_date_dt].tolist()[0]
    else:
        # If not in the dataset, then find the closest day
        closest_date = df.iloc[(df['Date'] - receive_date_dt).abs().argsort()[:1]]
        row_number = closest_date.index[0]

    # Choose the previous date's data to predict
    if row_number >= step:
        data_to_use = df_feature.iloc[(row_number - step):row_number]
    else:
        data_to_use = df_feature.iloc[:row_number]
        step = row_number  # Adjust step based on available data

    # Standardize the data
    data_to_use_scaled = x_scaler.transform(data_to_use)

    # Reshape the data
    data_to_use_reshaped = data_to_use_scaled.reshape(1, step, -1)

    # Predict the data
    predicted_weather = model.predict(data_to_use_reshaped)
    predictions = y_scaler.inverse_transform(predicted_weather)
    return predictions


def threaded(c):
    greeting_message = "Hello, this is oracle robot, how can I help today?"
    c.send(greeting_message.encode())

    while True:
        try:
            data = c.recv(1024)
            if not data:
                print('Client disconnected')
                break

            user_input = data.decode().lower()

            if "what is the average temperature" in user_input:
                response = "Please enter the date (e.g. 2002-05-29):"
                c.send(response.encode())
                data = c.recv(1024)
                date_input = data.decode()

                try:
                    predictions = Predict(date_input)
                    response = f"Predicted average temperature for {date_input}: {predictions[0][0]:.2f}"
                except Exception as e:
                    print(e)
                    response = "Error in processing date. Please make sure it is in YYYY-MM-DD format."

                c.send(response.encode())

            else:
                response = "Please ask about the average temperature on a specific date."
                c.send(response.encode())
        except ConnectionAbortedError:
            print('Connection aborted by the client')
            break

    c.close()


def Main():
    host = ""
    port = 65432
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("Socket binded to port", port)
    s.listen(5)
    print("Socket is listening")

    while True:
        c, addr = s.accept()
        print('Connected to:', addr[0], ':', addr[1])
        start_new_thread(threaded, (c,))

    s.close()


if __name__ == '__main__':
    Main()
