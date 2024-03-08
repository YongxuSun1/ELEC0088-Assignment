import socket
from _thread import *
import threading

import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load data and preprocessing
df = pd.read_csv('merged_data.csv')
df_feature = df.drop(['mean_temp', 'Date'], axis=1)
df['Date'] = pd.to_datetime(df['Date'])
step = 5
# Load the model and scaler outside of the threaded function
model = load_model('myModel.h5')
x_scaler = joblib.load('LSTM_x_scaler.save')
y_scaler = joblib.load('LSTM_y_scaler.save')

def threaded(c, model, x_scaler):
    greeting_message = "hello"
    c.send(greeting_message.encode())

    while True:
        try:
            data = c.recv(1024)
            if not data:
                print('Client disconnected')
                break

            user_input = data.decode().lower()

            if "average" in user_input:
                response = "Please enter the date (e.g. 2002 5 29):"
                c.send(response.encode())
                data = c.recv(1024)
                date_input = data.decode()

                try:
                    date_formatted = pd.to_datetime(date_input)

                    # Check if the date is in the dataset or find the closest date before the input date
                    if date_formatted in df['Date'].values:
                        row_number = df.index[df['Date'] == date_formatted][0]
                    else:
                        closest_dates = df[df['Date'] < date_formatted]
                        row_number = closest_dates.index[-1]

                    # Ensure we have enough data to create a sequence
                    start_index = max(row_number - step + 1, 0)
                    data_to_use = df_feature.iloc[start_index:row_number + 1]

                    # Pad the sequence if necessary
                    if len(data_to_use) < step:
                        padding = np.zeros((step - len(data_to_use), data_to_use.shape[1]))
                        data_to_use = np.vstack([padding, x_scaler.transform(data_to_use)])

                    else:
                        data_to_use = x_scaler.transform(data_to_use)

                    data_to_use = np.expand_dims(data_to_use, axis=0)  # Reshape for the model if necessary

                    predictions = model.predict(data_to_use)
                    predictions = y_scaler.inverse_transform(predictions)
                    response = f"Predicted average temperature for {date_input}: {predictions[0][0]}"
                except ValueError:
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
        start_new_thread(threaded, (c, model, x_scaler))

    s.close()

if __name__ == '__main__':
    Main()
