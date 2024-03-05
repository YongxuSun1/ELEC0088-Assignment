from keras.models import load_model
import joblib
import pandas as pd

# load the model and dataset
x_scaler = joblib.load('LSTM_x_scaler.save')
y_scaler = joblib.load('LSTM_y_scaler.save')
model = load_model('LSTM_best_model.h5')
df = pd.read_csv('merged_data.csv')
df_feature = df.drop(['mean_temp', 'Date'], axis=1)
step = 5  # Day step

# Preprocess the dataset
df['Date'] = pd.to_datetime(df['Date'])

# Type the date
receive_date = input('Type the date')
receive_date_dt = pd.to_datetime(receive_date)

# Find the date location
if receive_date_dt in df['Date'].values:
    row_number = df.index[df['Date'] == receive_date_dt].tolist()[0]

else:
    # if not in the dataset, then find the closed day
    next_date = df[df['Date'] > receive_date_dt].min()['Date']
    row_number = df.index[df['Date'] == next_date].tolist()[0]

# Choose the previous date's data to predict
if row_number >= 5:
    data_to_use = df_feature.iloc[(row_number - 5):row_number]

else:
    data_to_use = df_feature.iloc[:row_number]
    step = row_number

# standardized the data
data_to_use = x_scaler.transform(data_to_use)

# Reshape the data
data_to_use_reshaped = data_to_use.reshape(1, step, -1)
# Predict the data
predicted_weather = model . predict(data_to_use_reshaped)
predicted_weather = y_scaler.inverse_transform(predicted_weather)
print(predicted_weather)

