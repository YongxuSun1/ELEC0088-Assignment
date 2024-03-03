import pandas as pd

# Read the dataset ( Weather & Stock)
weather = pd.read_csv("london_weather.csv")
stock = pd.read_csv("London stock.csv")

# Normalized the expression of dates
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')
stock['Date'] = pd.to_datetime(stock['Date'])
# print(weather)
# print(stock)

# Transfer the 'date' to 'Date'
weather.rename(columns={'date': 'Date'}, inplace=True)

# Merged these datasets
merged = pd.merge(stock, weather, on="Date")
print(merged)

# Whether dataset has 'NAN' or '', if so, insert it use the previous value
if merged.isna().any().any():
    # Use the previous value to insert
    merged = merged.ffill()
else:
    print("DataFrame does not contain NaN values.")

# Save the merged dataset
merged.to_csv('merged_data.csv', index=False)
