import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam


import pandas as pd

# Load the datasets
temperature_df = pd.read_csv(r"C:\Users\blind\Downloads\NHEE\NEE2106\PROJECT PARTB REVA\DATA\BOM_year.csv")
house3_east_df = pd.read_csv(r"C:\Users\blind\Downloads\NHEE\NEE2106\PROJECT PARTB REVA\DATA\House 3_Melb East.csv")
house4_west_df = pd.read_csv(r"C:\Users\blind\Downloads\NHEE\NEE2106\PROJECT PARTB REVA\DATA\House 4_Melb West.csv")
house4_solar_df = pd.read_csv(r"C:\Users\blind\Downloads\NHEE\NEE2106\PROJECT PARTB REVA\DATA\House 4_Solar.csv")

# Preview the column headers and first few rows to understand the structure
(temperature_df.head(), house3_east_df.head(), house4_west_df.head(), house4_solar_df.head())


# --- Step 1: Clean and Format the Data ---

# Clean column names
temperature_df.columns = ['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Temp9AM', 'Temp3PM']
temperature_df['Date'] = pd.to_datetime(temperature_df['Date'], format='%d-%b-%Y')
temperature_df = temperature_df[['Date', 'MinTemp', 'MaxTemp', 'Temp9AM', 'Temp3PM']]

# Parse power CSVs
def parse_power_csv(df):
    df.columns = ['Timestamp', 'Power']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna()
    df['Date'] = df['Timestamp'].dt.date
    daily_avg = df.groupby('Date')['Power'].mean().reset_index()
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    return daily_avg

house3_east_avg = parse_power_csv(house3_east_df)
house4_west_avg = parse_power_csv(house4_west_df)
house4_solar_avg = parse_power_csv(house4_solar_df)

# Combine power data (optional: average across all houses)
power_combined = pd.concat([house3_east_avg, house4_west_avg, house4_solar_avg]).groupby('Date')['Power'].mean().reset_index()

# --- Step 2: Merge with Temperature Data on Date ---
merged_df = pd.merge(temperature_df, power_combined, on='Date', how='inner')

# Drop rows with NaNs (if any)
merged_df = merged_df.dropna()

# Extract input features and target
X = merged_df[['MinTemp', 'MaxTemp', 'Temp9AM', 'Temp3PM']].values
y = merged_df['Power'].values.reshape(-1, 1)

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- Step 3: Train-Test Split (80/20) ---
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

# Reshape for RNN: [samples, time_steps, features]
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# --- Step 4: Build and Train RNN Model ---
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(1, 4)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
history = model.fit(X_train_rnn, y_train, epochs=100, verbose=0)

# --- Step 5: Predict and Inverse Scale ---
y_pred_scaled = model.predict(X_test_rnn)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# --- Step 6: Plot and Evaluate ---
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label='Actual Power')
plt.plot(y_pred, label='Predicted Power')
plt.title('Actual vs Predicted Power Consumption (Test Data)')
plt.xlabel('Sample Index')
plt.ylabel('Power (Watts)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 7: Error Metrics ---
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

(mae, rmse)
print("MAE:", mae)
print("RMSE:", rmse)

