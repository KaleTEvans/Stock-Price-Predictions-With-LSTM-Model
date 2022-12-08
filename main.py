import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf

stock_price_df = pd.read_csv('Files/stock.csv')
stock_vol_df = pd.read_csv('Files/stock_volume.csv')

stock_price_df.sort_values(by = ['Date'])
stock_vol_df.sort_values(by = ['Date'])

# Plot function for data visualization
def interactive_plot(df, title):
    fig = px.line(title = title)

    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)

    fig.show()

# Concatenate the date, stock price, and volume in one dataframe
def individual_stock(price_df, vol_df, ticker):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[ticker], 'Volume': vol_df[ticker]})

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')

# Get the close and volume data as training data
training_data = price_volume_df.iloc[:, 1:3].values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_data)

# Create the training and testing data
x = []
y = []

for i in range(1, len(price_volume_df)):
    x.append(training_set_scaled[i-1:i, 0])
    y.append(training_set_scaled[i, 0])

    # Convert into array format
X = np.asarray(x)
Y = np.asarray(y)

# Split the data for 70% training and 30% testing
split = int(0.7 * len(X))
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]

# Reshape the 1D arrays to 3D arrays to feed to model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))