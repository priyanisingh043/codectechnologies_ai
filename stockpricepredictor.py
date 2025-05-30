import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Optional: uncomment below to download data directly via yfinance
# !pip install yfinance

import yfinance as yf

def download_stock_data(ticker='AAPL', period='5y'):
    print(f"Downloading {ticker} stock data...")
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)
    return df

def main():
    # 1. Load data (uncomment to download fresh data)
    # df = download_stock_data('AAPL', '5y')
    
    # Or load from CSV if you have local data:
    # df = pd.read_csv('AAPL.csv')
    
    # For demonstration, let's download
    df = download_stock_data('AAPL', '5y')

    # 2. Visualize Closing Price
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Closing Price')
    plt.title('AAPL Closing Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

    # 3. Feature Engineering
    # Convert Date to ordinal for regression (numeric)
    df['DateOrdinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df[['DateOrdinal']]
    y = df['Close']

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 5. Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Predict on Test Data
    y_pred = model.predict(X_test)

    # 7. Evaluation
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")

    # 8. Plot predictions vs actual
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'][-len(y_test):], y_test, label='Actual Price')
    plt.plot(df['Date'][-len(y_test):], y_pred, label='Predicted Price', linestyle='--')
    plt.title('Stock Price Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

    # 9. Forecast future prices (next 30 days)
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_ordinal = future_dates.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    future_preds = model.predict(future_ordinal)

    # 10. Plot forecast
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Historical Price')
    plt.plot(future_dates, future_preds, label='Forecasted Price', linestyle='--', color='red')
    plt.title('Stock Price Forecast (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
