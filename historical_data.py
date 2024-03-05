import yfinance as yf

# Define a function to fetch and save historical stock data
def fetch_and_save(stocks):
    # Loop through each stock symbol in the provided list
    for stock in stocks:
        # Download historical data from start date to end date for the given stock
        data = yf.download(stock, start='2020-02-27', end='2024-02-27')
        # Save the data to a CSV file named after the stock symbol
        data.to_csv(f'{stock}_data.csv')

