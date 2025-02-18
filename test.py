import yfinance as yf

# Fetch stock data for Apple (AAPL)
ticker = "KO"
stock = yf.Ticker(ticker)

# Fetch historical market data
hist = stock.history(period="5d")

# Display the fetched data
print(hist)
