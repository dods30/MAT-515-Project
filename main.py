import historical_data
import preliminary_analysis
import portfolio_analysis

# define the symbols to include in analysis 
stocks = ['TSLA','AMD','SOFI','WBA','MDLZ','AAPL','NVDA','PLTR','SNOW','AMZN']

# fetch and save the data 
historical_data.fetch_and_save(stocks)

# perform preliminary analysis
preliminary_analysis.analyze(stocks)

# perform portfolio analysis
portfolio_analysis.optimize_portfolio(stocks)

