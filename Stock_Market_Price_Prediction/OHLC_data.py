def get_records(stock_name,start_value,end_value,interval):
    import yfinance as yf
    import warnings
    warnings.filterwarnings("ignore")
    #start and end values in YYMMDD
    # Example: Get OHLC data for Apple (AAPL)
    data = yf.download(stock_name, start=start_value, end=end_value, interval=interval)
    return data