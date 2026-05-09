import yfinance as yf


def download_finance_data(ticker, start, end, interval):

    df = yf.download(ticker, start= start, end = end, interval= interval)
    df.reset_index(inplace=True)

    data_close = df.loc[:,'Close']
    data = data_close.values
    test_data = data
    train_data = data[:-40]

    return test_data, train_data




