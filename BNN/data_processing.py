import yfinance as yf
import numpy as np


def download_finance_data(ticker, start, end, interval):

    df = yf.download(ticker, start= start, end = end, interval= interval)
    df.reset_index(inplace=True)

    data_list = []
    data_test_list = []

    data_close = df.loc[:,'Close']
    data = data_close.values
    test_data = data
    train_data = data[:-40]

    data_list.append(data)
    data_test_list.append(test_data)
    data_test_set = np.concatenate(data_test_list, axis=0)
    dataset = np.concatenate(data_list, axis=0)


    return dataset, data_test_set




