
from binance.client import Client
import pandas as pd
from datetime import date
import os
from dotenv import load_dotenv
import time
# TEST
load_dotenv()
apikey = os.getenv('apikey')
secret = os.getenv('secret')

client = Client(apikey, secret)

def binance_data(ticker, date_from, end="", print_falg=True ):
    hist_df = pd.DataFrame()
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.BaseClient.KLINE_INTERVAL_15MINUTE
    while len(hist_df) < 1:
        try:
            if end: historical = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, date_from, end)
            else: historical = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, date_from)
            hist_df = pd.DataFrame(historical)
        except:
            print('Wasnt able to download the data for ', ticker)
            time.sleep(20)
            continue
    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                       'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

    hist_df.drop(['Open Time', 'Quote Asset Volume',
                  'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'], axis=1, inplace=True)

    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time'] / 1000, unit='s')
    hist_df.set_index(hist_df['Close Time'], inplace=True)
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)
    if print_falg: print("Data loaded", ticker, date_from)
    return hist_df

# tickers = ['TRX','LTC','ATOM','FTM','LUNA']
#
#
# for ticker in tickers:
#     df = binance_data(ticker+'USDT', '2021-03-01', end='2021-09-01')
#     df.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\Raw data\\" + ticker + " 1hr.csv")
# tickers = ['ADA', 'BAT', 'BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET', 'XLM', 'XMR', 'XRP', 'ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', 'TRX']
# for ticker in tickers:
#     df = binance_data(ticker+'USDT', '2021-09-20', end='2021-10-28')
#     df.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\Verif\\" + ticker + " 1hr.csv")

# from selenium import webdriver
# from datetime import datetime
#
# driver = webdriver.Chrome(r'C:\Users\Vlad\Downloads\chromedriver.exe')
# driver.get('https://www.binance.com/en/trade/ADA_USDT?theme=dark&type=spot')

# def scrap_latest_data():
#     try:
#         driver.find_element_by_xpath('/ html / body / div[5] / div[2] / div[1] / svg').click()
#     except:
#         pass
#     price = driver.find_element_by_xpath(
#         '//*[@id="__APP"]/div/div/div[6]/div/div/div[2]/div/div/div[2]/div[1]/div/div/div[1]/div/div[1]').text
#     now = datetime.now()
#
#     df = pd.DataFrame(columns=['Close', 'Close Time'])
#     df.loc[0] = list([price, now])
#     return df

