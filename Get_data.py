
from binance.client import Client
import pandas as pd
from datetime import date
import os
from dotenv import load_dotenv

# TEST
load_dotenv()
apikey = os.getenv('apikey')
secret = os.getenv('secret')

client = Client(apikey, secret)




today = date.today()
today = today.strftime("%Y-%m-%d")

def binance_data(ticker, date_from, end_date=today):
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.BaseClient.KLINE_INTERVAL_15MINUTE
    historical = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_4HOUR, date_from, end_date)

    hist_df = pd.DataFrame(historical)

    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                       'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

    hist_df.drop(['Open Time', 'Open', 'High', 'Low', 'Volume', 'Quote Asset Volume',
                  'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'], axis=1, inplace=True)

    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time'] / 1000, unit='s')
    hist_df.set_index(hist_df['Close Time'], inplace=True)
    numeric_columns = ['Close']

    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)
    print("Data loaded", ticker, date_from, end_date)
    return hist_df



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
