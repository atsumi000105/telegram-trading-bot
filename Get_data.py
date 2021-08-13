#!pip install python-binance mplfinance
from binance.client import Client
import pandas as pd

# Prod
# apikey = 'e1FmlgmhGU2qbaBIMy3TifENlG4esqfBPkfVp6LOxABbMCzkLxjrszY5pKOhxiUj'
# secret = 'i8nAXkmxhMbAkAlsCeX8oF8rSIr3f2B9eyWBT2EuECS3zw8vc4BDiYXaV7WB7X81'
# TEST
apikey = 'e1FmlgmhGU2qbaBIMy3TifENlG4esqfBPkfVp6LOxABbMCzkLxjrszY5pKOhxiUj'
secret = 'i8nAXkmxhMbAkAlsCeX8oF8rSIr3f2B9eyWBT2EuECS3zw8vc4BDiYXaV7WB7X81'

client = Client(apikey, secret)


# client.API_URL = 'https://testnet.binance.vision/api'
# print(client.get_account())
# print(client.get_asset_balance(asset='ETH'))

# timestamp = client._get_earliest_valid_timestamp('ADAUSDT', '15m')
# bars = client.get_historical_klines('ADAUSDT', '15m', timestamp, limit=1000)
# for line in bars:
#     del line[5:]
#     del line[1:4]
# btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
# btc_df['Close Time'] = pd.to_datetime(btc_df.date / 1000, unit='s')
# print(btc_df.tail())


def binance_data(ticker, date_from, end_date=""):
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.BaseClient.KLINE_INTERVAL_15MINUTE
    historical = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_15MINUTE, date_from)

    hist_df = pd.DataFrame(historical)

    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                       'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

    hist_df.drop(['Open Time', 'Open', 'High', 'Low', 'Volume', 'Quote Asset Volume',
                  'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore'], axis=1, inplace=True)

    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time'] / 1000, unit='s')

    numeric_columns = ['Close']

    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)
    print("Data loaded")
    return hist_df


# from selenium import webdriver
# from datetime import datetime
#
# driver = webdriver.Chrome(r'C:\Users\Vlad\Downloads\chromedriver.exe')
# driver.get('https://www.binance.com/en/trade/ADA_USDT?theme=dark&type=spot')

def scrap_latest_data():
    try:
        driver.find_element_by_xpath('/ html / body / div[5] / div[2] / div[1] / svg').click()
    except:
        pass
    price = driver.find_element_by_xpath(
        '//*[@id="__APP"]/div/div/div[6]/div/div/div[2]/div/div/div[2]/div[1]/div/div/div[1]/div/div[1]').text
    now = datetime.now()

    df = pd.DataFrame(columns=['Close', 'Close Time'])
    df.loc[0] = list([price, now])
    return df
