import Get_data

import pandas as pd
from binance.client import Client
import os
from dotenv import load_dotenv
load_dotenv()
apikey = os.getenv('apikey')
secret = os.getenv('secret')
client = Client(apikey, secret)
#.loc[row_indexer,col_indexer]



# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))



def calculate_balance(data, commission=0.001, array=False):
    balance = 100
    comm_buy = 0
    #print("Изначальный бюджет", balance, "$")
    price_buy = 0
    price_sell = 0
    balance_arr = []
    for i in range(len(data)):
        if data.positions.iloc[i] == 1:
            price_buy = float(data.Close.iloc[i])
            comm_buy = balance*commission
        elif data.positions.iloc[i] == -1 and price_buy != 0:
            price_sell = float(data.Close.iloc[i])
            balance = balance*price_sell/price_buy*(1-commission)-comm_buy
        balance_arr.append(balance)
    if array:
        return balance_arr
    else:
        return round(balance, 2)

def check_decimals(symbol):
    info = client.get_symbol_info(symbol)
    val = info['filters'][2]['stepSize']
    decimal = 0
    is_dec = False
    for c in val:
        if is_dec is True:
            decimal += 1
        if c == '1':
            break
        if c == '.':
            is_dec = True
    return decimal

def append_to_excel(fpath, df):
    old_df = pd.read_excel(fpath)
    new_df = pd.concat([old_df, df],ignore_index=True)
    new_df.to_excel(fpath)


def sort_my_stupid_txt_file():

    Result = []
    ticker = []
    Time = []

    f = open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions.txt", "r")
    for line in f:
        nested_list = line.split(sep=",")
        for attrib in nested_list:

            if "Result" in attrib:
                Result.append(attrib)

            if "ticker" in attrib:
                ticker.append(attrib)

            if "Time" in attrib:
                Time.append(attrib)


    data = {"Result": Result,
            "ticker": ticker,
            "Time": Time}

    df = pd.DataFrame(data)
    df.to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\history tracker.xlsx')

def sort_my_stupid_txt_file1():
    Result = []
    ticker = []
    Time = []


    f = open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions1.txt", "r")
    for line in f:
        nested_list = line.split(sep=",")
        for attrib in nested_list:

            if "Result" in attrib:
                Result.append(attrib)

            if "ticker" in attrib:
                ticker.append(attrib)

            if "Time" in attrib:
                Time.append(attrib)

    data = {
            "Result": Result,
            "ticker": ticker,
            "Time": Time,
            }

    df = pd.DataFrame(data)
    df.to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\history tracker1.xlsx')


def market_change(path = "", online=False):

    tickers = ['BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET', 'XLM', 'XMR', 'XRP',
               'ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', 'TRX']

    tf_list = list()

    #24 hors
    for ticker in tickers:
        if online:
            data = Get_data.binance_data(ticker + 'USDT', '2021-12-01', print_falg=False)
        else:
            data = pd.read_csv(path + ticker + ' 1hr.csv')
        if ticker == "BEAM":
            df = pd.DataFrame(data['Close'].pct_change(periods=24))
        else:
            df[ticker]= data['Close'].pct_change(periods=24)

    tf_list.append(df.mean(axis=1))

    # 10 hours
    for ticker in tickers:
        if online:
            data = Get_data.binance_data(ticker + 'USDT', '2021-12-01', print_falg=False)
        else:
            data = pd.read_csv(path + ticker + ' 1hr.csv')
        if ticker == "BEAM":
            df = pd.DataFrame(data['Close'].pct_change(periods=10))
        else:
            df[ticker]= data['Close'].pct_change(periods=10)

    df['market_change_ten'] = df.mean(axis=1)
    df['market_change_day'] = tf_list[0]


    return df[['market_change_day', 'market_change_ten']]