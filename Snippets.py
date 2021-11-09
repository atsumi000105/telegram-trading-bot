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
    comm_sell = 0
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


def sort_my_stupid_txt_file():
    D30 = []
    Result = []
    ROC10 = []
    ROC30 = []
    ticker = []
    Time = []
    RSI10 = []
    RSI30 = []

    f = open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions.txt", "r")
    for line in f:
        nested_list = line.split(sep=",")
        for attrib in nested_list:
            if "D30" in attrib:
                D30.append(attrib)

            if "Result" in attrib:
                Result.append(attrib)

            if "ROC10" in attrib:
                ROC10.append(attrib)

            if "ROC30" in attrib:
                ROC30.append(attrib)

            if "ticker" in attrib:
                ticker.append(attrib)

            if "Time" in attrib:
                Time.append(attrib)

            if "RSI10" in attrib:
                RSI10.append(attrib)

            if "RSI30" in attrib:
                RSI30.append(attrib)

    data = {"D30": D30,
            "Result": Result,
            "ROC10": ROC10,
            "ROC30": ROC30,
            "ticker": ticker,
            "Time": Time,
            "RSI10": RSI10,
            "RSI30": RSI30}

    df = pd.DataFrame(data)
    df.to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\history tracker.xlsx')

def sort_my_stupid_txt_file1():
    D30 = []
    Result = []
    ROC10 = []
    ROC30 = []
    ticker = []
    Time = []
    RSI10 = []
    RSI30 = []

    f = open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions1.txt", "r")
    for line in f:
        nested_list = line.split(sep=",")
        for attrib in nested_list:
            if "D30" in attrib:
                D30.append(attrib)

            if "Result" in attrib:
                Result.append(attrib)

            if "ROC10" in attrib:
                ROC10.append(attrib)

            if "ROC30" in attrib:
                ROC30.append(attrib)

            if "ticker" in attrib:
                ticker.append(attrib)

            if "Time" in attrib:
                Time.append(attrib)

            if "RSI10" in attrib:
                RSI10.append(attrib)

            if "RSI30" in attrib:
                RSI30.append(attrib)

    data = {"D30": D30,
            "Result": Result,
            "ROC10": ROC10,
            "ROC30": ROC30,
            "ticker": ticker,
            "Time": Time,
            "RSI10": RSI10,
            "RSI30": RSI30}

    df = pd.DataFrame(data)
    df.to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\history tracker1.xlsx')