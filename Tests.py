from Algorithms import Double_avarage
import Get_data
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime, timedelta


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def calculate_balance(data_signal):
    balance = 100
    # print("Изначальный бюджет", balance, "$")
    price = 0
    for i in range(len(data_signal)):
        if data_signal.positions.iloc[i] == 1:
            price = float(data_signal.Close.iloc[i])
        elif data_signal.positions.iloc[i] == -1 and price != 0:
            balance = balance * float(data_signal.Close.iloc[i]) / price
            balance = balance * (1 - commision)
    return balance - 100


def calculate_balance_arr(data_signal):
    balance_arr = []
    price = 0
    for i in range(len(data_signal)):
        if data_signal.positions.iloc[i] == 1:
            price = data_signal.Close.iloc[i]
        elif data_signal.positions.iloc[i] == -1 and price != 0:
            balance = balance * data_signal.Close.iloc[i] / price
            balance = balance * (1 - commision)
        balance_arr.append(balance)


def check_all_variants(data_signal, ticker):
    balance_ar = []
    sw_ar = []
    lw_ar = []
    for sw in range(5, 160, 5):
        for lw in range(20, 160, 5):
            ts = Double_avarage.double_moving_average(data_signal, sw, lw)
            data_signal["positions"] = ts['positions']
            balance = 100
            price = 0
            market_perf = float(data_signal.iloc[len(data_signal.Close) - 1]['Close']) / float(
                data_signal.iloc[1]['Close']) * 100 - 100

            for i in range(len(data_signal)):
                if data_signal.positions.iloc[i] == 1:
                    price = data_signal.Close.iloc[i]
                elif data_signal.positions.iloc[i] == -1 and price != 0:
                    balance = balance * data_signal.Close.iloc[i] / price
                    balance = balance * (1 - commision)

            if balance - 100 > 5:
                balance_ar.append(float(balance - 100))
                sw_ar.append(sw)
                lw_ar.append(lw)

    data = {'Balance': balance_ar, 'short': sw_ar, 'long': lw_ar}
    df = pd.DataFrame(data)
    df.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\" + ticker + ".csv")
    print(df.sort_values("Balance").tail(15))
    # best_variant = int(input("What variant more suitable: "))

    # return df.sort_values("Balance").iloc[best_variant]


def handle_date_str(days):
    start_date = '2021-07-16'
    if days < 7:
        start_date = start_date[:9] + str(days)
        end_date = start_date[:9] + str(days + 5)
    elif days == 7:
        start_date = start_date[:9] + str(days)
        end_date = start_date[:8] + str(days + 2)
    else:
        start_date = start_date[:8] + str(days)
        end_date = start_date[:8] + str(days + 2)

    return start_date, end_date


# tickers = "XLMBNB"  # название валюты
# 'BNBUSDT'
# 'ADABNB'
# 'XLMBNB'
commision = 0.0001

ticker = ['XLMUSDT', 'BNBUSDT', 'ADAUSDT']

'''
Analyze whole month (or maybe -2 weeks?) for 3 tickers. 
Save results to CSV
Through Power Query analyze best parameters of LW and SW
Check performance those parameters for the last week
'''

lwsw = [[85, 45], [30, 55], [90, 45], [35, 60]]

start_date = '2021-08-05'  # начало периода #1 Jun 2021
end_date = '2021-08-13'  # конец периода

for tickers in ticker:
    data_signal = Get_data.binance_data(tickers, start_date, end_date)
    # save CSV files for searching sw and lw
    # check_all_variants(data_signal, tickers)
    for swlw in lwsw:
        ts = Double_avarage.double_moving_average(data_signal, swlw[0], swlw[1])
        data_signal["positions"] = ts['positions']

        print(tickers, swlw)
        print(calculate_balance(data_signal))
