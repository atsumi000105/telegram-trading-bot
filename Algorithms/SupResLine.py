import pandas as pd
import numpy as np


def trading_support_resistance(data, bin_width, resistance_per, support_per):
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(data)):
        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['price'])
        resistance_level = max(data_section['price'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['res_tolerance'][x] <= data['price'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['sup_tolerance'][x] >= data['price'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0
        if in_resistance > resistance_per:
            data['signal'][x] = 1
        elif in_support > support_per:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x - 1]
    data['positions'] = data['signal'].diff()
    return data


from Algorithms import Double_avarage
import Get_data
import matplotlib.pyplot as plt
import time
from Telegram import Bot
from datetime import date


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
    return balance


today = date.today()
today = today.strftime("%Y-%m-%d")
# тут можешь изменять данные
start_date = '2021-08-01'  # начало периода #1 Jun 2021
# end_date = today # конец периода
tickers = "ADABNB"  # название валюты ADABNB
# ticker = ['XLMBNB','BNBUSDT','ADABNB']

commision = 0.0001

data_signal = Get_data.binance_data(tickers, start_date)
data_signal['price'] = data_signal['Close']
def check_variants_supres():
    bin_width_arr = []
    r_arr = []
    s_arr = []
    balance_arr = []
    for bin_width in range(5, 120, 5):
        print(bin_width)
        for r in range(2, 20):
            for s in range(2, 20):
                test_data = data_signal.loc[:, :]
                resistance_per = r
                support_per = s
                test_data = trading_support_resistance(test_data, bin_width, r, s)
                bin_width_arr.append(bin_width)
                r_arr.append(r)
                s_arr.append(s)
                balance_arr.append(calculate_balance(test_data))
    data = {'Bin': bin_width_arr, 'Resistance': r_arr, 'Support': s_arr, 'Balance': balance_arr}
    df = pd.DataFrame(data)
    df.sort_values("Balance").tail(20)

trading_support_resistance(data_signal, 15, 2, 4)
'''
1317   25           3        5  104.359818
3629   60           5       13  104.366921
3630   60           5       14  104.392075
21      5           3        5  104.431160
974    20           2        4  104.437269
1318   25           3        6  104.568096
343    10           3        3  104.626934
973    20           2        3  104.911396
667    15           3        3  104.952986
650    15           2        4  105.169082
'''