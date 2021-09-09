import pandas as pd
import numpy as np
import Snippets
import time
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')


def double_moving_average(financial_data, short_window, long_window):
    signals = pd.DataFrame(index=financial_data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = financial_data['Close']. \
        rolling(window=short_window,
                min_periods=1, center=False).mean()
    signals['long_mavg'] = financial_data['Close']. \
        rolling(window=long_window,
                min_periods=1, center=False).mean()
    signals['signal'][short_window:] = \
        np.where(signals['short_mavg'][short_window:]
                 > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    signals['Close'] = financial_data['Close']
    signals['Close Time'] = financial_data['Close Time']
    return signals


def check_all_variants(data_signal):
    balance_ar = []
    sw_ar = []
    lw_ar = []
    for sw in range(10, 300, 5):
        for lw in range(10, 300, 5):
            # try:
            ts = double_moving_average(data_signal, sw, lw)
            data_signal["positions"] = ts['positions']
            balance = Snippets.calculate_balance(data_signal)
            if balance - 100 > 1:
                balance_ar.append(balance - 100)
                sw_ar.append(sw)
                lw_ar.append(lw)
            # except:
            #     continue
    data = {'Balance': balance_ar, 'short': sw_ar, 'long': lw_ar}
    df = pd.DataFrame(data)
    # df.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\" + ticker + ".csv")
    return df


def get_lw_sw(start_point, end_point, ada, bnb):


    ada = ada.iloc[start_point:end_point, :]
    ada_var = check_all_variants(ada)

    ada_var['Key'] = ada_var['short'].astype(str) + ada_var['long'].astype(str)


    bnb = bnb.iloc[start_point:end_point, :]
    bnb_var = check_all_variants(bnb)

    bnb_var['Key'] = bnb_var['short'].astype(str) + bnb_var['long'].astype(str)

    data = pd.merge(ada_var, bnb_var, on='Key')
    data["Sum_Bal"] = data['Balance_x'] + data['Balance_y'] #+ data['Balance']
    sw = data.sort_values("Sum_Bal").iloc[-1, 1]
    lw = data.sort_values("Sum_Bal").iloc[-1, 2]
    return sw, lw


def get_results(sw, lw, start_poimt, end_point, ada, bnb):

    ada = ada.iloc[start_poimt:end_point, :]

    data_signal = double_moving_average(ada, sw, lw)


    startdate_arr.append(ada.iloc[0, 2])
    enddate_arr.append(ada.iloc[-1, 2])
    close_arr.append(ada.iloc[0, 1])
    balance_arr.append(100-Snippets.calculate_balance(data_signal))
    sw_arr.append(sw)
    lw_arr.append(lw)
    past_arr.append(past_interval)
    ticker_arr.append('ADA')

    bnb = bnb.iloc[start_poimt:end_point, :]

    data_signal = double_moving_average(bnb, sw, lw)

    startdate_arr.append(bnb.iloc[0, 2])
    enddate_arr.append(bnb.iloc[-1, 2])
    close_arr.append(bnb.iloc[0, 1])
    balance_arr.append(100-Snippets.calculate_balance(data_signal))
    sw_arr.append(sw)
    lw_arr.append(lw)
    past_arr.append(past_interval)
    ticker_arr.append('BNB')


ada = pd.read_csv("C:\\Users\\Vlad\Desktop\\Finance\\ADA 15 min.csv")
bnb = pd.read_csv("C:\\Users\\Vlad\Desktop\\Finance\\BNB 15 min.csv")


close_arr = []
startdate_arr = []
enddate_arr = []
balance_arr = []
sw_arr = []
lw_arr = []
ticker_arr = []
past_arr = []

# train_data = bnb.iloc[:round(len(bnb)*0.8), :]
# test_data = bnb.iloc[round(len(bnb)*0.8):, :]
#
# ada_var = check_all_variants(train_data)
#


future_interval = 7*96
for past_interval in range(96, 1920, 96):
    for start_range in range(0, len(ada), future_interval):
        print(start_range/len(ada)*100)
        try:
            sw, lw = get_lw_sw(start_range, start_range + past_interval, ada, bnb)
            get_results(sw, lw, past_interval + start_range, past_interval + start_range + future_interval, ada, bnb)
        except:
            continue


final_data = {'StartDate': startdate_arr, 'Balance': balance_arr,
              'SW': sw_arr, 'LW': lw_arr, 'Ticker': ticker_arr, 'Close': close_arr}

final_df = pd.DataFrame(final_data)
final_df.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\EMA15 " + str(future_interval/96) + " days.csv")


df = check_all_variants(ada)