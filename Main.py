# Load libraries

import pickle

import numpy as np
import pandas as pd

import Get_data
from datetime import datetime
import time

import Snippets
from ML_finance import CatBoost
from Binance import Binance
account = Binance.Binance_acc()

from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')



def prof_loss(sell_price, buy_price):
    ratio = sell_price / buy_price
    if ratio > 1:
        return ' прибыль ' + str(round(ratio * 100 - 100, 2)) + '%'
    else:
        return ' убыток ' + str(round(100 - ratio * 100, 2)) + '%'


def sell_all():
    with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'rb') as handle:
        curr_dict = pickle.load(handle)
    print(curr_dict)
    for currency in curr_dict:
        if curr_dict[currency]:
            account.sell(symbol=currency)
            curr_dict[currency] = 0

        with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'wb') as handle:
            pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(60 * 60 * 24)


def ada_AB():
    df = pd.DataFrame()
    order_limit = 20
    avg_chnage = list()
    with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'rb') as handle:
        curr_dict = pickle.load(handle)
    print(curr_dict)

    # curr_dict = dict()
    # for f in files:
    #     curr_dict[f[:len(f) - 8]] = 0
    # curr_dict['DOT'] = 0
    # curr_dict['BNB'] = 0
    # with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'wb') as handle:
    #     pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    while True:

        avg_chnage.clear()
        for currency in curr_dict:
            if currency in ['ATOM', 'XLM', 'BEAM']: continue
            with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\ML_finance\\Model_by_model\\Models2\\' + currency + ".pickle", 'rb') as handle:
                model = pickle.load(handle)
            with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' +
                      currency + ' columns.pickle', 'rb') as handle:
                columns = pickle.load(handle)

            recent_data = Get_data.binance_data(currency + 'USDT', '2021-11-10', print_falg=False)
            closed_time = recent_data['Close Time'][-2]
            recent_data.drop(['Close Time'], axis=1, inplace=True)
            close_price = recent_data.Close.iloc[-2]
            signals = CatBoost.pred_test(model, recent_data, columns)

            avg_chnage.append((recent_data.Close.iloc[-1]/recent_data.Close.iloc[-24])-1)

            df["time"] = [closed_time]
            df[currency] = [0]


            # BUY
            if int(signals.positions.iloc[-2]) == 1 and \
                    curr_dict[currency] == 0 and \
                    account.get_acc_balance() > order_limit:

                curr_dict[currency] = close_price
                qty = round(order_limit/close_price, account.get_decimal(currency))
                account.buy(symbol=currency + "USDT", quantity=qty)
                df[currency] = [1]
            # SELL
            if int(signals.positions.iloc[-2]) == -1 and \
                    curr_dict[currency]:

                try:
                    account.sell(symbol=currency)
                    curr_dict[currency] = 0
                    df[currency] = [-1]
                except:
                    print('WASNT ABLE TO SELL', currency)
                    continue

            #save current possitions. Unable to do this from Binance, as i have a lot of currencies as leftovers
            with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'wb') as handle:
                pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # In case market is down, better to sell everythong. Bot cannot handle this
        np_avg =  np.asarray(avg_chnage, dtype=np.float32).mean()
        df['Total Status'] = round(np_avg*100, 3)
        Snippets.append_to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\tracking status.xlsx', df)
        #df.to_excel('C:\\Users\\Vlad\\Desktop\\Finance\\tracking status.xlsx')
        # print 24h market avarage price change. Helping to look at market situation - bear or bull
        print(round(np_avg*100, 3), "%")
        if np_avg < -0.04: sell_all()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        #run script only when new hour starts So we have full data for previous hour.
        while True:
            if datetime.now().minute in [00]:
                time.sleep(20)
                break
            time.sleep(58)

#todo
# RSI > 40. Otherwise sell everything

id = -467554548

ada_AB()

#sell_all()

