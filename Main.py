# Load libraries

import pickle
import Get_data
from datetime import datetime
import time
from ML_finance import CatBoost
from Binance import Binance
account = Binance.Binance_acc()
# Libraries for Deep Learning Models
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

def ada_AB():

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

        for currency in curr_dict:
            if currency in ['LUNA','ATOM','LTC','TRX','MATIC','ALGO','LINK', 'ADA', 'BAT']: continue
            with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\ML_finance\\Models\\' + currency + ".pickle", 'rb') as handle:
                model = pickle.load(handle)
            recent_data = Get_data.binance_data(currency + 'USDT', '2021-09-29', print_falg=True)
            recent_data.drop(['Close Time'], axis=1, inplace=True)
            signals = CatBoost.pred_real(model, recent_data)
            close_price = signals.Close.iloc[-1]

            order_limit = 20


            # BUY
            # if int(signals.positions.iloc[-1]) == 1 and \
            #         curr_dict[currency] == 0 and \
            #         40 < signals['%D30'][-1] < 90 and \
            #         -1 < signals['ROC30'][-1] < 4 and \
            #         0 < signals['ROC10'][-1] < 4 and \
            #         float(client.get_asset_balance('USDT')['free']) > order_limit:

            # if int(signals.positions.iloc[-1]) == 1 and \
            #         curr_dict[currency] == 0 and \
            #         0 < signals['ROC10'][-1] < 4 and \
            #         account.get_acc_balance() > order_limit:
            #
            #     curr_dict[currency] = signals.Close.iloc[-1]
            #     qty = round(order_limit/close_price, account.get_decimal(currency))
            #     account.buy(symbol=currency + "USDT", quantity=qty)

            # SELL
            if int(signals.positions.iloc[-1]) == -1 and \
                    curr_dict[currency]:

                try:
                    account.sell(symbol=currency + "USDT")
                    curr_dict[currency] = 0
                except:
                    print('WASNT ABLE TO SELL', currency)
                    continue

            with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\Live\\positionsLIVE.pickle', 'wb') as handle:
                pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        while True:
            if datetime.now().minute in [00]:
                time.sleep(20)
                break
            time.sleep(58)

#todo
# RSI > 40. Otherwise sell everything

id = -467554548

#ada_AB()

sell_all()

