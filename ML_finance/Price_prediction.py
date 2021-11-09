# Load libraries
import os
import pickle
import Snippets
import Indicators
import Get_data
from Telegram import Bot
from datetime import datetime, timedelta, date
import time
from ML_finance import CatBoost

def prof_loss(sell_price, buy_price):
    ratio = sell_price / buy_price
    if ratio > 1:
        return ' прибыль ' + str(round(ratio * 100 - 100, 2)) + '%'
    else:
        return ' убыток ' + str(round(100 - ratio * 100, 2)) + '%'

def ada_AB():
    current_tickers = ['ADA', 'BAT', 'BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET']
    with open('positions1.pickle', 'rb') as handle:
        curr_dict = pickle.load(handle)

    with open('transac_hist1.pickle', 'rb') as handle:
        trnsact_hist = pickle.load(handle)
    print(curr_dict)
    # curr_dict = dict()
    # for f in files:
    #     curr_dict[f[:len(f) - 8]] = 0

    while True:

        # load files, just to get current active tickers
        for currency in curr_dict:
            if currency not in trnsact_hist:
                trnsact_hist[currency] = {}
            if currency in current_tickers:
                # for each ticker load it's model
                with open('C:\\Users\\Vlad\\PycharmProjects\\Time-Series-Analysis\\ML_finance\\Model_by_model\\Models2\\'+currency + ".pickle", 'rb') as handle:
                    model = pickle.load(handle)
                recent_data = Get_data.binance_data(currency + 'USDT', '2021-10-10', print_falg=False)
                recent_data.drop(['Close Time'], axis=1, inplace=True)

                close_price = recent_data.Close.iloc[-1]
                signals = CatBoost.pred_test(model, recent_data, currency)



                # Buy
                if int(signals.positions.iloc[-1]) == 1 and \
                        curr_dict[currency] == 0:
                    curr_dict[currency] = close_price

                    # Bot.send_msg('Покупай ' + currency + ' цена ' + str(close_price), id)
                    print('Покупай ' + currency + ' цена ' + str(close_price))
                    trnsact_hist[currency]["Time"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
                    trnsact_hist[currency]["ROC10"] = signals['%D30'][-1]
                    trnsact_hist[currency]["ROC30"] = signals['ROC30'][-1]
                    trnsact_hist[currency]["D30"] = signals['ROC10'][-1]
                    trnsact_hist[currency]["ticker"] = currency
                    trnsact_hist[currency]["RSI10"] = signals['RSI10'][-1]
                    trnsact_hist[currency]["RSI30"] = signals['RSI30'][-1]

                # Sell
                elif int(signals.positions.iloc[-1]) == -1 and \
                        curr_dict[currency]:
                    # Bot.send_msg('Продаем ' + currency + ' , цена: ' + str(close_price) + prof_loss(
                    #     close_price, curr_dict[currency]), id)
                    print('Продаем ' + currency + ' , цена: ' + str(close_price) + prof_loss(
                         close_price, curr_dict[currency]))
                    # update transaction history
                    trnsact_hist[currency]["Result"] = prof_loss(
                        close_price, curr_dict[currency])
                    # save transactions history
                    with open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions1.txt", "a") as myfile:
                        myfile.write(str(trnsact_hist[currency]) + '\n')

                    curr_dict[currency] = 0

                with open('positions1.pickle', 'wb') as handle:
                    pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('transac_hist1.pickle', 'wb') as handle:
            pickle.dump(trnsact_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        while True:
            if datetime.now().minute in [00, 30]:
                time.sleep(20)
                Snippets.sort_my_stupid_txt_file1()
                break
            time.sleep(58)



id = -467554548

ada_AB()

