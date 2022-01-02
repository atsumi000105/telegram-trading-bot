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
import pandas as pd


def clear_records():
    with open('positions.pickle', 'rb') as handle:
        curr_dict = pickle.load(handle)

    # trnsact_hist - log file with my transactions history. Comming to txt and then to Power BI
    with open('transac_hist.pickle', 'rb') as handle:
        trnsact_hist = pickle.load(handle)

    for currency in curr_dict:
        trnsact_hist[currency] = {}
        curr_dict[currency] = 0

        with open('positions.pickle', 'wb') as handle:
            pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('transac_hist.pickle', 'wb') as handle:
        pickle.dump(trnsact_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prof_loss(sell_price, buy_price):
    ratio = sell_price / buy_price
    if ratio > 1:
        return ' прибыль ' + str(round(ratio * 100 - 100, 2)) + '%'
    else:
        return ' убыток ' + str(round(100 - ratio * 100, 2)) + '%'


def ada_AB():
    with open('positions.pickle', 'rb') as handle:
        curr_dict = pickle.load(handle)

    # trnsact_hist - log file with my transactions history. Comming to txt and then to Power BI
    with open('transac_hist.pickle', 'rb') as handle:
        trnsact_hist = pickle.load(handle)
    print(curr_dict)

    #Bot.send_msg("тут без выхода при падении", id)
    while True:
        df_with_changes_data = Snippets.market_change(online=True)
        market_status = df_with_changes_data['market_change_day'][-2]
        print("market ", market_status)

        for currency in curr_dict:
            if currency in ['TRX']: continue
            if currency not in trnsact_hist:
                trnsact_hist[currency] = {}
            # for each ticker load it's model
            with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' +
                      currency + '.pickle', 'rb') as handle:
                model = pickle.load(handle)
            with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' +
                      currency + ' columns.pickle', 'rb') as handle:
                columns = pickle.load(handle)
            recent_data = Get_data.binance_data(currency + 'USDT', '2021-12-01', print_falg=False)
            recent_data.drop(['Close Time'], axis=1, inplace=True)
            recent_data = pd.concat([recent_data, df_with_changes_data], axis=1)
            close_price = recent_data.Close.iloc[-2]

            signals = CatBoost.pred_test(model, recent_data, columns)

            # Buy
            if int(signals.positions.iloc[-2]) == 1 and \
                    curr_dict[currency] == 0:
                curr_dict[currency] = close_price

                Bot.send_msg('Покупай ' + currency + ' цена ' + str(close_price), id)
                print('Покупай ' + currency + ' цена ' + str(close_price))
                trnsact_hist[currency]["Time"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
                trnsact_hist[currency]["ticker"] = currency
            # Sell
            elif int(signals.positions.iloc[-2]) == -1 and \
                    curr_dict[currency]:
                Bot.send_msg('Продаем ' + currency + ' , цена: ' + str(close_price) + prof_loss(
                    close_price, curr_dict[currency]), id)
                print('Продаем ' + currency + ' , цена: ' + str(close_price) + prof_loss(
                     close_price, curr_dict[currency]))
                # update transaction history
                trnsact_hist[currency]["Result"] = prof_loss(
                    close_price, curr_dict[currency])
                # save transactions history
                with open(r"C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Transactions\transactions.txt", "a") as myfile:
                    myfile.write(str(trnsact_hist[currency]) + '\n')

                curr_dict[currency] = 0

            with open('positions.pickle', 'wb') as handle:
                pickle.dump(curr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('transac_hist.pickle', 'wb') as handle:
            pickle.dump(trnsact_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        msg = "Рынок в среднем " + str(round(market_status * 100, 2)) + " %"
        Bot.send_msg(msg, id)

        while True:
            if datetime.now().minute in [00]:
                time.sleep(20)
                Snippets.sort_my_stupid_txt_file()
                break
            time.sleep(58)



id = -565150126

ada_AB()
#clear_records()

