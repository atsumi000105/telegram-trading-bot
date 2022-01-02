# Load libraries
import os
import pickle
import random

import Get_data
import Snippets
import Indicators
from ML_finance.Model_by_model import Indicators_all

from datetime import datetime, timedelta, date
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')


def train_model(df, model_init):
    dataset = df.copy()

    # # Create short simple moving average over the short window
    # dataset['short_mavg'] = dataset['Close'].rolling(window=35, min_periods=1, center=False).mean()
    # #
    # # # Create long simple moving average over the long window
    # dataset['long_mavg'] = dataset['Close'].rolling(window=75, min_periods=1, center=False).mean()
    #
    # # Create signals
    # dataset['positions'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 0.0, 1.0)
    # *********************
    # dataset['Close2'] = dataset['Close'].shift(1)
    #
    #
    #
    # # dataset['short_mavg'] = dataset['Close'].rolling(window=20, min_periods=1, center=False).mean()
    #
    #
    # dataset['positions'] = np.where(dataset['Close'] > dataset['Close2'], 0, 1)
    # #dataset.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    # dataset.drop(['Close2'], axis=1, inplace=True)
    #
    # # dataset.drop(['short_mavg', 'long_mavg', 'Close Time.1'], axis=1, inplace=True)
    #
    # dataset = add_indicators(dataset)
    #
    # dataset = dataset.dropna(axis=0)
    # *********************
    #    dataset = dataset.drop(['Close Time', 'Close Time.1'], axis=1)

    # split out validation dataset for the end
    # subset_dataset= dataset.iloc[-round(len(dataset)*0.2):]
    # subset_dataset.drop(['positions'], axis=1, inplace=True)
    # subset_dataset.drop(['positions'], axis=1, inplace=True)

    Y = dataset["positions"]
    X = dataset.loc[:, dataset.columns != 'positions']
    validation_size = 0.2

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=1)

    # test_models()

    # model =  RandomForestClassifier(criterion='gini', max_depth=30,  n_jobs=-1, n_estimators=150)
    # model = LinearDiscriminantAnalysis()
    # model = GradientBoostingClassifier()
    # model = AdaBoostClassifier(n_estimators=n_estimators,)

    model = model_init
    model.fit(X, Y)
    # model.fit(X_train, Y_train)
    # estimate accuracy on validation set
    # predictions = model.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))

    return model


def add_indicators(dataset):
    # calculation of rate of change
    def ROC(df, n):
        M = df.diff(n - 1)
        N = df.shift(n - 1)
        ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
        return ROC

    dataset['ROC10'] = ROC(dataset['Close'], 10)
    dataset['ROC30'] = ROC(dataset['Close'], 30)
    dataset['ROC50'] = ROC(dataset['Close'], 50)
    dataset['ROC100'] = ROC(dataset['Close'], 100)
    dataset['ROC150'] = ROC(dataset['Close'], 150)

    # Calculation of price momentum
    def MOM(df, n):
        MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
        return MOM

    dataset['MOM10'] = MOM(dataset['Close'], 10)
    dataset['MOM30'] = MOM(dataset['Close'], 30)
    dataset['MOM50'] = MOM(dataset['Close'], 50)
    dataset['MOM100'] = MOM(dataset['Close'], 100)
    dataset['MOM150'] = MOM(dataset['Close'], 150)

    # calculation of relative strength index
    dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
    dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
    dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
    dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)
    dataset['RSI200'] = Indicators.RSI(dataset['Close'], 200)

    # calculation of stochastic osillator.

    def STOK(close, low, high, n):
        STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
        return STOK

    def STOD(close, low, high, n):
        STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
        STOD = STOK.rolling(3).mean()
        return STOD

    dataset['%K10'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
    dataset['%D10'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
    dataset['%K30'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
    dataset['%D30'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)

    dataset['%K50'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 50)
    dataset['%D50'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 50)
    dataset['%K90'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 90)
    dataset['%D90'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 90)

    dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
    dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

    # Calculation of moving average

    dataset['SMA10'] = Indicators.SMA(dataset['Close'], 10) / dataset['Close']
    dataset['SMA20'] = Indicators.SMA(dataset['Close'], 20) / dataset['Close']
    dataset['SMA55'] = Indicators.SMA(dataset['Close'], 55) / dataset['Close']
    dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90) / dataset['Close']
    dataset['SMA155'] = Indicators.SMA(dataset['Close'], 155) / dataset['Close']

    # calculation of exponential moving average
    dataset['EMA20'] = Indicators.EMA(dataset['Close'], 20) / dataset['Close']
    dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
    dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
    dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']

    dataset['Low'] = dataset['Low'] / dataset['Close']
    dataset['High'] = dataset['High'] / dataset['Close']
    dataset['Open'] = dataset['Open'] / dataset['Close']

    dataset.drop(['Volume', 'Low', 'High', 'Open'], axis=1, inplace=True)

    dataset = dataset.dropna(axis=0)

    return dataset

    # predictions = model.predict(X_validation)
    # print(accuracy_score(Y_validation, predictions))
    # print(confusion_matrix(Y_validation, predictions))
    # print(classification_report(Y_validation, predictions))

    # predictions = model.predict(subset_dataset)
    # subset_dataset['predictions'] = predictions
    # subset_dataset['positions'] = subset_dataset['predictions'].diff()
    # print(100-Snippets.calculate_balance(subset_dataset))


def pred(model, dataset, temp_close):
    df = dataset.copy()
    # dataset = Indicators_new.ADA(df)
    # dataset = add_indicators(df)

    # dataset = dataset.dropna(axis=0)

    predictions = model.predict(df)
    df['predictions'] = predictions
    df['positions'] = df['predictions'].diff()
    df['Close'] = temp_close

    # print(dataset.tail(10))
    # dataset = dataset.tail(1)
    # print(100 - Snippets.calculate_balance(dataset))

    # Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=X.columns)
    # Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r' )
    # plt.xlabel('Variable Importance')

    # fig = plt.figure()
    # fig.set_size_inches(22.5, 10.5)
    # ax1 = fig.add_subplot(111, ylabel='Google price in $')
    # dataset["Close"].plot(ax=ax1, color='g', lw=.5)
    #
    # ax1.plot(dataset.loc[dataset.positions == 1.0].index, dataset["Close"][dataset.positions == 1.0],
    #          '^', markersize=7, color='k')
    #
    # ax1.plot(dataset.loc[dataset.positions == -1.0].index, dataset["Close"][dataset.positions == -1.0],
    #          'v', markersize=7, color='k')
    #
    # plt.legend(["Price", "Buy", "Sell"])
    # plt.title("AI ADA pred")
    #
    # plt.show()

    return df


def get_random_columns():
    # first two - market changes
    cols = [1, 1, random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random()]
    colmns_select = []
    for count, column in enumerate(cols):
        if column > 0.5: colmns_select.append(count)
    return colmns_select


def market_change(path="", online=False):
    tickers = ['BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET', 'XLM', 'XMR', 'XRP',
               'ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', 'TRX']

    tf_list = list()

    # 24 hors
    for ticker in tickers:
        if online:
            data = Get_data.binance_data(ticker + 'USDT', '2021-12-15')
        else:
            data = pd.read_csv(path + ticker + ' 1hr.csv')
        if ticker == "BEAM":
            df = pd.DataFrame(data['Close'].pct_change(periods=24))
        else:
            df[ticker] = data['Close'].pct_change(periods=24)

    tf_list.append(df.mean(axis=1))

    # 10 hours
    for ticker in tickers:
        if online:
            data = Get_data.binance_data(ticker + 'USDT', '2021-12-15')
        else:
            data = pd.read_csv(path + ticker + ' 1hr.csv')
        if ticker == "BEAM":
            df = pd.DataFrame(data['Close'].pct_change(periods=10))
        else:
            df[ticker] = data['Close'].pct_change(periods=10)

    df['market_change_ten'] = df.mean(axis=1)
    df['market_change_day'] = tf_list[0]

    return df[['market_change_day', 'market_change_ten']]


# tickers = ['BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'SOL', 'VET', 'XLM', 'XMR', 'XRP']
# tickers = ['ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', "XLM", "TRX"]
# tickers = ['BEAM']
# date for verif - '2021-12-15'
# date for training '2021-10-28', end='2021-12-15')
df_with_changes_data = market_change('C:\\Users\\Vlad\\Desktop\\Finance\\Raw data\\')
df_with_changes_recent = market_change('C:\\Users\\Vlad\\Desktop\\Finance\\Verif\\')
df_with_changes_verif = market_change(online=True)

# cat_class = [LinearDiscriminantAnalysis(),
#              CatBoostClassifier(logging_level='Silent'),
#              KNeighborsClassifier(),
#              GaussianNB()]

cat_class = [LinearDiscriminantAnalysis(shrinkage="auto", solver='lsqr'), QuadraticDiscriminantAnalysis()]
# tickers = ['ADA', 'ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', "XLM", "TRX",
#            'ETH', 'BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'SOL', 'VET', 'XLM', 'XMR', 'XRP']
tickers = ["BAT", "XMR", "LUNA", "FTM", "DOT", "AVAX", "ATOM", "XRP", "MATIC", "LUNA"]
# ml = cat_class[0]
# colmns_select = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 18, 22, 25, 27, 28, 30, 31, 32, 33, 34, 35]


for ticker in tickers:
    iter_arr = []
    possitionsv_arr = []
    possitionsr_arr = []
    ticker_arr = []
    recent_balance = []
    verif_balance = []
    model_arr = []
    columns_arr = []

    # load my initial data for training
    data = pd.read_csv('C:\\Users\\Vlad\\Desktop\\Finance\\Raw data\\' + ticker + ' 1hr.csv')

    # add columns with 24 and 10 period changes on avarage between all my tickers together
    data = pd.concat([data, df_with_changes_data], axis=1)
    data = data.set_index('Close Time')
    data['Close2'] = data['Close'].shift(1)
    positions = np.where(data['Close'] > data['Close2'], 0, 1)

    data.drop(['Close2', 'Close Time.1'], axis=1, inplace=True)
    data = add_indicators(data)
    data = data.loc['2021-06-28':]
    # some indicators are based on 'Closed', so it should be removed after add_indicators()
    data.drop(['Close'], axis=1, inplace=True)

    # load data with period 20 Sep to 28 Oct. We will check first performance on this set
    # You can call it Test_set :)
    recent_data1 = pd.read_csv(
        'C:\\Users\\Vlad\\Desktop\\Finance\\Verif\\' + ticker + ' 1hr.csv')
    recent_data1 = pd.concat([recent_data1, df_with_changes_recent], axis=1)
    recent_data1.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)
    recent_data1 = add_indicators(recent_data1)
    recent_close = recent_data1['Close']
    recent_data1.drop(['Close'], axis=1, inplace=True)

    # load most recent data and see how algo would behave.
    verif_data1 = Get_data.binance_data(ticker + 'USDT', '2021-12-15')
    verif_data1 = pd.concat([verif_data1, df_with_changes_verif], axis=1)
    verif_data1.drop(['Close Time'], axis=1, inplace=True)
    verif_data1 = add_indicators(verif_data1)
    verif_close = verif_data1['Close']
    verif_data1.drop(['Close'], axis=1, inplace=True)

    # data1 = data.iloc[:, colmns_select]
    # data1['positions'] = positions[2850:]
    # #data1 = data1.iloc[i:, ]
    # model = train_model(data1, ml)
    # with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' + ticker +'.pickle',
    #           'wb') as handle:
    #     pickle.dump(model, handle)
    # print("model has been saved")
    # recent_data = recent_data1.iloc[:, colmns_select].copy()
    # verif_data = verif_data1.iloc[:, colmns_select].copy()
    # signalsr = pred(model, recent_data, recent_close)
    # print("Nr of transactions ", len(signalsr[signalsr['positions'] == 1]))
    # print("Balance for recent data ", Snippets.calculate_balance(signalsr) - 100)
    # signals = pred(model, verif_data, verif_close)
    # print("Balance for verif data ", Snippets.calculate_balance(signals) - 100)
    # print("Nr of transactions ", len(signals[signals['positions'] == 1]))
    # with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' + ticker +' columns.pickle',
    #           'wb') as handle:
    #     pickle.dump(colmns_select, handle)


    with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' +
              ticker + '.pickle', 'rb') as handle:
        model = pickle.load(handle)
    with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' +
              ticker + ' columns.pickle', 'rb') as handle:
        colmns_select = pickle.load(handle)
    recent_data = recent_data1.iloc[:, colmns_select].copy()
    verif_data = verif_data1.iloc[:, colmns_select].copy()
    signalsr = pred(model, recent_data, recent_close)
    #print("Nr of transactions ", len(signalsr[signalsr['positions'] == 1]))
    print("Balance for recent data ", Snippets.calculate_balance(signalsr) - 100)
    signals = pred(model, verif_data, verif_close)
    print("Balance for verif data ", Snippets.calculate_balance(signals) - 100)
    print("Nr of transactions ", len(signals[signals['positions'] == 1]))

    # for random_count in range(1, 500):
    #     colmns_select = get_random_columns()
    #     data1 = data.iloc[:, colmns_select].copy()
    #     data1['positions'] = positions[2850:]
    #
    #     recent_data = recent_data1.iloc[:, colmns_select].copy()
    #     verif_data = verif_data1.iloc[:, colmns_select].copy()
    #
    #     for ml in cat_class:
    #         model = train_model(data1, ml)
    #
    #         # calculation for Recent data
    #         signalsr = pred(model, recent_data, recent_close)
    #
    #         recent_balance.append(Snippets.calculate_balance(signalsr) - 100)
    #         possitionsr_arr.append(len(signalsr[signalsr['positions'] == 1]))
    #
    #         # calculation for Verif data
    #         signals = pred(model, verif_data, verif_close)
    #
    #         verif_balance.append(Snippets.calculate_balance(signals) - 100)
    #         possitionsv_arr.append(len(signals[signals['positions'] == 1]))
    #
    #         model_arr.append(str(ml)[:5])
    #         ticker_arr.append(ticker)
    #
    #         columns_arr.append(str(colmns_select))
    #
    #     colmns_select.clear()
    #
    # df = pd.DataFrame({'model': model_arr,
    #                    'pos_ver': possitionsv_arr, 'pos_rec': possitionsr_arr,
    #                    'verif_balance': verif_balance, 'recent_balance': recent_balance,
    #                    'columns': columns_arr})
    #
    # df['Result'] = df['verif_balance'] / df['pos_ver'] + df['recent_balance'] / df['pos_rec']
    # df['Result2'] = (df['verif_balance'] + df['recent_balance']) / (df['pos_ver'] + df['pos_rec'])
    # df.to_csv('C:\\Users\\Vlad\\Desktop\\Finance\\final ' + ticker + '.csv')

# signal_max_arr = []
# signal_min_arr = []
# data_max_arr = []
# data_min_arr = []
# print("Mean balance ", np.mean(balance_arr))
# print("Median balance ", np.median(balance_arr))


# print("Mean balance ", np.mean(balance_arr))
# print("Median balance ", np.median(balance_arr))
# for i, j in zip(balance_arr, ticker_arr):
#     print(j, i)
