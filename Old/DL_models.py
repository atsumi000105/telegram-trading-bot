# Load libraries
import os
import pickle
import random

import Get_data
import Snippets
import Indicators
from ML_finance.Model_by_model import Indicators_all

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from datetime import datetime, timedelta, date
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

    import tensorflow as tf

    FEATURES = dataset.columns[:-1]
    LABEL = dataset.columns[-1]

    model = tf.estimator.LinearClassifier(
    n_classes = 2,
    model_dir="ongoing/train",
    feature_columns = FEATURES)

    def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
        return tf.compat.v1.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
            y=pd.Series(data_set[LABEL].values),
            batch_size=n_batch,
            num_epochs=num_epochs,
            shuffle=shuffle)

    model.train(input_fn=get_input_fn(dataset,
                                      num_epochs=None,
                                      n_batch=128,
                                      shuffle=False),
                steps=1000)


    #model = model_init
    model.fit(X, Y)
    # model.fit(X_train, Y_train)
    #estimate accuracy on validation set
    #predictions = model.predict(X_validation)
    #print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))

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
    dataset['High'] = dataset['High']/ dataset['Close']
    dataset['Open'] = dataset['Open']/ dataset['Close']

    dataset.drop(['Volume',   'Low', 'High', 'Open'], axis=1, inplace=True)

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
    #dataset = Indicators_new.ADA(df)
    #dataset = add_indicators(df)

    #dataset = dataset.dropna(axis=0)

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
    cols = [random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random()]
    colmns_select = []
    for count, column in enumerate(cols):
        if column > 0.5: colmns_select.append(count)
    return colmns_select




#tickers = ['BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET', 'XLM', 'XMR', 'XRP']
#tickers = ['ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', 'TRX']
tickers = ['BAT']




for ticker in tickers:
    iter_arr = []
    possitionsv_arr = []
    possitionsr_arr = []
    ticker_arr = []
    recent_balance = []
    verif_balance = []
    model_arr = []
    columns_arr = []

    data = pd.read_csv('C:\\Users\\Vlad\\Desktop\\Finance\\Raw data\\' + ticker + ' 1hr.csv')
    data.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)
    data['Close2'] = data['Close'].shift(1)
    positions = np.where(data['Close'] > data['Close2'], 0, 1)
    data.drop(['Close2'], axis=1, inplace=True)
    data = add_indicators(data)
    data.drop(['Close'], axis=1, inplace=True)


    recent_data1 = pd.read_csv(
        'C:\\Users\\Vlad\\Desktop\\Finance\\Verif\\' + ticker + ' 1hr.csv')
    recent_data1.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)
    recent_data1 = add_indicators(recent_data1)
    recent_close = recent_data1['Close']
    recent_data1.drop(['Close'], axis=1, inplace=True)

    verif_data1 = Get_data.binance_data(ticker + 'USDT', '2021-10-30')
    verif_data1.drop(['Close Time'], axis=1, inplace=True)
    verif_data1 = add_indicators(verif_data1)
    verif_close = verif_data1['Close']
    verif_data1.drop(['Close'], axis=1, inplace=True)


    # ml = cat_class[0]
    # i = 3600
    # colmns_select = [2, 7, 11, 13, 14, 17, 21, 25, 27, 28, 31, 33]
    # data1 = data.iloc[:, colmns_select]
    # data1['positions'] = positions[201:]
    # data1 = data1.iloc[i:, ]
    # model = train_model(data1, ml)
    # with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' + ticker +'.pickle',
    #           'wb') as handle:
    #     pickle.dump(model, handle)
    # print("model has been saved")
    # recent_data = recent_data1.iloc[:, colmns_select].copy()
    # verif_data = verif_data1.iloc[:, colmns_select].copy()
    # signalsr = pred(model, recent_data, recent_close)
    # print(Snippets.calculate_balance(signalsr) - 100)
    # signals = pred(model, verif_data, verif_close)
    # print(Snippets.calculate_balance(signals) - 100)
    # print(len(signalsr[signalsr['positions'] == 1]))
    # print(len(signals[signals['positions'] == 1]))
    # with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' + ticker +' columns.pickle',
    #           'wb') as handle:
    #     pickle.dump(colmns_select, handle)
    cat_class = ['a']
    # for i in range(3500, 4200, 100):
    #     for random_count in range(1, 500):


    data['positions'] = positions[201:]


    for ml in cat_class:

        model = train_model(data, ml)

        signalsr = pred(model, recent_data1, recent_close)

        recent_balance.append(Snippets.calculate_balance(signalsr) - 100)
        possitionsr_arr.append(len(signalsr[signalsr['positions'] == 1]))

        signals = pred(model, verif_data1, verif_close)

        verif_balance.append(Snippets.calculate_balance(signals) - 100)

        ticker_arr.append(ticker)

        possitionsv_arr.append(len(signals[signals['positions'] == 1]))
        model_arr.append(str(ml)[:5])

        print(len(recent_data1), len(verif_data1))
        print(recent_balance)
        print(verif_balance)




# df = pd.DataFrame({'model': model_arr,
#                    'pos_ver': possitionsv_arr, 'pos_rec': possitionsr_arr,
#                    'verif_balance': verif_balance, 'recent_balance': recent_balance,
#                    "iter": iter_arr, 'columns': columns_arr})
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
