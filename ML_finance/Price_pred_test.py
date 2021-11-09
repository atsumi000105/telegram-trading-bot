# Load libraries
import os
import pickle
import random

import Get_data
import Snippets
import Indicators
from ML_finance.Model_by_model import Indicators_all
Indicators_new = Indicators_all.model_by_model()

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

    def test_models():
        # test options for classification
        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        # scoring = 'precision'
        # scoring = 'recall'
        # scoring ='neg_log_loss'
        # scoring = 'roc_auc'

        # spot check the algorithms
        models = []
        # models.append(('LR', LogisticRegression(n_jobs=-1)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        # Neural Network
        # models.append(('NN', MLPClassifier()))
        # Ensable Models
        # Boosting methods
        models.append(('AB', AdaBoostClassifier()))
        # models.append(('GBM', GradientBoostingClassifier()))
        # Bagging methods
        models.append(('RF', RandomForestClassifier(n_jobs=-1)))

        results = []
        names = []
        for name, model in models:
            kfold = KFold(n_splits=num_folds)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # # compare algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        fig.set_size_inches(15, 8)
        plt.show()

    # test_models()

    # model =  RandomForestClassifier(criterion='gini', max_depth=30,  n_jobs=-1, n_estimators=150)
    # model = LinearDiscriminantAnalysis()
    # model = GradientBoostingClassifier()
    # model = AdaBoostClassifier(n_estimators=n_estimators,)

    model = model_init
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
    #dataset['EMA600'] = Indicators.EMA(dataset['Close'], 600)

    #dataset['EMA600'] = Indicators.EMA(dataset['Close'], 600)

    dataset['Low'] = dataset['Low'] / dataset['Close']
    dataset['High'] = dataset['High']/ dataset['Close']
    dataset['Open'] = dataset['Open']/ dataset['Close']
    #
    dataset['SMAV10'] = Indicators.SMA(dataset['Volume'], 10) / dataset['Volume']
    dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']

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


def check_all_variants(data, currency, cat_class):

    # for name, model_init in models:

    train_data = data[3500:]

    # recent_data = pd.read_csv(
    #      'C:\\Users\\Vlad\\Desktop\\Finance\\Verif\\' + currency + ' 1hr.csv')
    # recent_data.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)

    recent_data = Get_data.binance_data(currency + 'USDT', '2021-10-21')
    recent_data.drop(['Close Time'], axis=1, inplace=True)

    # if not balance_arr:
    #     print(train_data['Close'].min(), train_data['Close'].max(), train_data['Close'].mean())
    #     print(recent_data['Close'].min(), recent_data['Close'].max(), recent_data['Close'].mean())

    model = train_model(train_data, cat_class)
    # with open('Models\\' + currency + ".pickle", 'wb') as handle:
    #      pickle.dump(model, handle)
    # print("model for ", currency, " has been saved")
    signals = pred(model, recent_data)

    # signals = signals[-200:]
    #print(len(signals))

    #signals = signals[['Close', 'predictions', 'positions']]

    balance_arr.append(Snippets.calculate_balance(signals)-100)
    ticker_arr.append(currency)

    #print(np.sum(balance_arr))
    possitions_arr.append(len(signals[signals['positions'] == 1]))


    #t1 = time.time()
    #print(t1-t0)


    # final_data = {'balance_arr': balance_arr, 'possitions_arr': possitions_arr,
    #               'ticker':ticker_arr}
    #
    # return pd.DataFrame(final_data)


def simulation(data, cat_class, curr_dict, currency):

    for i in range(2160, 2320):
        train_data = data[2000:3000]
        recent_data = data[i+1000:i+1093]

        model = train_model(train_data, cat_class)
        signals = pred(model, recent_data)

        close_price = signals.Close.iloc[-1]

        if int(signals.positions.iloc[-1]) == 1 and \
                curr_dict[currency] == 0:
            curr_dict[currency] = signals.Close.iloc[-1]

            #print('Покупай ' + currency + ' цена ' + str(close_price))
        elif int(signals.positions.iloc[-1]) == -1 and \
                curr_dict[currency] != 0:
            #print('Продаем ' + currency + ' , цена: ' + str(close_price) + prof_loss(
            #close_price, curr_dict[currency]))
            ticker_arr.append(currency)
            balance_arr.append(prof_loss(close_price, curr_dict[currency]))
            curr_dict[currency] = 0


def get_random_columns():
    cols = [random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
            random.random(), random.random(), random.random(), random.random(), random.random(), random.random()]
    colmns_select = []
    for count, column in enumerate(cols):
        if column > 0.5: colmns_select.append(count)
    return colmns_select


def model_foreach_teacker():

    #tickers = ['ADA', 'BAT', 'BEAM', 'BNB', 'BTC', 'COTI', 'EOS', 'ETH', 'SOL', 'VET', 'XLM', 'XMR', 'XRP', 'ALGO', 'ATOM', 'AVAX', 'DOT', 'FTM', 'LINK', 'LTC', 'LUNA', 'MATIC', 'TRX']
    tickers = ['ADA']

    cat_class = [LinearDiscriminantAnalysis(),
                 CatBoostClassifier(logging_level='Silent'),
                 KNeighborsClassifier(),
                 GaussianNB(),
                 RandomForestClassifier(),
                 XGBClassifier()]


    for ticker in tickers:

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

        verif_data1 = Get_data.binance_data(ticker + 'USDT', '2021-10-25')
        verif_data1.drop(['Close Time'], axis=1, inplace=True)
        verif_data1 = add_indicators(verif_data1)
        verif_close = verif_data1['Close']
        verif_data1.drop(['Close'], axis=1, inplace=True)


        # ml = cat_class[0]
        # i = 4100
        # colmns_select = [1, 2, 5, 6, 8, 10, 12, 18, 22, 23, 25, 29, 30, 34, 35]
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
        # with open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\ML_finance\Model_by_model\Models2\\' + ticker +' columns.pickle',
        #           'wb') as handle:
        #     pickle.dump(colmns_select, handle)

        for i in range(3100, 4200, 100):
            for random_count in range(1, 500):
                colmns_select = get_random_columns()
                data1 = data.iloc[:, colmns_select].copy()
                data1['positions'] = positions[201:]
                data1 = data1.iloc[i:, ]

                recent_data = recent_data1.iloc[:, colmns_select].copy()
                verif_data = verif_data1.iloc[:, colmns_select].copy()

                for ml in cat_class:

                    model = train_model(data1, ml)

                    signalsr = pred(model, recent_data, recent_close)

                    recent_balance.append(Snippets.calculate_balance(signalsr) - 100)
                    possitionsr_arr.append(len(signalsr[signalsr['positions'] == 1]))

                    signals = pred(model, verif_data, verif_close)

                    verif_balance.append(Snippets.calculate_balance(signals) - 100)

                    ticker_arr.append(ticker)

                    possitionsv_arr.append(len(signals[signals['positions'] == 1]))
                    model_arr.append(str(ml)[:5])

                    iter_arr.append(i)

                    columns_arr.append(str(colmns_select))

                colmns_select.clear()


    #todo
    # check in the end with model without parameters

def main():
    files = os.listdir('C:\\Users\\Vlad\\Desktop\\Finance\\Raw data')
    for rate in np.arange(0.1, 1, 0.1):
        for dep in range(3, 10):


            cat_class = CatBoostClassifier( learning_rate=rate, depth=dep, logging_level='Silent')

            for f in files:
                currency = f[:len(f) - 8]

                data = pd.read_csv('C:\\Users\\Vlad\\Desktop\\Finance\\Raw data\\' + currency + ' 1hr.csv')
                data.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)
                # recent_data = pd.read_csv(
                #     'C:\\Users\\Vlad\\Desktop\\Finance\\Verif\\' + currency + ' 1hr verif.csv')
                # recent_data.drop(['Close Time'], axis=1, inplace=True)
                check_all_variants(data, currency, cat_class)
                # df.to_csv('C:\\Users\\Vlad\\Desktop\\Finance\\' + currency + ' verif.csv')
                #simulation(data, cat_class, curr_dict, currency)
            print(rate, dep)
            print("Mean balance ", np.mean(balance_arr))
            print("Median balance ", np.median(balance_arr))
            # balance_arr.clear()

iter_arr = []
possitionsv_arr = []
possitionsr_arr = []
ticker_arr = []
recent_balance = []
verif_balance = []
model_arr = []
temp_close = []

columns_arr = []
model_foreach_teacker()

# signal_max_arr = []
# signal_min_arr = []
# data_max_arr = []
# data_min_arr = []
# print("Mean balance ", np.mean(balance_arr))
# print("Median balance ", np.median(balance_arr))

df = pd.DataFrame({'model': model_arr,
                   'pos_ver': possitionsv_arr, 'pos_rec': possitionsr_arr,
                   'verif_balance': verif_balance, 'recent_balance': recent_balance,
                   "iter": iter_arr, 'columns': columns_arr})

df['Result'] = df['verif_balance']/df['pos_ver'] + df['recent_balance']/df['pos_rec']
df['Result2'] = (df['verif_balance']+df['recent_balance']) / (df['pos_ver'] + df['pos_rec'])
for i in range(0,36):
    df[str(i)] = [False]*len(df)

for j in range(0, len(df)):
    for i in range(0, 36):
        if " "+ str(i) + "," in df.iloc[j, 6] or \
                str(i) + "]" in df.iloc[j, 6] or \
            "["+ str(i) + "," in df.iloc[j, 6]:
            df.iloc[j, i+8] = True

df.to_csv('C:\\Users\\Vlad\\Desktop\\Finance\\final.csv')
# print("Mean balance ", np.mean(balance_arr))
# print("Median balance ", np.median(balance_arr))
# for i, j in zip(balance_arr, ticker_arr):
#     print(j, i)


'''
0 ROC10
1 ROC30
2 ROC50
3 ROC100
4 ROC150
5 MOM10
6 MOM30
7 MOM50
8 MOM100
9 MOM150
10 RSI10
11 RSI30
12 RSI50
13 RSI100
14 RSI200
15 %K10
16 %D10
17 %K30
18 %D30
19 %K50
20 %D50
21 %K90
22 %D90
23 %K200
24 %D200
25 SMA10
26 SMA20
27 SMA55
28 SMA90
29 SMA155
30 EMA20
31 EMA100
32 EMA200
33 EMA400
34 SMAV10
35 SMAV50
'''