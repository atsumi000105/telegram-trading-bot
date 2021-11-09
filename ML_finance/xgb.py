# Load libraries
import os

import Snippets
import Indicators
#import Get_data
from datetime import datetime, timedelta, date
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import xgboost as xgb



import warnings
warnings.filterwarnings('ignore')
# Libraries for Deep Learning Models
# ada 35 75
# bnb 149 128
def train_model(dataset, init_model):
    # Create short simple moving average over the short window
    dataset['short_mavg'] = dataset['Close'].rolling(window=35, min_periods=1, center=False).mean()
    #
    # # Create long simple moving average over the long window
    dataset['long_mavg'] = dataset['Close'].rolling(window=75, min_periods=1, center=False).mean()

    # Create signals
    dataset['positions'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 0.0, 1.0)
    #dataset['Close2'] = dataset['Close'].shift(1)

    #dataset['positions'] = np.where(dataset['Close'] > dataset['Close2'], 0, 1)
    dataset.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    #dataset.drop(['Close2'], axis=1, inplace=True)

    # dataset.drop(['short_mavg', 'long_mavg', 'Close Time.1'], axis=1, inplace=True)

    dataset = add_indicators(dataset)

    dataset = dataset.dropna(axis=0)
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
        #models.append(('LR', LogisticRegression(n_jobs=-1)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        #models.append(('KNN', KNeighborsClassifier()))
        #models.append(('CART', DecisionTreeClassifier()))
        #models.append(('NB', GaussianNB()))
        # Neural Network
        #models.append(('NN', MLPClassifier()))
        # Ensable Models
        # Boosting methods
        models.append(('AB', AdaBoostClassifier()))
        #models.append(('GBM', GradientBoostingClassifier()))
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

    model = init_model
    model.fit(X, Y)
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

    # Calculation of price momentum
    def MOM(df, n):
        MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
        return MOM

    dataset['MOM10'] = MOM(dataset['Close'], 10)
    dataset['MOM30'] = MOM(dataset['Close'], 30)

    # calculation of relative strength index
    dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
    dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
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
    dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
    dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

    # Calculation of moving average

    dataset['SMA10'] = Indicators.SMA(dataset['Close'], 10)
    dataset['SMA20'] = Indicators.SMA(dataset['Close'], 20)
    dataset['SMA35'] = Indicators.SMA(dataset['Close'], 35)
    dataset['SMA78'] = Indicators.SMA(dataset['Close'], 78)
    dataset['SMA55'] = Indicators.SMA(dataset['Close'], 55)
    dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90)
    dataset['SMA155'] = Indicators.SMA(dataset['Close'], 155)


    return dataset


def pred(model, df):
    dataset = add_indicators(df)

    dataset = dataset.dropna(axis=0)
    #dataset = dataset.drop(['Close Time'], axis=1)

    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()

    return dataset


def prof_loss(sell_price, buy_price):
    ratio = sell_price / buy_price
    if ratio > 1:
        return ' прибыль ' + str(round(ratio * 100 - 100, 2)) + '%'
    else:
        return ' убыток ' + str(round(100 - ratio * 100, 2)) + '%'


id = -467554548
text = "Запуск стратегии Aboost. Покупка в случаее цена > MA90"
print(text, id)
today = date.today()
today = today - timedelta(days=7)
today = today.strftime("%Y-%m-%d")
sol_df = pd.DataFrame()
buy_signal = 0
files = os.listdir('C:\\Users\\Vlad\\Desktop\\Finance\\AboostDATA')
curr_dict = dict()
for f in files:
    curr_dict[f[:len(f) - 8]] = 0
data = pd.read_csv('C:\\Users\\Vlad\\Desktop\\Finance\\ADA 1 hr.csv')
data.drop(['Close Time.1', 'Close Time'], axis=1, inplace=True)

model = xgb.XGBClassifier()
train_data = data[1000:5000]
model = train_model(train_data, model)
# models = []
# #models.append(('XGB', xgb.()))
#
#
#
# for name, model_init in models:
#     train_data = data[1000:5000]
#
#     model = train_model(train_data)
#
recent_data = data[5000:]
signals = pred(model, recent_data)
print( Snippets.calculate_balance(signals),len(signals[signals['positions'] == 1]))

#print(signals.predictions.sum())
signals = signals[['Close', 'predictions', 'positions']]
signals['Balance'] = Snippets.calculate_balance(signals, array=True)
print('min ',signals['Balance'].min(),  ' max ', signals['Balance'].max())
