# Load libraries
import os
import pickle
import Get_data
import Snippets
import Indicators

from datetime import datetime, timedelta, date
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score


class model_by_model:

    def ADA(self, dataset):
        # ml = cat_class[0]
        # i = 4200
        # calculation of rate of change
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)
        dataset['ROC50'] = ROC(dataset['Close'], 50)

        def MOM(df, n):
            MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
            return MOM

        dataset['MOM10'] = MOM(dataset['Close'], 10)
        dataset['MOM30'] = MOM(dataset['Close'], 30)

        # calculation of relative strength index
        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['SMA20'] = Indicators.SMA(dataset['Close'], 20) / dataset['Close']
        dataset['SMA55'] = Indicators.SMA(dataset['Close'], 55) / dataset['Close']

        dataset['SMAV10'] = Indicators.SMA(dataset['Volume'], 10) / dataset['Volume']
        dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def BAT(self, dataset):
        # ml = cat_class[0]
        # i = 4200
        # calculation of rate of change
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)
        dataset['ROC50'] = ROC(dataset['Close'], 50)


        # calculation of relative strength index
        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['SMAV10'] = Indicators.SMA(dataset['Volume'], 10) / dataset['Volume']
        dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def BEAM(self, dataset):
        # ml = cat_class[1]
        # i = 4100
        # calculation of rate of change
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)
        dataset['ROC50'] = ROC(dataset['Close'], 50)

        # calculation of relative strength index
        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['SMAV10'] = Indicators.SMA(dataset['Volume'], 10) / dataset['Volume']
        dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def BNB(self, dataset):
        # ml = cat_class[1]
        # i = 4200
        # calculation of rate of change
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)
        dataset['ROC50'] = ROC(dataset['Close'], 50)

        # calculation of relative strength index
        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['SMA10'] = Indicators.SMA(dataset['Close'], 10) / dataset['Close']
        dataset['SMA20'] = Indicators.SMA(dataset['Close'], 20) / dataset['Close']

        dataset['SMAV10'] = Indicators.SMA(dataset['Volume'], 10) / dataset['Volume']
        dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']

        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def BTC(self, dataset):
        # ml = cat_class[0]
        # i = 4200

        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)

        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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
        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def COTI(self, dataset):
        # ml = cat_class[0]
        # i = 3900
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)

        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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
        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def EOS(self, dataset):
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
        dataset['SMA55'] = Indicators.SMA(dataset['Close'], 55) / dataset['Close']
        dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90) / dataset['Close']
        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']
        # dataset['SMAV50'] = Indicators.SMA(dataset['Volume'], 50) / dataset['Volume']
        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def ETH(self, dataset):
        # ml = cat_class[1]
        # i = 4200
        # calculation of rate of change
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)

        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']

        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def SOL(self, dataset):
        # ml = cat_class[0]
        # i = 4100

        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)

        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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

        dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90) / dataset['Close']
        dataset['SMA155'] = Indicators.SMA(dataset['Close'], 155) / dataset['Close']

        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']

        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def VET(self, dataset):
        # ml = cat_class[1]
        # i = 4200
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)

        def MOM(df, n):
            MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
            return MOM

        dataset['MOM10'] = MOM(dataset['Close'], 10)

        dataset['RSI10'] = Indicators.RSI(dataset['Close'], 10)
        dataset['RSI30'] = Indicators.RSI(dataset['Close'], 30)
        dataset['RSI50'] = Indicators.RSI(dataset['Close'], 50)
        dataset['RSI100'] = Indicators.RSI(dataset['Close'], 100)

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
        #
        dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
        dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

        dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90) / dataset['Close']
        dataset['SMA155'] = Indicators.SMA(dataset['Close'], 155) / dataset['Close']

        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset

    def XLM(self, dataset):
        def ROC(df, n):
            M = df.diff(n - 1)
            N = df.shift(n - 1)
            ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
            return ROC

        dataset['ROC10'] = ROC(dataset['Close'], 10)
        dataset['ROC30'] = ROC(dataset['Close'], 30)
        dataset['ROC50'] = ROC(dataset['Close'], 50)

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


        dataset['SMA10'] = Indicators.SMA(dataset['Close'], 10) / dataset['Close']
        dataset['SMA20'] = Indicators.SMA(dataset['Close'], 20) / dataset['Close']

        dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100) / dataset['Close']
        dataset['EMA200'] = Indicators.EMA(dataset['Close'], 200) / dataset['Close']
        dataset['EMA400'] = Indicators.EMA(dataset['Close'], 400) / dataset['Close']

        dataset.drop(['Volume', 'Close', 'Low', 'High', 'Open'], axis=1, inplace=True)

        return dataset