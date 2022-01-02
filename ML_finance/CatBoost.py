
import Indicators


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score


# Libraries for Deep Learning Models
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class ML:

    def __init__(self, dataset, model_init):
        self.dataset = dataset
        self.model_init = model_init

    def real(self):

        def train_model(dataset, model_init):

            dataset['Close2'] = dataset['Close'].shift(1)



            dataset['positions'] = np.where(dataset['Close'] > dataset['Close2'], 0, 1)

            dataset.drop(['Close2'], axis=1, inplace=True)



            dataset = add_indicators(dataset)

            dataset = dataset.dropna(axis=0)


            Y = dataset["positions"]
            X = dataset.loc[:, dataset.columns != 'positions']
            validation_size = 0.2





            # model =  RandomForestClassifier(criterion='gini', max_depth=30,  n_jobs=-1, n_estimators=150)
            # model = LinearDiscriminantAnalysis()
            # model = GradientBoostingClassifier()
            # model = AdaBoostClassifier(n_estimators=n_estimators,)

            model = model_init
            model.fit(X, Y)

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

            dataset['SMA55'] = Indicators.SMA(dataset['Close'], 55)
            dataset['SMA90'] = Indicators.SMA(dataset['Close'], 90)
            dataset['SMA155'] = Indicators.SMA(dataset['Close'], 155)

            dataset['EMA100'] = Indicators.EMA(dataset['Close'], 100)

            return dataset

        return add_indicators(self.dataset)

    def test(self):

        def add_indicators(dataset):
            # calculation of rate of change

            dataset['ROC10'] = Indicators.ROC(dataset['Close'], 10)
            dataset['ROC30'] = Indicators.ROC(dataset['Close'], 30)
            dataset['ROC50'] = Indicators.ROC(dataset['Close'], 50)
            dataset['ROC100'] = Indicators.ROC(dataset['Close'], 100)
            dataset['ROC150'] = Indicators.ROC(dataset['Close'], 150)

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



            dataset['%K10'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
            dataset['%D10'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
            dataset['%K30'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
            dataset['%D30'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)

            dataset['%K50'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], 50)
            dataset['%D50'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], 50)
            dataset['%K90'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], 90)
            dataset['%D90'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], 90)

            dataset['%K200'] = Indicators.STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
            dataset['%D200'] = Indicators.STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

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
            # dataset['EMA600'] = Indicators.EMA(dataset['Close'], 600)

            # dataset['EMA600'] = Indicators.EMA(dataset['Close'], 600)

            dataset['Low'] = dataset['Low'] / dataset['Close']
            dataset['High'] = dataset['High'] / dataset['Close']
            dataset['Open'] = dataset['Open'] / dataset['Close']

            dataset.drop(['Volume', 'Low', 'High', 'Open'], axis=1, inplace=True)

            dataset = dataset.dropna(axis=0)

            return dataset

        return add_indicators(self.dataset)
        #return train_model(self.dataset, self.model_init)

def pred_real(model, df):
    dataset = ML(df, model).real()
    dataset = dataset.dropna(axis=0)

    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()

    return dataset

def pred_test(model, df, columns):


    dataset = ML(df, model).test()

    dataset = dataset.iloc[:, columns]

    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()

    # print(dataset.tail(10))
    #dataset = dataset.tail(1)
    # print(100 - Snippets.calculate_balance(dataset))

    # Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=X.columns)
    # Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r' )
    # plt.xlabel('Variable Importance')


    return dataset
