
import Indicators


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

        def train_model(dataset, model_init):
            # # Create short simple moving average over the short window
            # dataset['short_mavg'] = dataset['Close'].rolling(window=35, min_periods=1, center=False).mean()
            # #
            # # # Create long simple moving average over the long window
            # dataset['long_mavg'] = dataset['Close'].rolling(window=75, min_periods=1, center=False).mean()
            #
            # # Create signals
            # dataset['positions'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 0.0, 1.0)
            dataset['Close2'] = dataset['Close'].shift(1)

            # dataset['short_mavg'] = dataset['Close'].rolling(window=20, min_periods=1, center=False).mean()

            dataset['positions'] = np.where(dataset['Close'] > dataset['Close2'], 0, 1)
            # dataset.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
            dataset.drop(['Close2'], axis=1, inplace=True)

            # dataset.drop(['short_mavg', 'long_mavg', 'Close Time.1'], axis=1, inplace=True)

            dataset = add_indicators(dataset)

            dataset = dataset.dropna(axis=0)

            # dataset = dataset.drop(['Close Time', 'Close Time.1'], axis=1)
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

            #test_models()

            #model =  RandomForestClassifier(criterion='gini', max_depth=30,  n_jobs=-1, n_estimators=150)
            #model = LinearDiscriminantAnalysis()
            #model = GradientBoostingClassifier()
            #model = AdaBoostClassifier(n_estimators=n_estimators,)

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

            # predictions = model.predict(X_validation)
            # print(accuracy_score(Y_validation, predictions))
            # print(confusion_matrix(Y_validation, predictions))
            # print(classification_report(Y_validation, predictions))

            # predictions = model.predict(subset_dataset)
            # subset_dataset['predictions'] = predictions
            # subset_dataset['positions'] = subset_dataset['predictions'].diff()
            # print(100-Snippets.calculate_balance(subset_dataset))

        return add_indicators(self.dataset)
        #return train_model(self.dataset, self.model_init)

def pred_real(model, df):
    dataset = ML(df, model).real()
    dataset = dataset.dropna(axis=0)

    predictions = model.predict(dataset)
    dataset['predictions'] = predictions
    dataset['positions'] = dataset['predictions'].diff()

    return dataset

def pred_test(model, df, ticker):

    #dataset = ML(df, model).test()
    from ML_finance.Model_by_model import Indicators_all
    Indicators_new = Indicators_all.model_by_model()

    if ticker == "ADA":
        dataset = Indicators_new.ADA(df)
    if ticker == "BAT":
        dataset = Indicators_new.BAT(df)
    if ticker == "BEAM":
        dataset = Indicators_new.BEAM(df)
    if ticker == "BNB":
        dataset = Indicators_new.BNB(df)
    if ticker == "BTC":
        dataset = Indicators_new.BTC(df)
    if ticker == "COTI":
        dataset = Indicators_new.COTI(df)
    if ticker == "EOS":
        dataset = Indicators_new.EOS(df)
    if ticker == "ETH":
        dataset = Indicators_new.ETH(df)
    if ticker == "SOL":
        dataset = Indicators_new.SOL(df)
    if ticker == "VET":
        dataset = Indicators_new.VET(df)


    dataset = dataset.dropna(axis=0)
    #dataset = dataset.drop(['Close Time'], axis=1)

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
