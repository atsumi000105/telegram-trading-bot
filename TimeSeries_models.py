from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import pickle


class model_selection:
    '''
    model = choose model between
    points = nr of data points for wich we need make prediction
    data = our data frame. Column with prices names "Closed"

    '''

    def __init__(self, model, points, data):
        self.model = model
        self.points = points
        self.df = data
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prep(data, points)
        self.path = r'C:\Users\Vlad\Desktop\ML_price_pred'

    # Prepare data
    def data_prep(self, df, pred_point):
        # Create new column with shifted x points
        df["Prediction"] = df[["Close"]].shift(-pred_point)
        # Make an array from Close column. All but last rows that are empty. This will be feature column
        X = self.get_featuredata()
        #
        y = np.array(df["Prediction"])[:-pred_point].reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        return X_train, X_test, y_train, y_test

    # prepare
    def get_xfuture(self):

        x_future = self.df["Close"][:-self.points]
        x_future = x_future.tail(self.points)
        return np.array(x_future).reshape(-1, 1)

    def get_featuredata(self):
        return np.array(self.df["Close"])[:-self.points].reshape(-1, 1)

    # test models
    def decisiontree(self):
        model = DecisionTreeRegressor()
        model = model.fit(self.X_train, self.y_train)
        self.write_pickle(path=self.path + '\DecionTree.pickle', data=model)
        return model.predict(self.get_xfuture())

    def randomforest(self):
        model = RandomForestRegressor()
        model = model.fit(self.X_train, self.y_train)
        self.write_pickle(path=self.path + '\RandomForest.pickle', data=model)
        return model.predict(self.get_xfuture())

    def xgbboost(self):
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model = model.fit(self.X_train, self.y_train)
        self.write_pickle(path=self.path + '\XGBboost.pickle', data=model)
        return model.predict(self.get_xfuture())

    def linerregression(self):
        model = LinearRegression()
        model = model.fit(self.X_train, self.y_train)
        self.write_pickle(path=self.path + '\LinerRegression.pickle', data=model)
        return model.predict(self.get_xfuture())

    def run_model(self):
        if self.model == "DecisionTree": return self.decisiontree()
        if self.model == "LinerRegression": return self.linerregression()
        if self.model == "RandomForest": return self.randomforest()
        if self.model == "XGBboost": return self.xgbboost()

    def write_pickle(self, path, data, print_flag=False):
        if print_flag:
            print("Write to pickle", path)
        with open(path, 'wb') as handle:
            return pickle.dump(data, handle)

    def read_pickle(self, path, print_flag=False):
        if print_flag:
            print("Loading pickle", path)
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def predict(self, data):

        if self.model == "DecisionTree": model = self.read_pickle(path=self.path + '\DecionTree.pickle')
        if self.model == "LinerRegression": model = self.read_pickle(path=self.path + '\LinerRegression.pickle')
        if self.model == "RandomForest": model = self.read_pickle(path=self.path + '\RandomForest.pickle')
        if self.model == "XGBboost": model = self.read_pickle(path=self.path + '\XGBboost.pickle')

        return model.predict(np.array(data).reshape(-1, 1))
