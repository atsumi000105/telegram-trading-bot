# Install the dependencies
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from TimeSeries_models import model_selection

plt.style.use('bmh')

'''
period: data period to download (either use period parameter or use start and end) Valid periods are:
“1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”
interval: data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days) Valid intervals are:
“1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”

df = yf.download('ETH-USD','2021-06-01', '2021-06-06', interval="1m")
'''


# Get data from Yahoo. if data_prep = True, then keep column with Dates and Close price. Add column with price changes
def get_data(currency, date_from, date_to, interval, data_prep=False):
    df = yf.download(currency, date_from, date_to, interval=interval)
    df.reset_index(inplace=True)
    df = df.set_index("Datetime")
    if data_prep:
        if "d" in interval:
            df = df[["Close"]]
        else:
            df = df[["Close"]]
    return df


currency = 'ETH-USD'

date_from = datetime.today() - timedelta(days=5)
date_to = datetime.today()  # - timedelta(days=3)

# interval: data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days)
# Valid intervals are: “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”
interval = "5m"
df_5m = get_data(currency, date_from, date_to, interval, True)
df_real = get_data(currency, date_from, date_to, interval, True)
#df_real.to_csv(r'C:\Users\Vlad\Desktop\ML_price_pred\data.csv')
# Nr of point we will predict
pred_point = 5

# Supported models : DecisionTree, RandomForest, XGBboost
models = ["DecisionTree", "RandomForest", "XGBboost"]
for model in models:
    ml_class = model_selection(model=model, points=pred_point, data=df_5m)
    X = ml_class.get_featuredata()
    prediction = ml_class.run_model()
    # Visualize results

    valid = df_5m[X.shape[0]:]

    prediction_real = ml_class.predict(df_real)
    prediction_real = prediction_real[-pred_point:]

    # Assigne future time to predicted values. (time = index)
    df = pd.DataFrame({'index': valid.index + pd.Timedelta(pred_point * pred_point, unit='minutes'),
                       "values": prediction_real})

    df = df.set_index('index')

    prediction_real = df
    plt.figure(figsize=(16, 8))
    plt.title(model)
    plt.xlabel("Points(5 min)")
    plt.ylabel('Close Price USD')
    plt.plot(df_real["Close"][-24:])
    plt.plot(prediction_real)
    plt.legend(['Оригинал', 'Предсказание'])
    plt.show()
    print()
    print(prediction_real)
print(df_5m.tail(10))
print()

