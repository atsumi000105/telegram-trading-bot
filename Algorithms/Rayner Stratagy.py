# https://www.youtube.com/watch?v=pB8eJwg7LJU&t=285s&ab_channel=Algovibes

import matplotlib.pyplot as plt
import numpy as np
import Get_data
import Snippets
import warnings
warnings.filterwarnings('ignore')

start_date = '2021-05-16'


def RSIcalc(df, window, rsi):

    df['MA200'] = df["Close"].rolling(window=window).mean()
    df['price change'] = df['Close'].pct_change()
    df['Upmove'] = df['price change'].apply(lambda x: x if x > 0 else 0)
    df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['avg Up'] = df['Upmove'].ewm(span=19).mean()
    df['avg Down'] = df['Downmove'].ewm(span=19).mean()
    df = df.dropna()
    df['RS'] = df['avg Up'] / df['avg Down']
    df['RSI'] = df['RS'].apply(lambda x: 100 - (100 / (x + 1)))
    df['Buy'] = 'No'
    df.loc[(df['Close'] > df['MA200']) & (df['RSI'] < rsi), 'Buy'] = 'Yes'
    #df.loc[(df['Close'] < df['MA200']) | (df['RSI'] < 30), 'Buy'] = 'No'
    return df


def get_signals(df):
    buy_date = []
    sell_date = []

    for i in range(len(df)):
        if 'Yes' in df['Buy'].iloc[i]:
            buy_date.append(df.iloc[i + 1].name)
            for j in range(1, 11):
                if df['RSI'].iloc[i + j] > 40:
                    sell_date.append(df.iloc[i + j + 1].name)
                    break
                elif j == 10:
                    sell_date.append(df.iloc[i + j + 1].name)

    return buy_date, sell_date

# df = Get_data.binance_data("ADAUSDT", start_date)
# for window in range(100,250,10):
#     for rsi in range(20,60,5):
#         frame = RSIcalc(df, window, rsi)
#         buy, sell = get_signals(frame)
#
#         # plt.figure(figsize=(12, 5))
#         # plt.scatter(frame.loc[buy].index, frame.loc[buy]['Close'], marker='^', c='g')
#         # plt.plot(frame['Close'], alpha=0.7)
#         # plt.show()
#
#         Profits = (frame.loc[sell].Close.values - frame.loc[buy].Close.values)/frame.loc[buy].Close.values
#
#         wins = [i for i in Profits if i > 0]
#         if len(Profits) != 0:
#             print(window, rsi, round(len(wins)/len(Profits), 2))
#         else:
#             print(window, rsi, 'failed')

df = Get_data.binance_data("ADAUSDT", start_date)

frame = RSIcalc(df, 120, 22)
buy, sell = get_signals(frame)

plt.figure(figsize=(12, 5))
plt.scatter(frame.loc[buy].index, frame.loc[buy]['Close'], marker='^', c='g')
plt.plot(frame['Close'], alpha=0.7)
plt.show()

Profits = (frame.loc[sell].Close.values - frame.loc[buy].Close.values)/frame.loc[buy].Close.values

wins = [i for i in Profits if i > 0]

print(round(len(wins)/len(Profits), 2))
