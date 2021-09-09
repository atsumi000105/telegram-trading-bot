import pandas as pd
import numpy as np
import Snippets
import Get_data

start_date = '2021-06-16'
end_date = '2021-08-20'
ticker = ['BNBUSDT', 'ADAUSDT']


data_signal = Get_data.binance_data('BNBUSDT', start_date)

max_price = 0
orders = []
for i in range(len(data_signal)):
    price = data_signal.iloc[i, 0]
    if price > max_price:
        max_price = price
        orders.append(0)
        continue
    if max_price/price >= 2:
        print("sell all ", price, i)
        orders.append(-1)
    elif max_price/price >= 1.1:
        print("buy ", price, i)
        orders.append(1)
        max_price = price
    else:
        orders.append(0)
data_signal['orders'] = orders

from matplotlib import pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google price in $')
data_signal["Close"].plot(ax=ax1, color='g', lw=.5)

ax1.plot(data_signal.loc[data_signal.orders== 1.0].index,
         data_signal["Close"][data_signal.orders == 1],
         '^', markersize=7, color='k')

ax1.plot(data_signal.loc[data_signal.orders== -1.0].index,
         data_signal["Close"][data_signal.orders == -1],
         'v', markersize=7, color='k')

plt.legend(["Price","Buy","Sell"])
plt.title("Vovan strategy")

plt.show()
