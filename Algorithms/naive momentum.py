import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import Get_data
from datetime import date

def naive_momentum_trading(financial_data, nb_conseq_days):
    signals = pd.DataFrame(index=financial_data.index)
    signals['orders'] = 0
    cons_day=0
    prior_price=0
    init=True
    for k in range(len(financial_data['Close'])):
        price=financial_data['Close'][k]
        if init:
            prior_price=price
            init=False
        elif price>prior_price:
            if cons_day<0:
                cons_day=0
            cons_day+=1
        elif price<prior_price:
            if cons_day>0:
                cons_day=0
            cons_day-=1
        if cons_day==nb_conseq_days:
            signals['orders'][k]=1
        elif cons_day == -nb_conseq_days:
            signals['orders'][k]=-1


    return signals

today = date.today()

today = today.strftime("%Y-%m-%d")
start_date = '2021-05-01'  # начало периода
end_date = today  # конец периода
ticker = [ 'XLMUSDT', 'BNBUSDT', 'ADAUSDT', "XLMBNB", "ADABNB"]


df = Get_data.binance_data( 'ADAUSDT' , start_date, today)
df = df.set_index(pd.DatetimeIndex(df['Close Time'].values))

ts=naive_momentum_trading(df, 20)

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google price in $')
df["Close"].plot(ax=ax1, color='g', lw=.5)

ax1.plot(ts.loc[ts.orders== 1.0].index,
         df["Close"][ts.orders == 1],
         '^', markersize=7, color='k')

ax1.plot(ts.loc[ts.orders== -1.0].index,
         df["Close"][ts.orders == -1],
         'v', markersize=7, color='k')

plt.legend(["Price","Buy","Sell"])
plt.title("Naive Momentum Trading Strategy")

plt.show()
