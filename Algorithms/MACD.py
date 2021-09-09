import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import Get_data
plt.style.use('fivethirtyeight')
import Snippets


def buy_sell(signal):
    buy = []
    sell = []
    flag = -1

    for i in range(0, len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            sell.append(0)
            if flag != 1:
                buy.append(1)
                flag = 1
            else:
                buy.append(0)

        elif signal['MACD'][i] < signal['Signal Line'][i]:
            buy.append(0)
            if flag != 0:
                sell.append(-1)
                flag = 0
            else:
                sell.append(0)
        else:
            buy.append(0)
            sell.append(0)
    return (buy, sell)

today = date.today()

today = today.strftime("%Y-%m-%d")
start_date = '2021-05-01'  # начало периода
end_date = today  # конец периода
ticker = [ 'XLMUSDT', 'BNBUSDT', 'ADAUSDT', "XLMBNB", "ADABNB"]


df = Get_data.binance_data( 'XLMUSDT' , start_date, today)
df = df.set_index(pd.DatetimeIndex(df['Close Time'].values))

def MACD(sw, lw, signal_span):
    #calculate the short term exponential moving average (EMA)
    ShortEMA = df.Close.ewm(span=sw, adjust=False).mean()
    #calculate the long term exponential moving average (EMA)
    LongEMA = df.Close.ewm(span=lw, adjust=False).mean()

    #calculate teh MACD line
    MACD = ShortEMA - LongEMA
    #calculate the signal line
    signal = MACD.ewm(span=signal_span, adjust=False).mean()


    df['MACD'] = MACD
    df['Signal Line'] = signal

    a = buy_sell(df)
    df['Buy_Signal_Price'] = a[0]
    df['Sell_Signal_Price'] = a[1]
    df['positions'] = df['Buy_Signal_Price'] + df['Sell_Signal_Price']
    return df


# df_f = MACD(147, 17, 145)
#
# balance = Snippets.calculate_balance(df_f, array=True)

# fig = plt.figure()
# fig.set_size_inches(22.5, 10.5)
# ax1 = fig.add_subplot(111, ylabel='XLMUSDT in $')
# df_f["Close"].plot(ax=ax1, color='g', lw=.5)
#
#
# ax1.plot(df_f.loc[df.position == 1.0].index, df_f["Close"][df_f.position == 1.0],
#          '^', markersize=7, color='k')
#
# ax1.plot(df_f.loc[df_f.position == -1.0].index, df_f["Close"][df_f.position== -1.0],
#          'v', markersize=7, color='k')
#
# plt.legend(["Price", "Buy", "Sell"])
# plt.title("MACD 35/17" )
#
# plt.show()


balance_arr = []
sw_arr = []
lw_arr = []
sgl_arr = []
for sw in range(1,250,5):
    print(5/250*100)
    for lw in range(1,250,5):
        for sgl in range(1,250,5):
            df = MACD(sw, lw, sgl)
            balance_arr.append(Snippets.calculate_balance(df))
            sw_arr.append(sw)
            lw_arr.append(lw)
            sgl_arr.append(sgl)

final_data = {'Balance': balance_arr, 'SW': sw_arr, 'LW': lw_arr, "SGL": sgl_arr}
final_df = pd.DataFrame(final_data)
final_df.sort_values("Balance").tail(10)



