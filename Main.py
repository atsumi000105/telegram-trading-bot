from Algorithms import Double_avarage
import Get_data
import matplotlib.pyplot as plt
import time
from Telegram import Bot
from datetime import date, timedelta
import Snippets


past_interval=96




while True:
    today = date.today()
    past_range = today - timedelta(days=1)
    past_range = past_range.strftime("%Y-%m-%d")
    ada = Get_data.binance_data("ADAUSDT", past_range)
    bnb = Get_data.binance_data("BNBUSDT", past_range)


    sw, lw = Double_avarage.get_lw_sw(0, 100, ada, bnb)
    text = str(sw) + " " + str(lw)
    Bot.send_msg(text)
    for i in range(0, 96*5):
        today = date.today()
        past_range = today - timedelta(days=1)
        past_range = past_range.strftime("%Y-%m-%d")
        ada = Get_data.binance_data("ADAUSDT", past_range)
        bnb = Get_data.binance_data("BNBUSDT", past_range)

        signals_ada = Double_avarage.double_moving_average(ada, sw, lw)
        signals_bnb = Double_avarage.double_moving_average(bnb, sw, lw)


        if int(signals_ada.positions.iloc[-1]) == 1:
            Bot.send_msg('Покупай ADA цена ' + str(signals_ada.Close.iloc[-1]))
        elif int(signals_ada.positions.iloc[-1]) == -1:
            Bot.send_msg("Продаем ADA, цена: " + str(signals_ada.Close.iloc[-1]))

        if int(signals_bnb.positions.iloc[-1]) == 1:
            Bot.send_msg('Покупай ADA цена ' + str(signals_bnb.Close.iloc[-1]))
        elif int(signals_bnb.positions.iloc[-1]) == -1:
            Bot.send_msg("Продаем ADA, цена: " + str(signals_bnb.Close.iloc[-1]))

        time.sleep(895)

# fig = plt.figure()
# fig.set_size_inches(22.5, 10.5)
# plt.axhline(y=100, color='r', linestyle='-')
# plt.plot(np.squeeze(balance_arr))
# plt.show()

fig = plt.figure()
fig.set_size_inches(22.5, 10.5)
ax1 = fig.add_subplot(111, ylabel='Google price in $')
data_signal["Close"].plot(ax=ax1, color='g', lw=.5)
ts["short_mavg"].plot(ax=ax1, color='r', lw=2.)
ts["long_mavg"].plot(ax=ax1, color='b', lw=2.)

ax1.plot(ts.loc[ts.orders == 1.0].index, data_signal["Close"][ts.orders == 1.0],
         '^', markersize=7, color='k')

ax1.plot(ts.loc[ts.orders == -1.0].index, data_signal["Close"][ts.orders == -1.0],
         'v', markersize=7, color='k')

plt.legend(["Price", "Short mavg", "Long mavg", "Buy", "Sell"])
plt.title("Double Moving Average Trading Strategy 105/65 " + tickers)

plt.show()
