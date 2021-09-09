from Algorithms import Double_avarage
import Get_data
import matplotlib.pyplot as plt
import time
from Telegram import Bot
from datetime import date
import Snippets




def check_performance(tickers, start_date):
    data_signal = Get_data.binance_data(tickers, start_date)
    signals = Double_avarage.double_moving_average(data_signal, 85, 45)
    print(Snippets.calculate_balance(signals))


today = date.today()

today = today.strftime("%Y-%m-%d")
# тут можешь изменять данные
start_date = '2021-08-01'  # начало периода
end_date = today  # конец периода
# tickers = "XLMBNB"  # название валюты ADABNB
ticker = ['XLMUSDT', 'BNBUSDT', 'ADAUSDT', "XLMBNB", "ADABNB"]
short_window = 85
long_window = 45
# short_window = 5
# long_window = 135


for tickers in ticker:
    check_performance(tickers, start_date)

# Bot.send_msg('Новые параметры стратегии Double MA: ' + str(short_window) + '/' + str(long_window))
# Bot.send_msg('Валюта: ' + str(short_window) + '/' + str(long_window) + ". BNB пары не брались в расчет.")
while True:
    for tickers in ticker:
        data_signal = Get_data.binance_data(tickers, start_date)
        signals = Double_avarage.double_moving_average(data_signal, short_window, long_window)
        data_signal['positions'] = signals['positions']
        print(tickers, int(data_signal.positions.iloc[-1]))
        if int(data_signal.positions.iloc[-1]) == 1:
            Bot.send_msg('Покупай ' + str(tickers) + " цена: " + str(data_signal.Close.iloc[-1]))
        elif int(data_signal.positions.iloc[-1]) == -1:
            Bot.send_msg('Наверное лучше продать сейчас ' + str(tickers) + " цена: " + str(data_signal.Close.iloc[-1]))
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
