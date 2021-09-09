import matplotlib.pyplot as plt
import numpy as np
import Get_data
import Snippets
# start_date = '2021-08-16'
# end_date = '2021-08-20'
ticker = ['BNBUSDT', 'ADAUSDT']


#data_signal = Get_data.binance_data('ADAUSDT', start_date, )
def main(data_signal):
    price = data_signal["Close"]

    price_min = min(price)
    price_max = max(price)

    # Fibonacci Levels considering original trend as upward move
    diff = price_max - price_min
    level1 = np.round(price_max - 0.236 * diff, 2)
    level2 = np.round(price_max - 0.382 * diff, 2)
    level3 = np.round(price_max - 0.618 * diff, 2)
    level4 = np.round(price_max - 0.786 * diff, 2)

    # print("Level \t Price")
    # print(f"0% \t {price_max}")
    # print(f"23.6% \t {level1}")
    # print(f"38.2% \t {level2}")
    # print(f"61.8% \t {level3}")
    # print(f"78.6% \t {level4}")
    # print(f"100% \t {price_min}")

    # Calulating the MACD Line and the Signal Line indicator
    # Calculate tthe Short Term Exponential Moving Average
    ShortEMA = data_signal.Close.ewm(span=12, adjust=False).mean()
    # Calculate the Long Term Eponenatial
    LongEMA = data_signal.Close.ewm(span=28, adjust=False).mean()
    # Calculate the MACD
    MACD = ShortEMA - LongEMA
    # calculate the Signal line
    signal = MACD.ewm(span=10, adjust=False).mean()

    # Create new columns for the df
    data_signal['MACD'] = MACD
    data_signal['Signal line'] = signal

    # create a function to be eused in our strategy to get the upper and lower Fibonaci level of current price
    def getlevels(price):
        if price > - level1:
            return (price_max, level1)
        elif price > - level2:
            return (level1, level2)
        elif price > - level3:
            return (level2, level3)
        elif price > - level4:
            return (level3, level4)
        else:
            return (level4, price_min)

    # Create a function for the trading strategy

    # Strategy When the signal line crosses above the MACD Line and the current price crossed above or below the last
    # Fibonaci level - buy IF the signal line croses below the MACD line and the current price crossed above or below
    # the last Fibonacci level then sell Never sell at a price lower that i bought

    def strategy(df):
        buy_list = []
        sell_list = []
        possitions = []
        flag = 0
        last_buy_price = 0

        # Loop through the data set
        for i in range(0, df.shape[0]):
            price = df['Close'][i]
            # If this is the first data point within the data set, then get the level above and below it
            if i == 0:
                upper_lvl, lower_lvl = getlevels(price)
                buy_list.append(np.nan)
                sell_list.append(np.nan)
                possitions.append(0)
            # Else if the current price greater then or equel to the upper level or less then or equal to the lower
            # level, then we know the price 'hit' or crossed a new Fibonaci level

            elif price >= upper_lvl or price <= lower_lvl:
                # check to see if the MACD line crossed above or below the signal line
                if df['Signal line'][i] > df['MACD'][i] and flag == 0:
                    last_buy_price = price
                    possitions.append(1)
                    buy_list.append(price)
                    sell_list.append(np.nan)
                    # set the flag to 1 to signal that share was bough

                    flag = 1
                    # todo check if i need the last condition with last_but_price
                #elif df['Signal line'][i] < df['MACD'][i] and flag == 1 and price >= last_buy_price:
                elif df['Signal line'][i] < df['MACD'][i] and flag == 1:
                    buy_list.append(np.nan)
                    sell_list.append(price)
                    possitions.append(-1)
                    # set the flag to 0 to signal that share was sold
                    flag = 0
                else:
                    buy_list.append(np.nan)
                    sell_list.append(np.nan)
                    possitions.append(0)
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)
                possitions.append(0)

            # update new levels
            upper_lvl, lower_lvl = getlevels(price)
        return buy_list, sell_list, possitions

    buy, sell, possitions = strategy(data_signal)
    data_signal['Buy_Signal_Price'] = buy
    data_signal['Sell_Signal_Price'] = sell
    data_signal['positions'] = possitions
    # Plot the Fibonacci levels along with the close price and the MACD and Signal Line

    # Plot the Fibonacci Levels
    fig = plt.figure()
    fig.set_size_inches(22.5, 10.5)
    plt.plot(data_signal.index, data_signal.Close)
    plt.scatter(data_signal.index, data_signal['Buy_Signal_Price'], color='green', marker='^', alpha=1)
    plt.scatter(data_signal.index, data_signal['Sell_Signal_Price'], color='red', marker='v', alpha=1)
    plt.axhline(price_max, linestyle='--', alpha=0.5, color='red')
    plt.axhline(level1, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(level2, linestyle='--', alpha=0.5, color='yellow')
    plt.axhline(level3, linestyle='--', alpha=0.5, color='green')
    plt.axhline(level4, linestyle='--', alpha=0.5, color='blue')
    plt.axhline(price_min, linestyle='--', alpha=0.5, color='purple')
    plt.ylabel("Close Price in USD")
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    plt.savefig(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Plots\Fig1.png'
                )

    plt.show()

#   print(Snippets.calculate_balance(data_signal))
    return Snippets.calculate_balance(data_signal)

