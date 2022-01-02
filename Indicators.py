import pandas as pd
import bottleneck as bn
import statistics as stats
import math as math
import numpy as np
import numba as nb

# Simple moving average
def calc_SMA(price_date, time_period):
    return

def SMA(price_date, time_period):

    return bn.move_mean(price_date, window=time_period)

def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC
# Exponential moving average
def EMA(price_date, time_period):
    alpha = 2 / (time_period + 1.0)
    n = price_date.shape[0]
    scale_arr = (1 - alpha) ** (-1 * np.arange(n))
    weights = (1 - alpha) ** np.arange(n)
    pw0 = (1 - alpha) ** (n - 1)
    mult = price_date * pw0 * scale_arr
    cumsums = mult.cumsum()
    return cumsums * scale_arr[::-1] / weights.cumsum()


# Absolute price oscillator
def APO(price_data=None, slow=None, fast=None, ema_fast=None, ema_slow=None):
    if price_data:
        ema_fast = EMA(price_data, fast)
        ema_slow = EMA(price_data, slow)
    return np.subtract(ema_fast, ema_slow)


# Moving average convergence divergence
def MACD(price_data=None, slow=None, fast=None, sma_slow=None, sma_fast=None):
    if price_data:
        sma_fast = SMA(price_data, fast)
        sma_slow = SMA(price_data, slow)

    return np.subtract(sma_fast, sma_slow)

# Bollinger bands
def BolBands(price_data=None, time_period=20, stdev_factor=2):
    sma_values = SMA(price_data, time_period)
    return sma_values + stdev_factor * np.std(price_data), sma_values - stdev_factor * np.std(price_data)


# Relative strength indicator
def RSI(price_data, time_period = 10):
    @nb.jit(fastmath=True, nopython=True)
    def calc_rsi(array, deltas, avg_gain, avg_loss, n):

        # Use Wilder smoothing method
        up = lambda x: x if x > 0 else 0
        down = lambda x: -x if x < 0 else 0
        i = n + 1
        for d in deltas[n + 1:]:
            avg_gain = ((avg_gain * (n - 1)) + up(d)) / n
            avg_loss = ((avg_loss * (n - 1)) + down(d)) / n
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                array[i] = 100 - (100 / (1 + rs))
            else:
                array[i] = 100
            i += 1

        return array

    def get_rsi(array, n):

        deltas = np.append([0], np.diff(array))

        avg_gain = np.sum(deltas[1:n + 1].clip(min=0)) / n
        avg_loss = -np.sum(deltas[1:n + 1].clip(max=0)) / n

        array = np.empty(deltas.shape[0])
        array.fill(np.nan)

        return calc_rsi(array, deltas, avg_gain, avg_loss, n)

    return get_rsi(price_data, n=time_period)

# Momentum
def MOM():

    mom = close_price - history[0]
    mom_values.append(mom)


def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK


def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD


