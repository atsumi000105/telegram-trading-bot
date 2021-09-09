import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance
import statistics as stats
import math as math

# load dataset
data_signal = pd.read_excel("C:\\Users\\Vlad\Desktop\\Finance\\AI labels.xlsx")
close = data_signal['Close']

#Simple moving average
time_period = 20 # number of days over which to average
history = [] # to track a history of prices
sma_values = [] # to track simple moving average values
for close_price in close:
  history.append(close_price)
  if len(history) > time_period: # we remove oldest price because we only average over last 'time_period' prices
    del (history[0])

  sma_values.append(stats.mean(history))

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(Simple20DayMovingAverage=pd.Series(sma_values, index=data_signal.index))

sma = data_signal['Simple20DayMovingAverage']

#Exponential moving average
num_periods = 20 # number of days over which to average
K = 2 / (num_periods + 1) # smoothing constant
ema_p = 0 #first observation

ema_values = [] # to hold computed EMA values
for close_price in close:
  if (ema_p == 0): # first observation, EMA = current-price
    ema_p = close_price
  else:
    ema_p = (close_price - ema_p) * K + ema_p

  ema_values.append(ema_p)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(Exponential20DayMovingAverage=pd.Series(ema_values, index=data_signal.index))

ema = data_signal['Exponential20DayMovingAverage']

#Absolute price oscillator

num_periods_fast = 10 # time period for the fast EMA
K_fast = 2 / (num_periods_fast + 1) # smoothing factor for fast EMA
ema_fast = 0
num_periods_slow = 40 # time period for slow EMA
K_slow = 2 / (num_periods_slow + 1) # smoothing factor for slow EMA
ema_slow = 0

ema_fast_values = [] # we will hold fast EMA values for visualization purposes
ema_slow_values = [] # we will hold slow EMA values for visualization purposes
apo_values = [] # track computed absolute price oscillator values
for close_price in close:
  if (ema_fast == 0): # first observation
    ema_fast = close_price
    ema_slow = close_price
  else:
    ema_fast = (close_price - ema_fast) * K_fast + ema_fast
    ema_slow = (close_price - ema_slow) * K_slow + ema_slow

  ema_fast_values.append(ema_fast)
  ema_slow_values.append(ema_slow)
  apo_values.append(ema_fast - ema_slow)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(FastExponential10DayMovingAverage=pd.Series(ema_fast_values, index=data_signal.index))
data_signal = data_signal.assign(SlowExponential40DayMovingAverage=pd.Series(ema_slow_values, index=data_signal.index))
data_signal = data_signal.assign(AbsolutePriceOscillator=pd.Series(apo_values, index=data_signal.index))

close_price = data_signal['ClosePrice']
ema_f = data_signal['FastExponential10DayMovingAverage']
ema_s = data_signal['SlowExponential40DayMovingAverage']
apo = data_signal['AbsolutePriceOscillator']

#Moving average convergence divergence
um_periods_fast = 10 # fast EMA time period
K_fast = 2 / (num_periods_fast + 1) # fast EMA smoothing factor
ema_fast = 0
num_periods_slow = 40 # slow EMA time period
K_slow = 2 / (num_periods_slow + 1) # slow EMA smoothing factor
ema_slow = 0
num_periods_macd = 20 # MACD EMA time period
K_macd = 2 / (num_periods_macd + 1) # MACD EMA smoothing factor
ema_macd = 0

ema_fast_values = [] # track fast EMA values for visualization purposes
ema_slow_values = [] # track slow EMA values for visualization purposes
macd_values = [] # track MACD values for visualization purposes
macd_signal_values = [] # MACD EMA values tracker
macd_historgram_values = [] # MACD - MACD-EMA
for close_price in close:
  if (ema_fast == 0): # first observation
    ema_fast = close_price
    ema_slow = close_price
  else:
    ema_fast = (close_price - ema_fast) * K_fast + ema_fast
    ema_slow = (close_price - ema_slow) * K_slow + ema_slow

  ema_fast_values.append(ema_fast)
  ema_slow_values.append(ema_slow)

  macd = ema_fast - ema_slow # MACD is fast_MA - slow_EMA
  if ema_macd == 0:
    ema_macd = macd
  else:
    ema_macd = (macd - ema_macd) * K_macd + ema_macd # signal is EMA of MACD values

  macd_values.append(macd)
  macd_signal_values.append(ema_macd)
  macd_historgram_values.append(macd - ema_macd)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(FastExponential10DayMovingAverage=pd.Series(ema_fast_values, index=data_signal.index))
data_signal = data_signal.assign(SlowExponential40DayMovingAverage=pd.Series(ema_slow_values, index=data_signal.index))
data_signal = data_signal.assign(MovingAverageConvergenceDivergence=pd.Series(macd_values, index=data_signal.index))
data_signal = data_signal.assign(Exponential20DayMovingAverageOfMACD=pd.Series(macd_signal_values, index=data_signal.index))
data_signal = data_signal.assign(MACDHistorgram=pd.Series(macd_historgram_values, index=data_signal.index))

close_price = data_signal['ClosePrice']
ema_f = data_signal['FastExponential10DayMovingAverage']
ema_s = data_signal['SlowExponential40DayMovingAverage']
macd = data_signal['MovingAverageConvergenceDivergence']
ema_macd = data_signal['Exponential20DayMovingAverageOfMACD']
macd_histogram = data_signal['MACDHistorgram']

#Bollinger bands

time_period = 20 # history length for Simple Moving Average for middle band
stdev_factor = 2 # Standard Deviation Scaling factor for the upper and lower bands
history = [] # price history for computing simple moving average
sma_values = [] # moving average of prices for visualization purposes
upper_band = [] # upper band values
lower_band = [] # lower band values

for close_price in close:
  history.append(close_price)
  if len(history) > time_period: # we only want to maintain at most 'time_period' number of price observations
    del (history[0])

  sma = stats.mean(history)
  sma_values.append(sma) # simple moving average or middle band
  variance = 0 # variance is the square of standard deviation
  for hist_price in history:
    variance = variance + ((hist_price - sma) ** 2)

  stdev = math.sqrt(variance / len(history)) # use square root to get standard deviation

  upper_band.append(sma + stdev_factor * stdev)
  lower_band.append(sma - stdev_factor * stdev)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(MiddleBollingerBand20DaySMA=pd.Series(sma_values, index=data_signal.index))
data_signal = data_signal.assign(UpperBollingerBand20DaySMA2StdevFactor=pd.Series(upper_band, index=data_signal.index))
data_signal = data_signal.assign(LowerBollingerBand20DaySMA2StdevFactor=pd.Series(lower_band, index=data_signal.index))

close_price = data_signal['ClosePrice']
mband = data_signal['MiddleBollingerBand20DaySMA']
uband = data_signal['UpperBollingerBand20DaySMA2StdevFactor']
lband = data_signal['LowerBollingerBand20DaySMA2StdevFactor']


#Relative strength indicator
time_period = 20 # look back period to compute gains & losses
gain_history = [] # history of gains over look back period (0 if no gain, magnitude of gain if gain)
loss_history = [] # history of losses over look back period (0 if no loss, magnitude of loss if loss)
avg_gain_values = [] # track avg gains for visualization purposes
avg_loss_values = [] # track avg losses for visualization purposes
rsi_values = [] # track computed RSI values
last_price = 0 # current_price - last_price > 0 => gain. current_price - last_price < 0 => loss.

for close_price in close:
  if last_price == 0:
    last_price = close_price

  gain_history.append(max(0, close_price - last_price))
  loss_history.append(max(0, last_price - close_price))
  last_price = close_price

  if len(gain_history) > time_period: # maximum observations is equal to lookback period
    del (gain_history[0])
    del (loss_history[0])

  avg_gain = stats.mean(gain_history) # average gain over lookback period
  avg_loss = stats.mean(loss_history) # average loss over lookback period

  avg_gain_values.append(avg_gain)
  avg_loss_values.append(avg_loss)

  rs = 0
  if avg_loss > 0: # to avoid division by 0, which is undefined
    rs = avg_gain / avg_loss

  rsi = 100 - (100 / (1 + rs))
  rsi_values.append(rsi)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(RelativeStrengthAvgGainOver20Days=pd.Series(avg_gain_values, index=data_signal.index))
data_signal = data_signal.assign(RelativeStrengthAvgLossOver20Days=pd.Series(avg_loss_values, index=data_signal.index))
data_signal = data_signal.assign(RelativeStrengthIndicatorOver20Days=pd.Series(rsi_values, index=data_signal.index))

close_price = data_signal['ClosePrice']
rs_gain = data_signal['RelativeStrengthAvgGainOver20Days']
rs_loss = data_signal['RelativeStrengthAvgLossOver20Days']
rsi = data_signal['RelativeStrengthIndicatorOver20Days']

#Momentum
time_period = 20 # how far to look back to find reference price to compute momentum
history = [] # history of observed prices to use in momentum calculation
mom_values = [] # track momentum values for visualization purposes

for close_price in close:
  history.append(close_price)
  if len(history) > time_period: # history is at most 'time_period' number of observations
    del (history[0])

  mom = close_price - history[0]
  mom_values.append(mom)

data_signal = data_signal.assign(ClosePrice=pd.Series(close, index=data_signal.index))
data_signal = data_signal.assign(MomentumFromPrice20DaysAgo=pd.Series(mom_values, index=data_signal.index))

close_price = data_signal['ClosePrice']
mom = data_signal['MomentumFromPrice20DaysAgo']


