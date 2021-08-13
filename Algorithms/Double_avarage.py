import pandas as pd
import numpy as np


def double_moving_average(financial_data, short_window, long_window):
    signals = pd.DataFrame(index=financial_data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = financial_data['Close']. \
        rolling(window=short_window,
                min_periods=1, center=False).mean()
    signals['long_mavg'] = financial_data['Close']. \
        rolling(window=long_window,
                min_periods=1, center=False).mean()
    signals['signal'][short_window:] = \
        np.where(signals['short_mavg'][short_window:]
                 > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals


