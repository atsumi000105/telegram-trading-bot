import pandas as pd
import numpy as np
import Snippets
import Get_data
from datetime import datetime, timedelta

start_date = '2021-03-01'  # начало периода #1 Jun 2021
# end_date = '2021-08-13'  # конец периода
start_date_dtm = datetime.strptime(start_date, '%Y-%m-%d')
ticker = ['BNBUSDT', 'ADAUSDT']


bnb = Get_data.binance_data('BNBUSDT', start_date)
ada = Get_data.binance_data('ADAUSDT', start_date)

bnb.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\BNB 15 min.csv")
ada.to_csv("C:\\Users\\Vlad\Desktop\\Finance\\ADA 15 min.csv")

