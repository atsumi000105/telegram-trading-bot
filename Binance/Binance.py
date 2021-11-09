from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()
apikey = os.getenv('apikey')
secret = os.getenv('secret')
client = Client(apikey, secret)


class Binance_acc:

    def __init__(self):
        pass
        #self.account =

    def get_decimal(self, ticker):
        info = client.get_symbol_info(ticker + "USDT")
        val = info['filters'][2]['stepSize']
        decimal = 0
        is_dec = False
        for c in val:
            if is_dec is True:
                decimal += 1
            if c == '1':
                break
            if c == '.':
                is_dec = True
        return decimal


    def get_acc_balance(self):
        return float(client.get_asset_balance('USDT')['free'])

    def buy(self, symbol, quantity):
        client.order_market_buy(symbol=symbol, quantity=quantity)
        print(symbol, quantity, " Was purchased")

    def sell(self, symbol):
        acc = client.get_account()
        assets = acc['balances']
        for acc in assets:
            if acc['asset'] == symbol:
                qty = float(acc['free'][:self.get_decimal(symbol) + 2])
                print('this ', qty, self.get_decimal(symbol))
                client.order_market_sell(symbol=symbol + "USDT", quantity=qty)
                print(symbol, " Sold")

    def sell_all(self):

        acc = client.get_account()
        assets = acc['balances']
        for acc in assets:
            if acc['asset'] == symbol:
                print('stock ', acc['free'])
                qty = acc['free'][:self.get_decimal(symbol) + 2]

        client.order_market_sell(symbol=symbol + "USDT", quantity=qty)