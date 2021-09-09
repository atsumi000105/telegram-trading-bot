# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))



def calculate_balance(data, commission=0.001, array=False):
    balance = 100
    comm_buy = 0
    comm_sell = 0
    #print("Изначальный бюджет", balance, "$")
    price_buy = 0
    price_sell = 0
    balance_arr = []
    for i in range(len(data)):
        if data.positions.iloc[i] == 1:
            price_buy = float(data.Close.iloc[i])
            comm_buy = balance*commission
        elif data.positions.iloc[i] == -1 and price_buy != 0:
            price_sell = float(data.Close.iloc[i])
            balance = balance*price_sell/price_buy*(1-commission)-comm_buy
        balance_arr.append(balance)
    if array:
        return balance_arr
    else:
        return round(balance, 2)

