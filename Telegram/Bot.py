
import os
import time

from dotenv import load_dotenv
#from ML_finance import Price_prediction
import telebot

#import Get_data
#from Algorithms import Fibonacci
#import Main



load_dotenv()
# token = os.getenv('TELEGRAM_TOKEN')
# bot2 = telebot.TeleBot(token)

token2 = os.getenv('TELEGRAM_TOKEN2')
bot2 = telebot.TeleBot(token2)

def send_msg(text, id):
    while True:
        try:
            bot2.send_message(id, text)
            break
        except:
            print('Unable to send msg in telegram')
            time.sleep(5)
            continue
    #-467554548
    #-565150126

@bot2.message_handler(commands=['start'])
def start(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = telebot.types.KeyboardButton("Алгоритмы")
    markup.add(item1)

    bot2.send_message(message.chat.id, "{0.first_name}, шо надо ?".format(message.from_user), reply_markup=markup)


@bot2.message_handler(content_types=['text'])
def bot_message(message):
    if message.chat.type == 'Алгоритмыs':
        if message.text == 'RandomChislo':
            bot2.send_message(message.chat.id, 'Resultat tut')
    elif message.text == 'Алгоритмы':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Фибоначи + MACD")
        item2 = telebot.types.KeyboardButton("Ввести параметры")
        item3 = telebot.types.KeyboardButton("Двойная Средняя")
        item4 = telebot.types.KeyboardButton("ML ADAboost")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, item3, item4, back)

        bot2.send_message(message.chat.id, "Хороший выбор", reply_markup=markup)


    elif message.text == 'Фибоначи + MACD':
        f = open('ticker.txt', 'r')
        ticker = str(f.read())
        ticker = ticker.upper()

        f = open('date.txt', 'r')
        start_date = str(f.read())


        bot2.send_message(message.chat.id, "Запускаю фибоначи для " + ticker + " начиная с " + start_date)
        data = Get_data.binance_data(ticker, start_date)
        bot2.send_message(message.chat.id, "Профит " + str(float(Fibonacci.main(data))-100) + " $")

        if data.size > 0:
            bot2.send_photo(message.chat.id,
                                      photo=open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Plots\Fig1.png',
                                                 'rb'))

    elif message.text == 'Двойная Средняя':
        Main.algo_strat2()

    elif message.text == 'ML ADAboost':
        Price_prediction.ada_AB()

    elif message.text == 'Ввести параметры':

        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        bot2.send_message(message.chat.id, "Выбери тип данных", reply_markup=markup)

    elif message.text == 'Стартовая дата':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        msg = bot2.send_message(message.chat.id, "Стартовая дата")
        bot2.register_next_step_handler(msg, save_date)

        bot2.send_message(message.chat.id, "Введи дату в формате 2000-01-30", reply_markup=markup)

    elif message.text == 'Торгующая пара':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        msg = bot2.send_message(message.chat.id, "Торгующая пара")
        bot2.register_next_step_handler(msg, save_ticker)

        bot2.send_message(message.chat.id, "Какой парой торгуем ?", reply_markup=markup)


    elif message.text == 'Назад':
        start(message)

def save_ticker(message):
    open('ticker.txt', 'w').write(message.text)
    bot2.send_message(message.chat.id, "Запомнил")
    print(message.text, ' was added')

def save_date(message):
    open('date.txt', 'w').write(message.text)
    bot2.send_message(message.chat.id, "Запомнил")
    print(message.text, ' was added')


# @bot2.message_handler(func=lambda m: True)
# def send_msg(message, text):
#     bot2.send_message(message.chat.id, text)


def main():
    bot2.infinity_polling(timeout=10, long_polling_timeout = 5)


if __name__ == '__main__':
    main()
