
import os
from dotenv import load_dotenv

import telebot

import Get_data
from Algorithms import Fibonacci




load_dotenv()
token = os.getenv('TELEGRAM_TOKEN')

bot = telebot.TeleBot(token)

def send_msg(text):
    bot.send_message(376012018, text)


@bot.message_handler(commands=['start'])
def start(message):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = telebot.types.KeyboardButton("Алгоритмы")
    markup.add(item1)

    bot.send_message(message.chat.id, str(message.chat.id) + "{0.first_name}, шо надо ?".format(message.from_user), reply_markup=markup)


@bot.message_handler(content_types=['text'])
def bot_message(message):
    if message.chat.type == 'Алгоритмыs':
        if message.text == 'RandomChislo':
            bot.send_message(message.chat.id, 'Resultat tut')
    elif message.text == 'Алгоритмы':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Фибоначи + MACD")
        item2 = telebot.types.KeyboardButton("Ввести параметры")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        bot.send_message(message.chat.id, "Хороший выбор", reply_markup=markup)


    elif message.text == 'Фибоначи + MACD':
        f = open('ticker.txt', 'r')
        ticker = str(f.read())
        ticker = ticker.upper()

        f = open('date.txt', 'r')
        start_date = str(f.read())


        bot.send_message(message.chat.id, "Запускаю фибоначи для " + ticker + " начиная с " + start_date)
        data = Get_data.binance_data(ticker, start_date)
        bot.send_message(message.chat.id, "Профит " + str(float(Fibonacci.main(data))-100) + " $")

        if data.size > 0:
            bot.send_photo(message.chat.id,
                                      photo=open(r'C:\Users\Vlad\PycharmProjects\Time-Series-Analysis\Plots\Fig1.png',
                                                 'rb'))

    elif message.text == 'Ввести параметры':

        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        bot.send_message(message.chat.id, "Выбери тип данных", reply_markup=markup)

    elif message.text == 'Стартовая дата':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        msg = bot.send_message(message.chat.id, "Стартовая дата")
        bot.register_next_step_handler(msg, save_date)

        bot.send_message(message.chat.id, "Введи дату в формате 2000-01-30", reply_markup=markup)

    elif message.text == 'Торгующая пара':
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = telebot.types.KeyboardButton("Стартовая дата")
        item2 = telebot.types.KeyboardButton("Торгующая пара")
        back = telebot.types.KeyboardButton("Назад")
        markup.add(item1, item2, back)

        msg = bot.send_message(message.chat.id, "Торгующая пара")
        bot.register_next_step_handler(msg, save_ticker)

        bot.send_message(message.chat.id, "Какой парой торгуем ?", reply_markup=markup)


    elif message.text == 'Назад':
        start(message)

def save_ticker(message):
    open('ticker.txt', 'w').write(message.text)
    bot.send_message(message.chat.id, "Запомнил")
    print(message.text, ' was added')

def save_date(message):
    open('date.txt', 'w').write(message.text)
    bot.send_message(message.chat.id, "Запомнил")
    print(message.text, ' was added')


@bot.message_handler(func=lambda m: True)
def repeat(message):
    bot.send_message(message.chat.id, message.text)


def main():
    bot.polling()


if __name__ == '__main__':
    main()
