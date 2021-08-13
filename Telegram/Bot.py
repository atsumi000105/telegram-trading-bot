import requests
import telebot


API_KEY = '1922104439:AAH8O2khlY_iV9xUP9UnbroYVOQHnctNfqo'
bot = telebot.TeleBot(API_KEY)


def send_msg(text):
  token = '1922104439:AAH8O2khlY_iV9xUP9UnbroYVOQHnctNfqo'
  chat_id = "-467554548"
  url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
  results = requests.get(url_req)
  print(results.json())




# bot.send_message('', "Buy")
# bot.send_message()

