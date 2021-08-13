import requests
import telebot


API_KEY =
bot = telebot.TeleBot(API_KEY)


def send_msg(text):
  token =
  chat_id =
  url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
  results = requests.get(url_req)
  print(results.json())




# bot.send_message('', "Buy")
# bot.send_message()

