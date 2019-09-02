#!/usr/bin/python
# -*- coding: windows-1251 -*-

import config
import telebot
import time
import openCVTest as IW
import os
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
mplLogger = logging.getLogger("matplotlib")
mplLogger.setLevel(logging.WARNING)

if config.use_proxy:
    telebot.apihelper.proxy = {'https': 'socks5h://{}:{}@{}:{}'.format(config.proxy_user,
                                                                      config.proxy_pass,
                                                                      config.proxy_address,
                                                                      config.proxy_port)}

bot = telebot.TeleBot(config.token)
cur_ticket = ""
tickets = os.listdir(config.ticket_directory)
adding_ticket = 0
file_to_write = ''
deleting_ticket = 0
sending_now = False


@bot.message_handler(commands=["start"])
def send_welcome(message):
    logging.debug('start command incoming')
    bot.reply_to(message, "I accept slides wia photo and return it with more compatible way")

@bot.message_handler(content_types=["photo"])
def send_ticket(message):
    logging.debug("got photo")
    fileID = message.photo[-1].file_id
    logging.debug(message.photo[-1])
    file_info = bot.get_file(fileID)
    bytes = bot.download_file(file_info.file_path)
    slide = np.array(Image.open(io.BytesIO(bytes)))
    result = IW.cut_image(slide)
    logging.debug("got res")
    res_byte = io.BytesIO()
    Image.fromarray(result).save(res_byte, format="JPEG")
    bot.send_photo(message.chat.id, res_byte.getvalue())
    time.sleep(1)

#
# @bot.message_handler(content_types=["text"])
# def send_ticket(message):
#     bot.reply_to(message, "Unexpected text")



if __name__ == '__main__':
     bot.polling(none_stop=True)
