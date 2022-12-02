from telegram.ext import Updater,Filters, CommandHandler, MessageHandler
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy

def process_and_predict(file):
    im = Image.open(file)
    width, height = im.size
    if width == height:
        im = im.resize((256,256), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((256,256), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((256,256), Image.ANTIALIAS)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 256, 256, 3)

    
    
    #maxnum = np.argmax(model.predict(ar))
    arr = model.predict(ar)[0][:15]
    #print(arr)
    sort_arr = np.sort(arr)
    first = sort_arr[-1]
    second = sort_arr[-2]
    #print(first,second)
    summ = 0
    for i in arr:
        if i == first:
            num1 = summ
        elif i == second:
            num2 = summ
        summ+=1
    #print(num1,num2)
    
    
    #print('Most likely the breed of this cat is a ' + str(ddict[num1]) )
    #print('Less likely the breed of this cat is a ' + str(ddict[num2]))
    
    return num1,num2#,ddict[num2]





model = keras.models.load_model('1st')



def start(updater,context):
    updater.message.reply_text("Hiii. Welcom to the cat breeds classifier bot! \nMy accuracy is 81%, but I still mistaken sometimes)))So don't be hard on me. \n I'm happy to see you here :)")

def help_(updater,context):
    updater.message.reply_text("I have only 15 cat breeds in my list yet(( Here they are: \n"
                               "Abyssinian \n"
                               "Bengal \n"
                               "Bombay \n"
                               "British_Shorthair \n"
                               "Exotic_Shorthair \n"
                               "Maine_Coon \n"
                               "Munchkin \n"
                               "Persian \n"
                               "Ragdoll \n"
                               "Russian_Blue \n"
                               "Scottish_Fold \n"
                               "Siamese \n"
                               "Siberian \n"
                               "Sphynx \n"
                               "Turkish_Angora \n"
                               "If you want to identify your cat or some pics from th internet just send me a photo")

def message (updater,context):
    msg = updater.message.text
    updater.message.reply_text("Sorry, I don't understand your language")

def image (updater,context):
    photo = updater.message.photo[-1].get_file()
    photo.download("img.jpg")

    
    d = {0:'Abyssinian',1:'Bengal',2:'Bombay',3:'British_Shorthair',4:'Exotic_Shorthair',5:'Maine_Coon',6:'Munchkin',7:'Persian',8:'Ragdoll',9:'Russian_Blue',
        10:'Scottish_Fold',11:'Siamese',12:'Siberian',13:'Sphynx',14:'Turkish_Angora'}


    pred1,pred2 = process_and_predict("img.jpg")
    updater.message.reply_text("Most likely the breed of this cat is a " + str(d[pred1]) +'\n'
                               "Less likely the breed of this cat is a " + str(d[pred2]))


                               



updater = Updater("5761936624:AAGHCW1KGy-mVv2Bwkgyh89r77hrMRvXmic")
dispatcher = updater.dispatcher


dispatcher.add_handler(CommandHandler("start",start))
dispatcher.add_handler(CommandHandler("help",help_))
dispatcher.add_handler(MessageHandler(Filters.text,message))
dispatcher.add_handler(MessageHandler(Filters.photo,image))

updater.start_polling()
##updater.idle()
