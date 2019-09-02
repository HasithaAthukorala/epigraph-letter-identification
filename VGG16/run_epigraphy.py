import cv2
import os
import shutil 
import os.path
from os.path import isfile, join
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')



classifier_path = './models/vgg16_model.h5'
classifier = load_model(classifier_path)
#classifier.summary()
model = VGG16(weights='imagenet', include_top=False )

#video_directory = './videos/'
img_directory = './images/'
num = 0
count = 0
#################################################
Letter_a = os.path.abspath('./gallery/Letter_a')
Letter_CannotRead = os.path.abspath('./gallery/Letter_CannotRead')
Letter_da = os.path.abspath('./gallery/Letter_da')
Letter_e = os.path.abspath('./gallery/Letter_e')
Letter_ga = os.path.abspath('./gallery/Letter_ga')
Letter_ha = os.path.abspath('./gallery/Letter_ha')
Letter_la = os.path.abspath('./gallery/Letter_la')
Letter_sha = os.path.abspath('./gallery/Letter_sha')
Letter_ta = os.path.abspath('./gallery/Letter_ta')
Letter_va = os.path.abspath('./gallery/Letter_va')

#########################################################

def predict(file):
    global model
    x = image.load_img(file, target_size=(150,150))
    x = image.img_to_array(x)
    x = x/255
    x = np.expand_dims(x, axis=0) 
    features = model.predict(x)
    result = classifier.predict_classes(features)
    if result[0] == 0:
        prediction = 'Letter_a'
    elif result[0] == 1:
        prediction = 'Letter_CannotRead'
    elif result[0] == 2:
        prediction = 'Letter_da'
    elif result[0] == 3:
        prediction = 'Letter_e'
    elif result[0] == 4:
        prediction = 'Letter_ga'
    elif result[0] == 5:
        prediction = 'Letter_ha'
    elif result[0] == 6:
        prediction = 'Letter_la'
    elif result[0] == 7:
        prediction = 'Letter_sha'
    elif result[0] == 8:
        prediction = 'Letter_ta'
    elif result[0] == 9:
        prediction = 'Letter_va'
    return prediction

list_img = os.listdir(img_directory)
    
for img in list_img:
    temp = predict(img_directory+img)
    if temp == 'Letter_a':
        shutil.move(os.path.abspath(img_directory+img),Letter_a)
    elif temp == 'Letter_CannotRead':
        shutil.move(os.path.abspath(img_directory+img),Letter_CannotRead)
    elif temp == 'Letter_da':
        shutil.move(os.path.abspath(img_directory+img),Letter_da)
    elif temp == 'Letter_e':
        shutil.move(os.path.abspath(img_directory+img),Letter_e)
    elif temp == 'Letter_ga':
        shutil.move(os.path.abspath(img_directory+img),Letter_ga)
    elif temp == 'Letter_ha':
        shutil.move(os.path.abspath(img_directory+img),Letter_ha)
    elif temp == 'Letter_la':
        shutil.move(os.path.abspath(img_directory+img),Letter_la)
    elif temp == 'Letter_sha':
        shutil.move(os.path.abspath(img_directory+img),Letter_sha)
    elif temp == 'Letter_ta':
        shutil.move(os.path.abspath(img_directory+img),Letter_ta)
    elif temp == 'Letter_va':
        shutil.move(os.path.abspath(img_directory+img),Letter_va)
        
   





